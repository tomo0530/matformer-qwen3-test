import os
import torch
import torch.nn as nn
import numpy as np
import gc
import json
import random
from tqdm.auto import tqdm
from safetensors import safe_open
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils import set_seed

# MatQwenMLPクラスの定義
class MatQwenMLP(nn.Module):
    """
    Qwen3のFFN(MLP)をMatFormerアーキテクチャに変更するカスタムクラス。
    """
    def __init__(self, config, ffn_dim):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, ffn_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        
        self.full_intermediate_size = ffn_dim
        self.current_intermediate_size = ffn_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        active_gate_weight = self.gate_proj.weight[:self.current_intermediate_size, :]
        active_up_weight = self.up_proj.weight[:self.current_intermediate_size, :]
        active_down_weight = self.down_proj.weight[:, :self.current_intermediate_size]

        if x.dtype != active_gate_weight.dtype:
            active_gate_weight = active_gate_weight.to(x.dtype)
            active_up_weight = active_up_weight.to(x.dtype)
            active_down_weight = active_down_weight.to(x.dtype)

        gate_output = nn.functional.linear(x, active_gate_weight)
        up_output = nn.functional.linear(x, active_up_weight)
        activated_output = self.act_fn(gate_output) * up_output
        output = nn.functional.linear(activated_output, active_down_weight, bias=None)
        
        return output

# モデルローダー関数
def load_mix_n_match_model_strictly(model_path, device):
    """Mix-n-Matchモデルを厳密にロード"""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    ffn_dims_path = os.path.join(model_path, "ffn_dims.json")
    with open(ffn_dims_path, 'r') as f:
        ffn_info = json.load(f)
    ffn_dims_per_layer = ffn_info['ffn_dims_per_layer']
    
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    for i, layer in enumerate(model.model.layers):
        layer.mlp = MatQwenMLP(config, ffn_dims_per_layer[i])
    
    model = model.to_empty(device=device)

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path, 'r') as f:
        index = json.load(f)

    shard_files = sorted(list(set(index['weight_map'].values())))

    for shard_file in tqdm(shard_files, desc="Loading shards"):
        shard_path = os.path.join(model_path, shard_file)
        
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for tensor_name in f.keys():
                saved_tensor = f.get_tensor(tensor_name)
                
                try:
                    param = model.get_parameter(tensor_name)
                except AttributeError:
                    continue

                if saved_tensor.shape != param.data.shape:
                    slices = tuple(slice(0, dim) for dim in saved_tensor.shape)
                    with torch.no_grad():
                        param.data[slices].copy_(saved_tensor.to(device, non_blocking=True))
                else:
                    with torch.no_grad():
                        param.data.copy_(saved_tensor.to(device, non_blocking=True))
                
                del saved_tensor
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model.eval()
    model.config.use_cache = True
    
    return model

# サブネットワーク設定関数
def configure_subnetwork_globally(model, flag: str, scale_factors):
    """モデル全体のサブネットワークサイズを設定"""
    if flag not in scale_factors:
        raise ValueError(f"無効なフラグ '{flag}' です。")

    target_size = scale_factors[flag]
    
    if hasattr(model, 'module'):
        actual_model = model.module
    else:
        actual_model = model
    
    for layer in actual_model.model.layers:
        layer.mlp.current_intermediate_size = min(layer.mlp.full_intermediate_size, target_size)

def add_configure_method(model, scale_factors):
    """モデルにconfigure_subnetworkメソッドを追加"""
    def configure_method(flag):
        configure_subnetwork_globally(model, flag, scale_factors)
    
    if hasattr(model, 'module'):
        model.module.configure_subnetwork = configure_method
    else:
        model.configure_subnetwork = configure_method

# カスタムデータコレーター
class MatFormerDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=False):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.flags = ['s', 'l', 'xl']

    def __call__(self, examples):
        batch = super().__call__(examples)
        flag = random.choice(self.flags)
        batch['flag'] = flag
        return batch

def main():
    # 設定
    local_output_path = "./output/qwen3_3b_model"
    final_output_path = "./output/matformer_qwen3_3b_finetuned"
    model_4b_id = "Qwen/Qwen3-4B"
    
    # スケールファクター
    scale_factors = {
        's': 8192,
        'm': 9728,
        'l': 11776,
        'xl': 13312
    }
    
    # Acceleratorの初期化
    accelerator = Accelerator()
    
    # シード設定
    set_seed(42)
    
    accelerator.print(f"Device: {accelerator.device}")
    accelerator.print(f"Distributed: {accelerator.distributed_type}")
    accelerator.print(f"Num processes: {accelerator.num_processes}")
    accelerator.print(f"Gradient accumulation steps: {accelerator.gradient_accumulation_steps}")
    
    # トークナイザーの読み込み
    tokenizer = AutoTokenizer.from_pretrained(model_4b_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # モデルの読み込み
    accelerator.print("Loading model...")
    model = load_mix_n_match_model_strictly(local_output_path, "cpu")
    add_configure_method(model, scale_factors)
    
    # Gradient checkpointing
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    
    # データセットの準備
    accelerator.print("Preparing dataset...")
    dataset = load_dataset("Abirate/english_quotes", split="train")
    dataset = dataset.shuffle(seed=42).select(range(1000))
    
    def tokenize_function(examples):
        return tokenizer(examples["quote"], truncation=True, max_length=64, padding="max_length")
    
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["quote", "author", "tags"]
    )
    
    # データローダーの作成
    data_collator = MatFormerDataCollator(tokenizer=tokenizer)
    train_dataloader = torch.utils.data.DataLoader(
        tokenized_dataset,
        batch_size=1,
        collate_fn=data_collator,
        shuffle=True,
    )
    
    # オプティマイザーとスケジューラー
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=5e-6,
        weight_decay=0.01,
        eps=1e-6,
        betas=(0.9, 0.95)
    )
    
    num_epochs = 1
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.5,
        total_iters=num_training_steps
    )
    
    # Acceleratorで準備
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    # トレーニング
    accelerator.print("Starting training...")
    model.train()
    total_loss = 0
    completed_steps = 0
    
    progress_bar = tqdm(
        range(num_training_steps), 
        disable=not accelerator.is_local_main_process,
        desc="Training"
    )
    
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            flag = batch.pop("flag", None)
            if flag is not None:
                unwrapped_model = accelerator.unwrap_model(model)
                if hasattr(unwrapped_model, 'configure_subnetwork'):
                    unwrapped_model.configure_subnetwork(flag)
            
            # DeepSpeedと互換性のある勾配累積の実装
            outputs = model(**batch)
            loss = outputs.loss
            
            # 勾配累積のために損失をスケール
            loss = loss / accelerator.gradient_accumulation_steps
            
            # バックワード
            accelerator.backward(loss)
            
            # 勾配累積が完了したらオプティマイザーをステップ
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # 統計情報の更新
            if accelerator.sync_gradients:
                total_loss += loss.detach().float() * accelerator.gradient_accumulation_steps
                completed_steps += 1
                
                if completed_steps % 10 == 0:
                    accelerator.print(f"Steps {completed_steps}/{num_training_steps} - Loss: {loss.item() * accelerator.gradient_accumulation_steps:.4f}")
                
                progress_bar.update(1)
                
                if completed_steps >= 100:  # デモ用の早期終了
                    break
    
    progress_bar.close()
    
    avg_loss = total_loss / completed_steps if completed_steps > 0 else 0
    accelerator.print(f"\nTraining completed! Average loss: {avg_loss:.4f}")
    
    # モデルの保存（メインプロセスのみ）
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.eval()
        
        if hasattr(unwrapped_model, 'gradient_checkpointing_disable'):
            unwrapped_model.gradient_checkpointing_disable()
        unwrapped_model.config.use_cache = True
        
        # ディレクトリ作成
        os.makedirs(final_output_path, exist_ok=True)
        
        unwrapped_model.save_pretrained(final_output_path)
        tokenizer.save_pretrained(final_output_path)
        
        # FFN情報の読み込みと保存
        with open(os.path.join(local_output_path, "ffn_dims.json"), 'r') as f:
            ffn_info = json.load(f)
        
        ffn_info_path = os.path.join(final_output_path, "ffn_dims.json")
        with open(ffn_info_path, "w") as f:
            json.dump({
                "ffn_dims_per_layer": ffn_info['ffn_dims_per_layer'],
                "source_intermediate_size": ffn_info['source_intermediate_size'],
                "method": "mix-n-match",
                "scale_factors": scale_factors,
                "fine_tuned": True
            }, f, indent=2)
        
        accelerator.print(f"✅ Model saved to {final_output_path}")

if __name__ == "__main__":
    main()