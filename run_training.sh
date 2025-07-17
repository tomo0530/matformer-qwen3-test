#!/bin/bash

# 環境変数の設定
export CUDA_VISIBLE_DEVICES=1,2,3

# 作業ディレクトリに移動
cd /home/tomotaka.harada/matformer-qwen3-test

# Accelerate設定ファイルのパスを指定
export ACCELERATE_CONFIG_FILE=./accelerate_config.yaml

# 設定の確認（オプション）
echo "=== Accelerate Configuration ==="
poetry run accelerate env
echo "================================"

# Accelerateで分散学習を実行
# --config_file: 設定ファイルを明示的に指定
poetry run accelerate launch \
    --config_file ./accelerate_config.yaml \
    train.py