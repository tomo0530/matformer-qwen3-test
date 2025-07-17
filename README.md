# MatFormer for Qwen3

MatFormerアーキテクチャを使用してQwen3-4Bモデルから3Bサイズのカスタムモデルを作成するプロジェクトです。

## 概要

- **ベースモデル**: Qwen3-4B (36層, 32 Q heads, 8 KV heads)
- **ターゲット**: 3Bパラメータのカスタムモデル (32層)
- **手法**: Mix-n-Match FFN次元によるMatFormerアーキテクチャ

## 参考文献

- [Qwen3 モデル](https://huggingface.co/Qwen/Qwen3-4B)
- [MatFormer アーキテクチャ](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide)
