# reproduce-transformer
基于PyTorch复现Transformer英中机器翻译模型，实现完整训练与贪婪解码推理，使用Parquet双语平行语料完成翻译实验。

## 环境依赖
```bash
pip install torch transformers datasets numpy pyarrow
