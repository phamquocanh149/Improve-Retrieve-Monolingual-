# Improving Retrieval-Augmented Neural Machine Translation with Monolingual Data​

CLIR4NMT: Cross-Lingual Information Retrieval for Retrieval Augmented Machine Translation

## installation requirements

An adapted version of Fairseq is required to be installed and is used for efficient data handling.

```bash
git clone https://github.com/Maxwell1447/fairseq-clir4ramt
cd fairseq
pip install .
python setup.py build_ext --inplace
```

To install this package:
```bash
git clone https://github.com/Maxwell1447/clir4ramt.git
cd clir
pip install .
```

```bash
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

## data preprocessing

Training data must be processed with command fairseq-preprocess applied on already tokenized data, with the following template:
```bash
fairseq-preprocess \
--source-lang $l1 --target-lang $l2 \
--srcdict $dict \
--joined-dictionary \
--trainpref  $DATA_TOK/train \
--validpref $DATA_TOK/valid \
--destdir $DATA_BIN/ \
--workers 5
```

## training

```bash
python -m clir.train.trainer fit --config=experiments/configs/{config}.yaml
```

## indexing

```bash
python -m clir.index.index --config experiments/configs/{config}.yaml
```

## retrieving

```bash
python -m clir.retrieve.retrieve --config experiments/configs/{config}.yaml
```

## Để cải tiến với bài toán dịch Việt - Khmer kết hợp retrieve để chạy offline trên Kaggle RTX thì đây là code đã được chỉnh sửa phù hợp, tuy nhiên có vẻ chưa được thực sự hiệu quả
