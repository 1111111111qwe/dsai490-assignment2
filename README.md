# DSAI 490 – Assignment 2: Conditional Date Generation

## Structure
```
repo/
├── data/
│   ├── data.txt
│   └── example_input.txt
├── model/
│   ├── tokenizer.py          # shared vocab, condition encoding, date helpers
│   ├── evaluate.py           # constraint satisfaction rate
│   ├── predict.py            # inference entry point
│   ├── cvae/
│   │   ├── model.py          # CVAE architecture
│   │   ├── train.py          # custom tf.GradientTape training
│   │   └── weights/
│   ├── cgan/
│   │   ├── model.py          # Generator + Discriminator
│   │   ├── train.py          # custom tf.GradientTape training
│   │   └── weights/
│   ├── seq2seq/
│   │   ├── model.py          # LSTM encoder-decoder
│   │   ├── train.py
│   │   └── weights/
│   └── transformer/
│       ├── model.py          # multi-head attention encoder-decoder
│       ├── train.py
│       └── weights/
└── environment.yml
```

## Setup
```bash
conda env create -f environment.yml
conda activate dsai490_a2
```

## Train models
```bash
# From inside each model folder:
cd model/cvae        && python train.py --data ../../data/data.txt
cd model/cgan        && python train.py --data ../../data/data.txt
cd model/seq2seq     && python train.py --data ../../data/data.txt
cd model/transformer && python train.py --data ../../data/data.txt
```

## Inference
```bash
python model/predict.py -i data/example_input.txt -o predictions.txt
```

Set `MODEL_CHOICE` at the top of `model/predict.py` to switch between
`"transformer"`, `"seq2seq"`, `"cvae"`, or `"cgan"`.
