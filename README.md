# DSAI 490 – Assignment 2: Conditional Date Generation


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
