# ConTrans
This repository provides the code for the paper [CONTRANS: Weak-to-Strong Alignment Engineering
via Concept Transplantation](https://arxiv.org/abs/2405.13578).

# Results
![pic](assets/emotion.png "Emotion Concept Acc.")
![pic](assets/generation.png "Generation Output")

# Usage
Replace the `model_name2path` in `util/utils.py` to your own path of models.

## Get hidden states and concept vectors
```bash
bash get_hidden.sh
```

## Train affine transformation
```bash
bash train_affine.sh
```

## Convert concept vector
```bash
python convert_vectors.py
```

## Evaluation emotion transplantation
```bash
bash eval_emotion.sh
```