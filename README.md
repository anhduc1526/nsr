# From Local Perspective to Global Reasoning: A Neuro-Symbolic Framework for Zero-Shot Relation Extraction

This repository contains code and datasets for the paper: From Local Perspective to Global Reasoning: A Neuro-Symbolic Framework for Zero-Shot Relation Extraction.

## Environment Setup

Install requirements:

```bash
pip install -r requirements.txt
```

## Datasets

The `Datasets` folder contains two benchmark datasets: `Wiki-ZSL`, and `FewRel`.

## G-NSR Training & Evaluation:

To train and evaluate the model on the `FewRel` dataset under the `m=15` setting, execute the following command:

```bash
python main.py \
--encoder_name="google-bert/bert-base-uncased" \
--dataset="fewrel" \
--num_unseen_labels="unseen_15"
```

*Note:* If encountering issues loading the `google-bert/bert-base-uncased` model, we recommend manually downloading the model and updating the `encoder_name` parameter to the local model path.
