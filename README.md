# Contextual Threat Detection 


Classify tweets as real-disaster vs. non-disaster so first-responders can be alerted faster.
A naive keyword scan over Twitter is misleading — *"Ronaldo scored two goals, he's on fire"* is not a fire — so we need a model that understands context.

## Approach

Two models are trained on the [Kaggle Real or Not? Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) dataset and compared:

| Model | Validation accuracy | F1 (macro) |
| --- | --- | --- |
| **BERT** (`bert_en_uncased_L-12_H-768_A-12`, fine-tuned 5 epochs) | **0.97** | **0.97** |
| Multinomial Naive Bayes (TF-IDF) | 0.81 | 0.80 |

BERT wins clearly on this task and is what we ship.

## Pipeline

1. **Load** `train.csv` / `test.csv`, drop duplicate tweet text, fill missing `keyword`/`location`.
2. **Clean** text — strip URLs, replace emojis with tokens (`:)` → `EMOJIsmile`), drop @mentions, collapse repeated letters, lemmatize, remove stopwords, lowercase.
3. **Split** stratified 95/5 train/validation.
4. **BERT branch** — TF-Hub preprocessor → BERT encoder (trainable) → dropout → sigmoid head, AdamW with warmup, binary cross-entropy.
5. **NB branch** — `TfidfVectorizer` → `MultinomialNB`.
6. **Evaluate** both with confusion matrix + classification report; write `submission.csv` from BERT (threshold 0.4).

## Dataset

`train.csv` (7,613 rows) and `test.csv` ship with the repo. Columns:

- `id`, `keyword`, `location`, `text` — the tweet
- `target` — 1 = real disaster, 0 = not (training only)

Class balance: ~57% non-disaster, ~43% disaster.

## Run it

Open `Predict_Tweets_with_Real_Threats (1).ipynb` in Google Colab (recommended — it expects `/content/train.csv`, `/content/test.csv`) or a local Jupyter with a GPU. Run cells top-to-bottom.

## Files

- `Predict_Tweets_with_Real_Threats (1).ipynb` — full pipeline: EDA, preprocessing, BERT + NB training, evaluation
- `train.csv`, `test.csv` — Kaggle dataset
- `Predict tweets with real threats.pptx` / `.pdf` — project presentation

## Future work

- Try larger / domain-tuned BERT variants (RoBERTa, DeBERTa, BERTweet) for additional gains.
- Wire into a live Twitter ingestion pipeline so flagged tweets fan out to disaster-relief teams.

## Authors

- Akhil Kamble (210941225005)
- Rushikesh Pawar (210941225036)
