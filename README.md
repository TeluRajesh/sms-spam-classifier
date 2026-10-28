# SMS Spam Classifier (NLP)

Classifies SMS messages as spam or legitimate (ham). Text-classification project covering
text vectorisation (TF-IDF), model comparison with cross-validation, hyperparameter tuning,
error analysis and a Streamlit app.

## Problem
Spam texts waste customers' time and carry scams. Goal: flag spam while keeping false alarms low,
because wrongly blocking a real message is costly.

## Data
[SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (UCI, CC BY 4.0):
5,572 labelled English messages. **403 exact duplicates were removed** before splitting so the same
message cannot appear in both train and test, leaving 5,169 messages (653 spam, 12.6%).

## Approach
1. **Split:** stratified 80/20 train/test (4,135 / 1,034), `random_state=42`. Test set used once, at the end.
2. **Features:** `TfidfVectorizer` (lowercase, unigrams + bigrams, `min_df=2`, sublinear TF).
3. **Models compared** with 5-fold stratified CV on the training set: majority-class baseline, Multinomial Naive Bayes, Logistic Regression, Linear SVM (probability-calibrated).
4. **Tuning:** `GridSearchCV` (scoring = F1 on the spam class) over n-gram range and each model's regularisation parameter.
5. **Selection:** best cross-validated F1.

## Results

5-fold cross-validation on the training set (before tuning):

| Model | F1 | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| Baseline (majority class) | 0.000 | 0.000 | 0.000 | 0.500 |
| Multinomial Naive Bayes | 0.936 | 0.989 | 0.889 | 0.988 |
| Logistic Regression | 0.934 | 0.930 | 0.939 | 0.991 |
| Linear SVM (calibrated) | 0.943 | 0.955 | 0.931 | 0.991 |

After tuning, the calibrated Linear SVM (`C=0.3`, unigrams + bigrams) had the best CV F1 (0.945) and was selected.

**Held-out test set (1,034 messages):**

| Metric | Value |
|---|---|
| F1 (spam) | 0.913 |
| Precision | 0.909 |
| Recall | 0.916 |
| ROC-AUC | 0.990 |
| Accuracy | 0.978 |

Plots and tables in `reports/`: `confusion_matrix.png`, `top_words.png`, `errors.csv`, `model_comparison.csv`.

## Error analysis
23 of 1,034 test messages were misclassified (11 missed spam, 12 legitimate messages flagged as spam); see `reports/errors.csv`.
- Several false positives are short, casual messages that contain words the model links to spam, such as "call" and "text" (for example "I am waiting for your call sir.").
- Several missed spam messages are adult-content chat promotions and missed-call alerts that avoid the typical prize/offer vocabulary.
- Most informative words from a logistic-regression fit on the same features: spam is associated with `call`, `txt`, `text`, `free`, `reply`, `www`, `mobile`, `150p`; ham with `my`, `ok`, `me`, `that`, `at`.

## Limitations
- Bag-of-words features ignore word order and context, so short ambiguous messages are the main source of errors. A transformer model (for example fine-tuned DistilBERT) is a natural next step.
- The dataset is from the 2000s UK/Singapore SMS context; modern spam and other languages may differ.

## Run it
```bash
pip install -r requirements.txt
python train.py          # trains, evaluates, writes models/ and reports/
streamlit run app.py     # interactive classifier
```

## Project structure
```
data/        sms.tsv
train.py     CV comparison, tuning, evaluation, error analysis, plots
app.py       Streamlit app
api.py       Flask REST API
experiments.py  ablation, leakage check, neural-network benchmark
tests/       unit tests
Dockerfile, .github/workflows/ci.yml
models/      spam_model.joblib
reports/     metrics.json, model_comparison.csv, errors.csv, plots
```

## Extra experiments (`experiments.py`)

**Data-leakage check.** The raw file contains 403 duplicate messages. Splitting it without removing them puts 128 of 1,115 test messages (11.5%) in the training set too, and inflates the score:

| Setup | Test messages also in train | F1 | Precision | Recall |
|---|---|---|---|---|
| Raw data (duplicates kept) | 128 | 0.949 | 0.965 | 0.933 |
| De-duplicated (used here) | 0 | 0.913 | 0.909 | 0.916 |

**Feature ablation (5-fold CV, training split):**

| Variant | F1 | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| Unigrams + bigrams, sublinear TF (selected) | 0.945 | 0.956 | 0.935 | 0.992 |
| unigrams only | 0.934 | 0.949 | 0.920 | 0.991 |
| without sublinear TF scaling | 0.941 | 0.951 | 0.933 | 0.991 |

**Neural network benchmark:** a scikit-learn `MLPClassifier` (64 hidden units, early stopping) got CV F1 0.916 (precision 0.996, recall 0.849, ROC-AUC 0.983) vs 0.945 for the calibrated Linear SVM, so the SVM was kept.

## REST API (`api.py`)
```bash
python api.py
curl -X POST http://127.0.0.1:5000/classify -H "Content-Type: application/json" -d '{"message": "WINNER!! Call 09061701461 now to claim your free prize"}'
```
`GET /health` for health checks. Invalid input returns 400 (413 if longer than 2,000 characters).

## Tests
```bash
python -m unittest discover -s tests -v    # 8 tests covering the API
```

## Docker and CI (provided, not verified by the author of the scaffolding)
`Dockerfile`: `docker build -t spam-api . && docker run -p 5000:5000 spam-api`. `.github/workflows/ci.yml` trains and tests on every push. **Run both yourself and confirm they pass before claiming them anywhere.**
