"""Extra experiments: (1) data-leakage check (duplicates), (2) feature ablation, (3) neural network (MLP) benchmark.

Run:  python experiments.py
Outputs: reports/leakage_check.csv, reports/ablation.csv, reports/mlp_benchmark.csv
"""
import warnings
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")
SEED = 42
ROOT = Path(__file__).parent
raw = pd.read_csv(ROOT / "data" / "sms.tsv", sep="\t", header=None, names=["label", "message"])
dedup = raw.drop_duplicates().reset_index(drop=True)


def svm(ngram=(1, 2), sublinear=True):
    return Pipeline([("tfidf", TfidfVectorizer(lowercase=True, ngram_range=ngram, min_df=2, sublinear_tf=sublinear)),
                     ("clf", CalibratedClassifierCV(LinearSVC(C=0.3, class_weight="balanced", random_state=SEED), cv=3))])


def split(df):
    y = (df["label"] == "spam").astype(int)
    return train_test_split(df["message"], y, test_size=0.2, stratify=y, random_state=SEED)


# ---------------------------------------------------------------- (1) duplicate leakage
rows = []
for name, df in [("Raw data (duplicates kept)", raw), ("De-duplicated (used in the project)", dedup)]:
    Xtr, Xte, ytr, yte = split(df)
    overlap = int(Xte.isin(set(Xtr)).sum())
    pred = svm().fit(Xtr, ytr).predict(Xte)
    rows.append({"setup": name, "n_test": len(Xte), "test_msgs_also_in_train": overlap,
                 "f1": round(f1_score(yte, pred), 4), "precision": round(precision_score(yte, pred), 4),
                 "recall": round(recall_score(yte, pred), 4)})
leak = pd.DataFrame(rows)
print(leak.to_string(index=False))
leak.to_csv(ROOT / "reports" / "leakage_check.csv", index=False)

# ---------------------------------------------------------------- (2) + (3) CV on the de-duplicated training split
Xtr, Xte, ytr, yte = split(dedup)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def cvscore(p):
    r = cross_validate(p, Xtr, ytr, cv=cv, scoring=["f1", "precision", "recall", "roc_auc"], n_jobs=1)
    return {k: round(r[f"test_{k}"].mean(), 4) for k in ["f1", "precision", "recall", "roc_auc"]}


abl = pd.DataFrame([
    {"variant": "Selected: unigrams + bigrams, sublinear TF", **cvscore(svm())},
    {"variant": "- unigrams only", **cvscore(svm(ngram=(1, 1)))},
    {"variant": "- without sublinear TF scaling", **cvscore(svm(sublinear=False))},
])
print(abl.to_string(index=False))
abl.to_csv(ROOT / "reports" / "ablation.csv", index=False)

mlp = Pipeline([("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                ("clf", MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, early_stopping=True,
                                      validation_fraction=0.15, max_iter=300, random_state=SEED))])
bench = pd.DataFrame([
    {"model": "Linear SVM, calibrated (selected)", **cvscore(svm())},
    {"model": "MLP neural network (64 hidden units)", **cvscore(mlp)},
])
print(bench.to_string(index=False))
bench.to_csv(ROOT / "reports" / "mlp_benchmark.csv", index=False)
