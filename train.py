"""Train and evaluate an SMS spam classifier (TF-IDF + classical ML).

Run:  python train.py
Outputs: models/spam_model.joblib, reports/*
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", message="Precision is ill-defined")

SEED = 42
ROOT = Path(__file__).parent
(ROOT / "models").mkdir(exist_ok=True)
(ROOT / "reports").mkdir(exist_ok=True)

# ---------------------------------------------------------------- data
df = pd.read_csv(ROOT / "data" / "sms.tsv", sep="\t", header=None, names=["label", "message"])
n_raw = len(df)
df = df.drop_duplicates().reset_index(drop=True)  # exact duplicate messages would leak between train and test
print(f"messages: {n_raw} raw -> {len(df)} after dropping {n_raw - len(df)} exact duplicates")
print(df.label.value_counts().to_dict())
y = (df["label"] == "spam").astype(int)
X = df["message"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"train={len(X_train)} test={len(X_test)} spam_rate={y.mean():.3f}")


def pipe(clf):
    return Pipeline([("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                     ("clf", clf)])


# ---------------------------------------------------------------- 5-fold CV comparison (train only)
candidates = {
    "Baseline (majority class)": DummyClassifier(strategy="most_frequent"),
    "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
    "Linear SVM (calibrated)": CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=SEED), cv=3),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rows = []
for name, clf in candidates.items():
    res = cross_validate(pipe(clf), X_train, y_train, cv=cv,
                         scoring=["f1", "precision", "recall", "roc_auc"], n_jobs=1)
    rows.append({"model": name, "cv_f1": res["test_f1"].mean(), "cv_precision": res["test_precision"].mean(),
                 "cv_recall": res["test_recall"].mean(), "cv_roc_auc": res["test_roc_auc"].mean()})
    print(f"{name:28s} CV F1={rows[-1]['cv_f1']:.3f} precision={rows[-1]['cv_precision']:.3f} "
          f"recall={rows[-1]['cv_recall']:.3f} ROC-AUC={rows[-1]['cv_roc_auc']:.3f}")
pd.DataFrame(rows).round(4).to_csv(ROOT / "reports" / "model_comparison.csv", index=False)

# ---------------------------------------------------------------- tuning (GridSearchCV, scoring = F1)
grids = {
    "Multinomial Naive Bayes": (pipe(MultinomialNB()),
                                {"tfidf__ngram_range": [(1, 1), (1, 2)], "clf__alpha": [0.03, 0.1, 0.3, 1.0]}),
    "Logistic Regression": (pipe(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
                            {"tfidf__ngram_range": [(1, 1), (1, 2)], "clf__C": [1, 10, 30, 100]}),
    "Linear SVM (calibrated)": (pipe(CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=SEED), cv=3)),
                                {"tfidf__ngram_range": [(1, 1), (1, 2)], "clf__estimator__C": [0.1, 0.3, 1, 3]}),
}
tuned = {}
for name, (p, grid) in grids.items():
    gs = GridSearchCV(p, grid, cv=cv, scoring="f1", n_jobs=1)
    gs.fit(X_train, y_train)
    tuned[name] = gs
    print(f"tuned {name}: best CV F1={gs.best_score_:.4f} params={gs.best_params_}")

best_name = max(tuned, key=lambda n: tuned[n].best_score_)
best = tuned[best_name].best_estimator_
print(f"\nSelected model (best CV F1): {best_name}")

# ---------------------------------------------------------------- final test evaluation
proba = best.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
metrics = {
    "selected_model": best_name,
    "best_params": {k: str(v) for k, v in tuned[best_name].best_params_.items()},
    "n_messages_after_dedup": int(len(df)), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
    "test_accuracy": accuracy_score(y_test, pred), "test_precision": precision_score(y_test, pred),
    "test_recall": recall_score(y_test, pred), "test_f1": f1_score(y_test, pred),
    "test_roc_auc": roc_auc_score(y_test, proba),
}
metrics = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}
print(json.dumps(metrics, indent=2))
(ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))

fig, ax = plt.subplots(figsize=(4.5, 4))
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Ham", "Spam"], cmap="Blues", ax=ax)
ax.set_title(f"Confusion matrix - {best_name}", fontsize=9)
fig.tight_layout(); fig.savefig(ROOT / "reports" / "confusion_matrix.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- error analysis
err = pd.DataFrame({"message": X_test.values, "actual": y_test.values, "pred": pred, "spam_prob": proba.round(3)})
errors = err[err.actual != err.pred].copy()
errors["error_type"] = np.where(errors.actual == 1, "missed spam (FN)", "ham flagged as spam (FP)")
errors.to_csv(ROOT / "reports" / "errors.csv", index=False)
print(f"\nmisclassified: {len(errors)}  (FN={int((errors.actual == 1).sum())}, FP={int((errors.actual == 0).sum())})")
print(errors.head(12).to_string(max_colwidth=90))

# top spam / ham words from a plain logistic regression on the same vectoriser
vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(X_train)
lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=10, random_state=SEED).fit(vec.transform(X_train), y_train)
coef = pd.Series(lr.coef_[0], index=vec.get_feature_names_out()).sort_values()
top = pd.concat([coef.head(10), coef.tail(10)])
print("\nTop spam indicators:", ", ".join(coef.tail(10)[::-1].index))
print("Top ham indicators:", ", ".join(coef.head(10).index))
fig, ax = plt.subplots(figsize=(6, 5))
colors = ["#2b6cb0" if v < 0 else "#c53030" for v in top.values]
ax.barh(top.index, top.values, color=colors); ax.set_xlabel("Logistic Regression coefficient (red = spam, blue = ham)")
ax.set_title("Most informative words"); fig.tight_layout()
fig.savefig(ROOT / "reports" / "top_words.png", dpi=150); plt.close(fig)

joblib.dump(best, ROOT / "models" / "spam_model.joblib")
print("\nSaved model and reports.")
