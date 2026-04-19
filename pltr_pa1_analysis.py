import os
import warnings
from itertools import chain, combinations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import xgboost as xgb
from scipy.stats import randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    TimeSeriesSplit,
    cross_validate,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 2025
DATA_FILE = "msds_getdata_yfinance_pltr.csv"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

pltr = pl.read_csv(DATA_FILE)
pltr = pltr.with_columns(
    pl.col("Date").str.slice(0, 10).str.strptime(pl.Date, "%Y-%m-%d")
)
print("Original schema:", pltr.schema)
print(f"Loaded {pltr.height} rows of PLTR daily data "
      f"({pltr['Date'].min()} to {pltr['Date'].max()})")

pltr = pltr.drop(["Dividends", "Stock Splits"])

pltr = pltr.with_columns(pl.col("Close").shift().alias("CloseLag1"))
pltr = pltr.with_columns(pl.col("CloseLag1").shift().alias("CloseLag2"))
pltr = pltr.with_columns(pl.col("CloseLag2").shift().alias("CloseLag3"))

pltr = pltr.with_columns((pl.col("High") - pl.col("Low")).alias("HML"))
pltr = pltr.with_columns(pl.col("HML").shift().alias("HMLLag1"))
pltr = pltr.with_columns(pl.col("HMLLag1").shift().alias("HMLLag2"))
pltr = pltr.with_columns(pl.col("HMLLag2").shift().alias("HMLLag3"))

pltr = pltr.with_columns((pl.col("Open") - pl.col("Close")).alias("OMC"))
pltr = pltr.with_columns(pl.col("OMC").shift().alias("OMCLag1"))
pltr = pltr.with_columns(pl.col("OMCLag1").shift().alias("OMCLag2"))
pltr = pltr.with_columns(pl.col("OMCLag2").shift().alias("OMCLag3"))

pltr = pltr.with_columns(pl.col("Volume").shift().alias("VolumeLag1"))
pltr = pltr.with_columns(pl.col("VolumeLag1").shift().alias("VolumeLag2"))
pltr = pltr.with_columns(pl.col("VolumeLag2").shift().alias("VolumeLag3"))

pltr = pltr.with_columns(
    pl.col("CloseLag1").ewm_mean(half_life=1, ignore_nulls=True).alias("CloseEMA2")
)
pltr = pltr.with_columns(
    pl.col("CloseLag1").ewm_mean(half_life=2, ignore_nulls=True).alias("CloseEMA4")
)
pltr = pltr.with_columns(
    pl.col("CloseLag1").ewm_mean(half_life=4, ignore_nulls=True).alias("CloseEMA8")
)

pltr = pltr.with_columns(
    np.log(pl.col("Close") / pl.col("CloseLag1")).alias("LogReturn")
)
pltr = pltr.with_columns(
    pl.when(pl.col("LogReturn") > 0.0).then(pl.lit(1)).otherwise(pl.lit(0)).alias("Target")
)

pltr = pltr.with_columns(
    pl.col("Volume").cast(pl.Float64).round(0),
    pl.col("VolumeLag1").cast(pl.Float64).round(0),
    pl.col("VolumeLag2").cast(pl.Float64).round(0),
    pl.col("VolumeLag3").cast(pl.Float64).round(0),
)

price_cols = [
    "Open", "High", "Low", "Close",
    "CloseLag1", "CloseLag2", "CloseLag3",
    "HML", "HMLLag1", "HMLLag2", "HMLLag3",
    "OMC", "OMCLag1", "OMCLag2", "OMCLag3",
    "CloseEMA2", "CloseEMA4", "CloseEMA8",
]
pltr = pltr.with_columns([pl.col(c).round(3) for c in price_cols])

pltr.write_csv("pltr-with-computed-features.csv")
pltr = pltr.drop_nulls()
print(f"Rows after dropping NaNs from lag/EMA warmup: {pltr.height}")

stats_df = pltr.drop("Date").describe()
stats_to_print = stats_df.transpose(include_header=True)
print("\nDescriptive statistics (transposed):")
with pl.Config(
    tbl_rows=60,
    tbl_width_chars=200,
    tbl_cols=-1,
    float_precision=3,
    tbl_hide_dataframe_shape=True,
    tbl_hide_column_data_types=True,
):
    print(stats_to_print)

print(pltr["Target"].value_counts())


X_df = pltr.drop([
    "Date", "LogReturn", "Target",
    "Open", "High", "Low", "Close", "Volume",
    "HML", "OMC",
])
feature_names = X_df.columns
print(f"\n{len(feature_names)} candidate features:")
for i, name in enumerate(feature_names):
    print(f"  ({i:2d}) {name}")

scaler = StandardScaler()
X = scaler.fit_transform(np.array(X_df))
y = np.array(pltr["Target"])

def get_aic(Xa: np.ndarray, ya: np.ndarray) -> float:
    """AIC for a logistic regression on the given feature columns."""
    model = LogisticRegression(max_iter=1000)
    model.fit(Xa, ya)
    loglik = -log_loss(ya, model.predict_proba(Xa)) * len(ya)
    k = Xa.shape[1] + 1  # +1 for the intercept
    return 2 * k - 2 * loglik


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


print("\nRunning all-possible-subsets logistic regression "
      f"({2**len(feature_names) - 1} subsets)...")
results_schema = {"trialNumber": pl.Int64, "features": pl.String, "aic": pl.Float64}
rows = []
for trial_idx, c in enumerate(powerset(range(X.shape[1])), start=1):
    aic = get_aic(X[:, c], y)
    rows.append({
        "trialNumber": trial_idx,
        "features": " ".join(map(str, c)),
        "aic": aic,
    })
results_df = pl.DataFrame(rows, schema=results_schema)

print("\nTop 10 lowest-AIC subsets:")
top10 = results_df.sort("aic").head(10)
print(top10)


counts = {name: 0 for name in feature_names}
for row in top10.iter_rows(named=True):
    for idx in row["features"].split():
        counts[feature_names[int(idx)]] += 1
print("\nFeature frequency in top 10 subsets:")
for name, c in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"  {name:>11s}: {c}")

selected = [n for n, c in sorted(counts.items(), key=lambda kv: -kv[1]) if c >= 5][:5]
if len(selected) < 3:
    selected = [n for n, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:5]]
print(f"\nSelected feature subset for downstream modeling: {selected}")

heatmap_cols = ["LogReturn", "CloseLag1"] + [c for c in selected if c != "CloseLag1"]
X_study = pltr.select(heatmap_cols)
corr = X_study.corr()
print("\nCorrelation matrix (selected features + LogReturn + CloseLag1):")
print(corr)

plt.figure(figsize=(8, 6))
sns.heatmap(corr.to_pandas().set_index(corr.columns[0]) if "column" in corr.columns[0]
            else corr.to_pandas(),
            cmap="coolwarm", annot=True, fmt=".2f",
            xticklabels=heatmap_cols, yticklabels=heatmap_cols)
plt.title("Correlation heat map — selected PLTR features")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "pltr_correlation_heatmap.png"), dpi=150)
plt.close()

X_selected_df = pltr.select(selected)
X_selected = scaler.fit_transform(np.array(X_selected_df))

tscv = TimeSeriesSplit(gap=10, n_splits=5)
all_splits = list(tscv.split(X_selected, y))
print("\nTimeSeriesSplit folds:")
for i, (tr, te) in enumerate(all_splits):
    print(f"  fold {i}: train [{tr.min()}..{tr.max()}] (n={len(tr)})  "
          f"test [{te.min()}..{te.max()}] (n={len(te)})")

baseline = XGBClassifier(
    objective="binary:logistic",
    n_estimators=1000,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
)


def evaluate(model, Xa, ya, cv):
    cv_results = cross_validate(model, Xa, ya, cv=cv, scoring=["accuracy"])
    acc = cv_results["test_accuracy"]
    return acc.mean(), acc.std()


mean_acc, std_acc = evaluate(baseline, X_selected, y, cv=tscv)
print(f"\nBaseline XGBoost CV accuracy: {mean_acc:.3f} +/- {std_acc:.3f}")


param_dist = {
    "max_depth": randint(3, 10),
    "min_child_weight": randint(1, 10),
    "subsample": uniform(0.5, 0.5),  
    "learning_rate": uniform(0.01, 0.1),
    "n_estimators": randint(100, 1000),
}
search_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
)
random_search = RandomizedSearchCV(
    estimator=search_model,
    param_distributions=param_dist,
    n_iter=100,
    scoring="accuracy",
    cv=TimeSeriesSplit(gap=10, n_splits=5),
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
print("\nRunning RandomizedSearchCV (100 iters)...")
random_search.fit(X_selected, y)
print(f"Best params: {random_search.best_params_}")
print(f"Best CV accuracy: {random_search.best_score_:.3f}")


best_params = random_search.best_params_
final_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    **best_params,
)
final_model.fit(X_selected, y)
y_pred = final_model.predict(X_selected)
y_proba = final_model.predict_proba(X_selected)[:, 1]

auc = roc_auc_score(y, y_proba)
print(f"\nIn-sample ROC AUC: {auc:.3f}")

fig, ax = plt.subplots(figsize=(6, 6))
RocCurveDisplay.from_predictions(y, y_proba, ax=ax, name="XGBoost (final)")
ax.set_title(f"PLTR ROC curve (AUC = {auc:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "pltr_roc_curve.png"), dpi=150)
plt.close()

print("\nConfusion matrix:")
print(confusion_matrix(y, y_pred))

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y, y_pred,
    display_labels=["Negative Return", "Positive Return"],
    cmap=plt.cm.Blues,
    ax=ax,
)
ax.set_title("Confusion Matrix — PLTR direction prediction")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "pltr_confusion_matrix.png"), dpi=150)
plt.close()

print("\nClassification report:")
print(classification_report(
    y, y_pred,
    target_names=["Negative Return", "Positive Return"],
))

fig, ax = plt.subplots(figsize=(7, 4))
importances = final_model.feature_importances_
order = np.argsort(importances)[::-1]
ax.bar([selected[i] for i in order], importances[order], color="#306998")
ax.set_ylabel("Importance (gain)")
ax.set_title("XGBoost feature importance — PLTR")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "pltr_feature_importance.png"), dpi=150)
plt.close()

print("\nFigures saved to ./figures/")
print("Done.")
