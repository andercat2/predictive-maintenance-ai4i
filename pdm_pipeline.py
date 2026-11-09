"""
Предиктивный ремонт и диагностика оборудования (Predictive Maintenance)
=======================================================================

Датасет: AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository, id=601)
         https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
         10 000 записей фрезерного станка: температура воздуха/процесса, обороты шпинделя,
         крутящий момент, износ инструмента, тип изделия (L/M/H) + метки отказов.

Задачи:
  1) Бинарная классификация — будет ли отказ оборудования (Machine failure).
  2) Мультиклассовая диагностика — какой тип отказа:
       TWF – износ инструмента, HDF – перегрев (плохой теплоотвод),
       PWF – выход мощности за допустимый диапазон, OSF – механическая перегрузка,
       Прочее – случайные/неклассифицированные отказы.

Выход:
  results/metrics_models.xlsx   – все метрики и качество моделей (главный файл)
  results/metrics_binary.csv    – сводная таблица бинарной задачи
  results/metrics_multiclass.csv
  results/figures/*.png         – графики
  results/demo_data.json        – данные для интерактивной демонстрации
  models/*.joblib               – обученные лучшие модели

Запуск:  python pdm_pipeline.py
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, classification_report, confusion_matrix, f1_score,
                             fbeta_score, matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------------------
# Конфигурация
# ----------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "ai4i2020.csv"
RES = ROOT / "results"
FIG = RES / "figures"
MODELS_DIR = ROOT / "models"
for p in (RES, FIG, MODELS_DIR):
    p.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 5
TEST_SIZE = 0.2
N_ITER_SEARCH = 40

# Условная экономика (руб.) — ДОПУЩЕНИЯ ДЛЯ ДЕМОНСТРАЦИИ, заменить на данные предприятия
COST_UNPLANNED_FAILURE = 1_000_000  # пропущенный отказ: аварийный простой + внеплановый ремонт
COST_PLANNED_REPAIR = 150_000       # отказ предсказан: плановое ТО до поломки
COST_FALSE_ALARM = 30_000           # ложная тревога: лишний осмотр / остановка

TYPE_MAP = {"L": 0, "M": 1, "H": 2}
RAW_FEATURES = ["type_code", "air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min"]
ENG_FEATURES = ["temp_diff_k", "power_w", "strain_nm_min"]
FEATURES = RAW_FEATURES + ENG_FEATURES

FEATURE_RU = {
    "type_code": "Тип изделия (L/M/H)",
    "air_temp_k": "Температура воздуха, K",
    "process_temp_k": "Температура процесса, K",
    "rpm": "Обороты шпинделя, об/мин",
    "torque_nm": "Крутящий момент, Нм",
    "tool_wear_min": "Износ инструмента, мин",
    "temp_diff_k": "Разница температур (процесс − воздух), K",
    "power_w": "Механическая мощность, Вт",
    "strain_nm_min": "Нагрузка на инструмент (износ × момент)",
}

FAILURE_RU = {
    "Нет отказа": "Нет отказа",
    "TWF": "TWF — износ инструмента",
    "HDF": "HDF — перегрев (теплоотвод)",
    "PWF": "PWF — мощность вне диапазона",
    "OSF": "OSF — механическая перегрузка",
    "Прочее": "Прочее — случайный отказ",
}


# ----------------------------------------------------------------------------------------
# Данные
# ----------------------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df = df.rename(columns={
        "UDI": "udi", "Product ID": "product_id", "Type": "type",
        "Air temperature [K]": "air_temp_k", "Process temperature [K]": "process_temp_k",
        "Rotational speed [rpm]": "rpm", "Torque [Nm]": "torque_nm",
        "Tool wear [min]": "tool_wear_min", "Machine failure": "failure",
    })
    df["type_code"] = df["type"].map(TYPE_MAP)
    # Физически обоснованные признаки (feature engineering)
    df["temp_diff_k"] = df["process_temp_k"] - df["air_temp_k"]           # теплоотвод -> HDF
    df["power_w"] = df["torque_nm"] * df["rpm"] * 2 * np.pi / 60           # мощность -> PWF
    df["strain_nm_min"] = df["tool_wear_min"] * df["torque_nm"]            # перегрузка -> OSF

    # Метка типа отказа (для отказов с несколькими режимами — по приоритету)
    def failure_type(r):
        if r["failure"] == 0:
            return "Нет отказа"
        for m in ("HDF", "PWF", "OSF", "TWF"):
            if r[m] == 1:
                return m
        return "Прочее"  # RNF или отказ без указанного режима

    df["failure_type"] = df.apply(failure_type, axis=1)
    return df


def dataset_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("Источник", "UCI ML Repository, AI4I 2020 Predictive Maintenance Dataset (id=601)"),
        ("Ссылка", "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset"),
        ("Лицензия", "CC BY 4.0"),
        ("Записей", len(df)),
        ("Исходных признаков", len(RAW_FEATURES)),
        ("Сгенерированных признаков", len(ENG_FEATURES)),
        ("Отказов (Machine failure = 1)", int(df["failure"].sum())),
        ("Доля отказов", f"{df['failure'].mean():.2%}"),
        ("Дисбаланс классов (норма : отказ)", f"{(1 - df['failure'].mean()) / df['failure'].mean():.1f} : 1"),
        ("Пропуски в данных", int(df.isna().sum().sum())),
    ]
    for m in ("TWF", "HDF", "PWF", "OSF", "RNF"):
        rows.append((f"Флаг режима {m}", int(df[m].sum())))
    rows.append(("Отказов с несколькими режимами", int((df[["TWF", "HDF", "PWF", "OSF", "RNF"]].sum(axis=1) > 1).sum())))
    rows.append(("Отказ без указанного режима", int(((df["failure"] == 1) & (df[["TWF", "HDF", "PWF", "OSF", "RNF"]].sum(axis=1) == 0)).sum())))
    rows.append(("RNF без отметки Machine failure", int(((df["failure"] == 0) & (df["RNF"] == 1)).sum())))
    rows.append(("Разбиение train / test", f"{1 - TEST_SIZE:.0%} / {TEST_SIZE:.0%}, стратифицированное, seed={RANDOM_STATE}"))
    rows.append(("Валидация", f"Stratified {N_FOLDS}-fold CV на train + отложенная test-выборка"))
    return pd.DataFrame(rows, columns=["Параметр", "Значение"])


# ----------------------------------------------------------------------------------------
# Модели
# ----------------------------------------------------------------------------------------
def binary_models(pos_weight: float) -> dict:
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0)),
        "k-NN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, weights="distance")),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=RANDOM_STATE),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", class_weight="balanced", probability=True, random_state=RANDOM_STATE)),
        "MLP (нейросеть)": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=800, random_state=RANDOM_STATE)),
        "Random Forest": RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE),
        "Extra Trees": ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=5, subsample=0.85, colsample_bytree=0.85,
                                 scale_pos_weight=np.sqrt(pos_weight), eval_metric="aucpr", n_jobs=4, random_state=RANDOM_STATE),
        "LightGBM": LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31, subsample=0.85, subsample_freq=1,
                                   colsample_bytree=0.85, scale_pos_weight=np.sqrt(pos_weight), verbose=-1, n_jobs=4, random_state=RANDOM_STATE),
        "CatBoost": CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6, auto_class_weights="SqrtBalanced",
                                       verbose=0, thread_count=4, random_seed=RANDOM_STATE),
    }


SEARCH_SPACES = {
    "XGBoost": (
        XGBClassifier(eval_metric="aucpr", n_jobs=1, random_state=RANDOM_STATE),
        {"n_estimators": [300, 600, 1000], "learning_rate": [0.02, 0.05, 0.1], "max_depth": [3, 4, 5, 6, 8],
         "min_child_weight": [1, 3, 5], "subsample": [0.7, 0.85, 1.0], "colsample_bytree": [0.7, 0.85, 1.0],
         "scale_pos_weight": [1, 3, 5, 10, 28], "gamma": [0, 0.5, 1], "reg_lambda": [1, 3, 10]},
    ),
    "LightGBM": (
        LGBMClassifier(verbose=-1, n_jobs=1, subsample_freq=1, random_state=RANDOM_STATE),
        {"n_estimators": [300, 600, 1000], "learning_rate": [0.02, 0.05, 0.1], "num_leaves": [7, 15, 31, 63],
         "max_depth": [-1, 4, 6, 10], "min_child_samples": [5, 10, 20, 40], "subsample": [0.7, 0.85, 1.0],
         "colsample_bytree": [0.7, 0.85, 1.0], "scale_pos_weight": [1, 3, 5, 10, 28], "reg_lambda": [0, 1, 5]},
    ),
    "CatBoost": (
        CatBoostClassifier(verbose=0, thread_count=1, random_seed=RANDOM_STATE),
        {"iterations": [500, 1000], "learning_rate": [0.03, 0.05, 0.1], "depth": [4, 6, 8],
         "l2_leaf_reg": [1, 3, 5, 10], "auto_class_weights": [None, "SqrtBalanced", "Balanced"]},
    ),
}


def multiclass_models() -> dict:
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000)),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=3, class_weight="balanced", random_state=RANDOM_STATE),
        "MLP (нейросеть)": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=800, random_state=RANDOM_STATE)),
        "Random Forest": RandomForestClassifier(n_estimators=500, min_samples_leaf=1, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE),
        "Extra Trees": ExtraTreesClassifier(n_estimators=500, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=5, subsample=0.85, colsample_bytree=0.85,
                                 n_jobs=4, random_state=RANDOM_STATE),
        "LightGBM": LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=15, min_child_samples=5,
                                   class_weight="balanced", verbose=-1, n_jobs=4, random_state=RANDOM_STATE),
        "CatBoost": CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6, auto_class_weights="SqrtBalanced",
                                       verbose=0, thread_count=4, random_seed=RANDOM_STATE),
    }


# ----------------------------------------------------------------------------------------
# Метрики
# ----------------------------------------------------------------------------------------
def best_f1_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[np.nanargmax(f1[:-1])])


def cost_of(y, pred) -> float:
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return tp * COST_PLANNED_REPAIR + fp * COST_FALSE_ALARM + fn * COST_UNPLANNED_FAILURE


def best_cost_threshold(y, p) -> float:
    grid = np.unique(np.concatenate([np.linspace(0.001, 0.999, 999), np.quantile(p, np.linspace(0, 1, 501))]))
    costs = [cost_of(y, (p >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmin(costs))])


def binary_metrics(y, p, thr) -> dict:
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "ROC-AUC": roc_auc_score(y, p),
        "PR-AUC": average_precision_score(y, p),
        "F1": f1_score(y, pred, zero_division=0),
        "F2 (акцент на полноту)": fbeta_score(y, pred, beta=2, zero_division=0),
        "Precision (точность тревог)": precision_score(y, pred, zero_division=0),
        "Recall (доля пойманных отказов)": recall_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
        "Balanced Accuracy": balanced_accuracy_score(y, pred),
        "Accuracy": accuracy_score(y, pred),
        "Brier score": brier_score_loss(y, p),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def cv_binary(model, X, y, skf):
    oof = np.zeros(len(y))
    fold_auc, fold_ap, t_fit = [], [], 0.0
    for tr, va in skf.split(X, y):
        m = clone(model)
        t0 = time.perf_counter()
        m.fit(X.iloc[tr], y.iloc[tr])
        t_fit += time.perf_counter() - t0
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        fold_auc.append(roc_auc_score(y.iloc[va], oof[va]))
        fold_ap.append(average_precision_score(y.iloc[va], oof[va]))
    return oof, np.array(fold_auc), np.array(fold_ap), t_fit / skf.get_n_splits()


def cv_multiclass(model, X, y, skf):
    oof = np.zeros(len(y), dtype=int)
    fold_f1 = []
    for tr, va in skf.split(X, y):
        m = clone(model)
        m.fit(X.iloc[tr], y[tr])
        oof[va] = np.asarray(m.predict(X.iloc[va])).ravel().astype(int)
        fold_f1.append(f1_score(y[va], oof[va], average="macro"))
    return oof, np.array(fold_f1)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------------------
# Основной сценарий
# ----------------------------------------------------------------------------------------
def main():
    df = load_data()
    log(f"Данные: {df.shape}, отказов {df['failure'].sum()} ({df['failure'].mean():.2%})")

    X = df[FEATURES]
    y = df["failure"]
    idx_tr, idx_te = train_test_split(df.index, test_size=TEST_SIZE, stratify=df["failure_type"], random_state=RANDOM_STATE)
    X_tr, X_te, y_tr, y_te = X.loc[idx_tr], X.loc[idx_te], y.loc[idx_tr], y.loc[idx_te]
    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ---------- 1. Бинарная задача: базовые модели ----------
    models = binary_models(pos_weight)

    # ---------- 1a. Тюнинг бустингов ----------
    tuned_params = []
    for name, (est, space) in SEARCH_SPACES.items():
        log(f"Тюнинг {name} ({N_ITER_SEARCH} комбинаций × {N_FOLDS} фолдов)")
        rs = RandomizedSearchCV(est, space, n_iter=N_ITER_SEARCH, scoring="average_precision", cv=skf,
                                n_jobs=-1, random_state=RANDOM_STATE, refit=False)
        rs.fit(X_tr, y_tr)
        best = clone(est).set_params(**rs.best_params_)
        if name == "CatBoost":
            best.set_params(thread_count=4)
        else:
            best.set_params(n_jobs=4)
        models[f"{name} (tuned)"] = best
        tuned_params.append({"Модель": name, "CV PR-AUC (поиск)": rs.best_score_,
                             "Лучшие гиперпараметры": json.dumps(rs.best_params_, ensure_ascii=False, default=str)})

    # ---------- 1b. CV + отложенный тест ----------
    rows_test, rows_cv, rows_biz, curves, fitted, oofs, test_probs = [], [], [], {}, {}, {}, {}

    def evaluate(name, model):
        log(f"Бинарная: {name}")
        oof, f_auc, f_ap, t_fit_cv = cv_binary(model, X_tr, y_tr, skf)
        thr = best_f1_threshold(y_tr, oof)
        thr_cost = best_cost_threshold(y_tr.values, oof)
        m = clone(model)
        t0 = time.perf_counter(); m.fit(X_tr, y_tr); t_fit = time.perf_counter() - t0
        t0 = time.perf_counter(); p = m.predict_proba(X_te)[:, 1]; t_pred = (time.perf_counter() - t0) / len(X_te) * 1000
        fitted[name], oofs[name], test_probs[name] = m, oof, p

        met = binary_metrics(y_te.values, p, thr)
        rows_test.append({"Модель": name, "Порог": thr, **met,
                          "Время обучения, с": t_fit, "Инференс, мс / 1000 записей": t_pred * 1000})
        oof_pred = (oof >= thr).astype(int)
        rows_cv.append({"Модель": name,
                        "CV ROC-AUC (mean)": f_auc.mean(), "CV ROC-AUC (std)": f_auc.std(),
                        "CV PR-AUC (mean)": f_ap.mean(), "CV PR-AUC (std)": f_ap.std(),
                        "CV F1 (OOF, подобр. порог)": f1_score(y_tr, oof_pred),
                        "CV Recall (OOF)": recall_score(y_tr, oof_pred),
                        "CV Precision (OOF)": precision_score(y_tr, oof_pred, zero_division=0),
                        "Среднее время обучения фолда, с": t_fit_cv})
        pred_c = (p >= thr_cost).astype(int)
        base = int(y_te.sum()) * COST_UNPLANNED_FAILURE
        cost = cost_of(y_te.values, pred_c)
        rows_biz.append({"Модель": name, "Порог (мин. затрат)": thr_cost,
                         "Отказов в тесте": int(y_te.sum()),
                         "Предсказано заранее (TP)": int(((pred_c == 1) & (y_te.values == 1)).sum()),
                         "Пропущено (FN)": int(((pred_c == 0) & (y_te.values == 1)).sum()),
                         "Ложных тревог (FP)": int(((pred_c == 1) & (y_te.values == 0)).sum()),
                         "Затраты без системы, руб.": base, "Затраты с системой, руб.": cost,
                         "Экономия, руб.": base - cost, "Экономия, %": (base - cost) / base})
        fpr, tpr, _ = roc_curve(y_te, p)
        pr, rc, _ = precision_recall_curve(y_te, p)
        curves[name] = {"fpr": fpr, "tpr": tpr, "prec": pr, "rec": rc}

    for name, model in models.items():
        evaluate(name, model)

    # ---------- 1c. Ансамбль топ-3 по CV PR-AUC ----------
    cv_df = pd.DataFrame(rows_cv).sort_values("CV PR-AUC (mean)", ascending=False)
    top3 = cv_df["Модель"].head(3).tolist()
    ens = VotingClassifier([(f"m{i}", clone(models[n])) for i, n in enumerate(top3)], voting="soft")
    ens_name = "Ансамбль (soft voting: " + " + ".join(top3) + ")"
    models[ens_name] = ens
    evaluate(ens_name, ens)

    test_df = pd.DataFrame(rows_test).sort_values("PR-AUC", ascending=False).reset_index(drop=True)
    cv_df = pd.DataFrame(rows_cv).sort_values("CV PR-AUC (mean)", ascending=False).reset_index(drop=True)
    biz_df = pd.DataFrame(rows_biz).sort_values("Экономия, руб.", ascending=False).reset_index(drop=True)
    test_df.insert(0, "Ранг", range(1, len(test_df) + 1))

    # Выбор лучшей модели: по CV PR-AUC (не по тесту — без подгонки под тест)
    best_name = cv_df.iloc[0]["Модель"]
    log(f"Лучшая модель по CV PR-AUC: {best_name}")
    best_model = fitted[best_name]
    best_thr = float(test_df.loc[test_df["Модель"] == best_name, "Порог"].iloc[0])
    joblib.dump({"model": best_model, "threshold": best_thr, "features": FEATURES}, MODELS_DIR / "best_binary_model.joblib")

    # ---------- 1d. Вклад feature engineering ----------
    log("Сравнение: исходные признаки vs + инженерные признаки")
    fe_rows = []
    for name, model in binary_models(pos_weight).items():
        _, auc_raw, ap_raw, _ = cv_binary(model, X_tr[RAW_FEATURES], y_tr, skf)
        r = cv_df[cv_df["Модель"] == name].iloc[0]
        fe_rows.append({"Модель": name,
                        "CV PR-AUC: 6 исходных признаков": ap_raw.mean(),
                        "CV PR-AUC: + 3 инженерных признака": r["CV PR-AUC (mean)"],
                        "Прирост PR-AUC": r["CV PR-AUC (mean)"] - ap_raw.mean(),
                        "CV ROC-AUC: 6 исходных": auc_raw.mean(),
                        "CV ROC-AUC: + инженерные": r["CV ROC-AUC (mean)"]})
    fe_df = pd.DataFrame(fe_rows).sort_values("Прирост PR-AUC", ascending=False)

    # ---------- 1e. Важность признаков ----------
    log("Permutation importance")
    pi = permutation_importance(best_model, X_te, y_te, scoring="average_precision", n_repeats=15,
                                random_state=RANDOM_STATE, n_jobs=1)
    imp_df = pd.DataFrame({"Признак": FEATURES, "Описание": [FEATURE_RU[f] for f in FEATURES],
                           "Падение PR-AUC при перемешивании (mean)": pi.importances_mean,
                           "std": pi.importances_std}).sort_values("Падение PR-AUC при перемешивании (mean)", ascending=False)

    # ---------- 2. Мультикласс: диагностика типа отказа ----------
    le = LabelEncoder()
    yt = df["failure_type"]
    classes_order = ["Нет отказа", "TWF", "HDF", "PWF", "OSF", "Прочее"]
    le.fit(classes_order)
    y_mc = le.transform(yt)
    y_mc_tr, y_mc_te = le.transform(yt.loc[idx_tr]), le.transform(yt.loc[idx_te])

    mc_rows, mc_fitted, mc_preds = [], {}, {}
    for name, model in multiclass_models().items():
        log(f"Мультикласс: {name}")
        _, f1s = cv_multiclass(model, X_tr, y_mc_tr, skf)
        m = clone(model)
        t0 = time.perf_counter(); m.fit(X_tr, y_mc_tr); t_fit = time.perf_counter() - t0
        pred = np.asarray(m.predict(X_te)).ravel().astype(int)
        mc_fitted[name], mc_preds[name] = m, pred
        fail_mask = y_mc_te != le.transform(["Нет отказа"])[0]
        mc_rows.append({"Модель": name,
                        "CV Macro-F1 (mean)": f1s.mean(), "CV Macro-F1 (std)": f1s.std(),
                        "Test Macro-F1": f1_score(y_mc_te, pred, average="macro"),
                        "Test Macro-F1 (без случайных отказов)": f1_score(y_mc_te, pred, average="macro", labels=le.transform([c for c in classes_order if c != "Прочее"])),
                        "Test Weighted-F1": f1_score(y_mc_te, pred, average="weighted"),
                        "Test Balanced Accuracy": balanced_accuracy_score(y_mc_te, pred),
                        "Test Accuracy": accuracy_score(y_mc_te, pred),
                        "Верно определён тип среди реальных отказов": accuracy_score(y_mc_te[fail_mask], pred[fail_mask]),
                        "Время обучения, с": t_fit})
    mc_df = pd.DataFrame(mc_rows).sort_values("CV Macro-F1 (mean)", ascending=False).reset_index(drop=True)
    mc_df.insert(0, "Ранг", range(1, len(mc_df) + 1))
    best_mc = mc_df.iloc[0]["Модель"]
    log(f"Лучшая мультикласс-модель: {best_mc}")
    joblib.dump({"model": mc_fitted[best_mc], "label_encoder": le, "features": FEATURES}, MODELS_DIR / "best_failure_type_model.joblib")

    rep = classification_report(y_mc_te, mc_preds[best_mc], labels=le.transform(classes_order),
                                target_names=[FAILURE_RU[c] for c in classes_order], output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(rep).T.reset_index().rename(columns={"index": "Класс", "precision": "Precision",
                                                               "recall": "Recall", "f1-score": "F1", "support": "Кол-во в тесте"})
    cm_mc = confusion_matrix(y_mc_te, mc_preds[best_mc], labels=le.transform(classes_order))
    cm_mc_df = pd.DataFrame(cm_mc, index=[f"Факт: {c}" for c in classes_order], columns=[f"Прогноз: {c}" for c in classes_order])

    cm_bin = confusion_matrix(y_te, (test_probs[best_name] >= best_thr).astype(int), labels=[0, 1])
    cm_bin_df = pd.DataFrame(cm_bin, index=["Факт: норма", "Факт: отказ"], columns=["Прогноз: норма", "Прогноз: отказ"])

    # ---------- 3. Графики ----------
    log("Графики")
    make_figures(test_df, curves, cm_bin_df, cm_mc_df, imp_df, mc_df, fe_df, best_name, best_mc)

    # ---------- 4. Excel ----------
    log("Excel")
    metrics_df = metrics_glossary()
    summary = summary_sheet(test_df, cv_df, mc_df, biz_df, best_name, best_mc)
    test_df.to_csv(RES / "metrics_binary.csv", index=False, encoding="utf-8-sig")
    mc_df.to_csv(RES / "metrics_multiclass.csv", index=False, encoding="utf-8-sig")
    sheets = {
        "Итоги": summary,
        "Бинарная — тест": test_df,
        "Бинарная — CV": cv_df,
        "Экономика": biz_df,
        "Тип отказа — модели": mc_df,
        "Тип отказа — по классам": rep_df,
        "Матрицы ошибок": None,
        "Важность признаков": imp_df,
        "Feature engineering": fe_df,
        "Гиперпараметры": pd.DataFrame(tuned_params),
        "Данные": dataset_profile(df),
        "Описание метрик": metrics_df,
    }
    write_excel(RES / "metrics_models.xlsx", sheets, cm_bin_df, cm_mc_df, best_name, best_mc)

    # ---------- 5. Данные для демо ----------
    log("Экспорт демо-данных")
    export_demo(df, idx_te, X_te, y_te, test_df, cv_df, mc_df, biz_df, imp_df, fe_df, curves, cm_bin_df, cm_mc_df, rep_df,
                test_probs[best_name], best_thr, best_name, mc_preds[best_mc], le, best_mc, X_tr, y_tr, y_mc_tr, pos_weight)
    log("Готово")


# ----------------------------------------------------------------------------------------
# Отчётность
# ----------------------------------------------------------------------------------------
def metrics_glossary() -> pd.DataFrame:
    return pd.DataFrame([
        ("ROC-AUC", "Способность модели ранжировать: насколько отказы получают более высокий риск, чем норма. 1.0 — идеально, 0.5 — случайно."),
        ("PR-AUC", "Главная метрика при редких отказах (3.4%). Площадь под кривой точность–полнота. Случайная модель ≈ 0.034."),
        ("Recall (полнота)", "Доля реальных отказов, которые система предсказала заранее."),
        ("Precision (точность)", "Доля тревог системы, которые оказались реальными отказами."),
        ("F1", "Гармоническое среднее точности и полноты."),
        ("F2", "Как F1, но полнота в 2 раза важнее точности — пропуск отказа дороже ложной тревоги."),
        ("MCC", "Коэффициент корреляции Мэтьюса: устойчивая к дисбалансу сводная метрика, от −1 до 1."),
        ("Balanced Accuracy", "Средняя доля верных ответов по классам «норма» и «отказ»."),
        ("Accuracy", "Доля верных ответов. При дисбалансе вводит в заблуждение: «всегда норма» даёт 96.6%."),
        ("Brier score", "Качество калибровки вероятностей, меньше — лучше."),
        ("Порог", "Вероятность, выше которой поднимается тревога. Подбирается на CV (out-of-fold), а не на тесте."),
        ("Порог (мин. затрат)", "Порог, минимизирующий суммарные затраты при заданной стоимости пропуска/ложной тревоги."),
        ("Macro-F1", "Среднее F1 по всем типам отказа без учёта их частоты — важно для редких типов."),
        ("CV mean / std", "Среднее и разброс метрики по 5 фолдам кросс-валидации — оценка стабильности."),
        ("Экономика", f"Допущения: пропущенный отказ {COST_UNPLANNED_FAILURE:,} ₽, плановый ремонт {COST_PLANNED_REPAIR:,} ₽, ложная тревога {COST_FALSE_ALARM:,} ₽ (настраивается в pdm_pipeline.py)."),
    ], columns=["Метрика", "Что означает"])


def summary_sheet(test_df, cv_df, mc_df, biz_df, best_name, best_mc) -> pd.DataFrame:
    b = test_df[test_df["Модель"] == best_name].iloc[0]
    c = cv_df[cv_df["Модель"] == best_name].iloc[0]
    m = mc_df[mc_df["Модель"] == best_mc].iloc[0]
    z = biz_df[biz_df["Модель"] == best_name].iloc[0]
    return pd.DataFrame([
        ("ЗАДАЧА 1: прогноз отказа", ""),
        ("Лучшая модель (выбор по CV PR-AUC)", best_name),
        ("CV PR-AUC", f"{c['CV PR-AUC (mean)']:.3f} ± {c['CV PR-AUC (std)']:.3f}"),
        ("CV ROC-AUC", f"{c['CV ROC-AUC (mean)']:.3f} ± {c['CV ROC-AUC (std)']:.3f}"),
        ("Test PR-AUC", f"{b['PR-AUC']:.3f}"),
        ("Test ROC-AUC", f"{b['ROC-AUC']:.3f}"),
        ("Test Recall — доля пойманных отказов", f"{b['Recall (доля пойманных отказов)']:.1%}  ({b['TP']} из {b['TP'] + b['FN']})"),
        ("Test Precision — доля верных тревог", f"{b['Precision (точность тревог)']:.1%}  (ложных тревог: {b['FP']} на {len_test(b)} записей)"),
        ("Test F1", f"{b['F1']:.3f}"),
        ("Базовая модель (Logistic Regression), Test PR-AUC", f"{test_df[test_df['Модель'] == 'Logistic Regression']['PR-AUC'].iloc[0]:.3f}"),
        ("", ""),
        ("ЗАДАЧА 2: диагностика типа отказа", ""),
        ("Лучшая модель (выбор по CV Macro-F1)", best_mc),
        ("CV Macro-F1", f"{m['CV Macro-F1 (mean)']:.3f} ± {m['CV Macro-F1 (std)']:.3f}"),
        ("Test Macro-F1", f"{m['Test Macro-F1']:.3f}"),
        ("Test Macro-F1 без случайных отказов (RNF непредсказуемы по датчикам)", f"{m['Test Macro-F1 (без случайных отказов)']:.3f}"),
        ("Верно определён тип среди реальных отказов", f"{m['Верно определён тип среди реальных отказов']:.1%}"),
        ("", ""),
        ("ЭКОНОМИЧЕСКИЙ ЭФФЕКТ (условный, тест 2000 записей, порог минимальных затрат)", ""),
        ("Порог минимальных затрат", f"{z['Порог (мин. затрат)']:.4f}"),
        ("Предсказано / пропущено / ложных тревог", f"{z['Предсказано заранее (TP)']} / {z['Пропущено (FN)']} / {z['Ложных тревог (FP)']}"),
        ("Затраты без системы (все отказы аварийные)", f"{z['Затраты без системы, руб.']:,.0f} ₽"),
        ("Затраты с системой", f"{z['Затраты с системой, руб.']:,.0f} ₽"),
        ("Экономия", f"{z['Экономия, руб.']:,.0f} ₽  ({z['Экономия, %']:.1%})"),
    ], columns=["Показатель", "Значение"])


def len_test(b):
    return int(b["TP"] + b["FP"] + b["FN"] + b["TN"])


def make_figures(test_df, curves, cm_bin_df, cm_mc_df, imp_df, mc_df, fe_df, best_name, best_mc):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    top = test_df["Модель"].head(6).tolist()
    if "Logistic Regression" not in top:
        top.append("Logistic Regression")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for n in top:
        c = curves[n]
        lw = 2.6 if n == best_name else 1.3
        short = n if len(n) < 34 else n[:31] + "…"
        ax[0].plot(c["fpr"], c["tpr"], lw=lw, label=short)
        ax[1].plot(c["rec"], c["prec"], lw=lw, label=short)
    ax[0].plot([0, 1], [0, 1], "--", color="grey", lw=0.8)
    ax[0].set(title="ROC-кривые (test)", xlabel="Доля ложных тревог (FPR)", ylabel="Доля пойманных отказов (TPR)")
    ax[1].set(title="Precision–Recall (test)", xlabel="Recall", ylabel="Precision")
    ax[1].legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "roc_pr_curves.png", dpi=150); plt.close(fig)

    d = test_df.sort_values("PR-AUC")
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(d) + 1))
    colors = ["#2a6fdb" if n == best_name else "#9db4d6" for n in d["Модель"]]
    ax.barh([n if len(n) < 45 else n[:42] + "…" for n in d["Модель"]], d["PR-AUC"], color=colors)
    for i, v in enumerate(d["PR-AUC"]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set(title="Сравнение моделей: PR-AUC на тестовой выборке", xlabel="PR-AUC", xlim=(0, 1.08))
    fig.tight_layout(); fig.savefig(FIG / "model_comparison.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1.8]})
    for a, cm, title in ((ax[0], cm_bin_df, f"Отказ / норма\n{best_name}"), (ax[1], cm_mc_df, f"Тип отказа\n{best_mc}")):
        v = cm.values
        a.imshow(np.log1p(v), cmap="Blues")
        a.set_xticks(range(v.shape[1])); a.set_yticks(range(v.shape[0]))
        a.set_xticklabels([c.replace("Прогноз: ", "") for c in cm.columns], rotation=30, ha="right")
        a.set_yticklabels([c.replace("Факт: ", "") for c in cm.index])
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                a.text(j, i, v[i, j], ha="center", va="center", color="white" if np.log1p(v[i, j]) > np.log1p(v.max()) * 0.6 else "black")
        a.set(title=title, xlabel="Прогноз", ylabel="Факт")
        a.spines[:].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "confusion_matrices.png", dpi=150); plt.close(fig)

    d = imp_df.sort_values("Падение PR-AUC при перемешивании (mean)")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.barh(d["Описание"], d["Падение PR-AUC при перемешивании (mean)"], xerr=d["std"], color="#2a6fdb")
    ax.set(title=f"Важность признаков (permutation importance)\n{best_name}", xlabel="Падение PR-AUC при перемешивании признака")
    fig.tight_layout(); fig.savefig(FIG / "feature_importance.png", dpi=150); plt.close(fig)

    d = mc_df.sort_values("Test Macro-F1")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.barh(d["Модель"], d["Test Macro-F1"], color=["#2a6fdb" if n == best_mc else "#9db4d6" for n in d["Модель"]])
    for i, v in enumerate(d["Test Macro-F1"]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set(title="Диагностика типа отказа: Macro-F1 на тесте", xlabel="Macro-F1", xlim=(0, 1.08))
    fig.tight_layout(); fig.savefig(FIG / "multiclass_comparison.png", dpi=150); plt.close(fig)


def write_excel(path, sheets, cm_bin_df, cm_mc_df, best_name, best_mc):
    from openpyxl.drawing.image import Image as XLImage
    from openpyxl.formatting.rule import ColorScaleRule
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    head_fill = PatternFill("solid", fgColor="1F3864")
    best_fill = PatternFill("solid", fgColor="E2EFDA")
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, d in sheets.items():
            if d is None:
                cm_bin_df.to_excel(xw, sheet_name=name, startrow=1)
                cm_mc_df.to_excel(xw, sheet_name=name, startrow=len(cm_bin_df) + 5)
                ws = xw.sheets[name]
                ws.cell(1, 1, f"Прогноз отказа — {best_name} (test)").font = Font(bold=True, size=12)
                ws.cell(len(cm_bin_df) + 5, 1, f"Тип отказа — {best_mc} (test)").font = Font(bold=True, size=12)
                ws.column_dimensions["A"].width = 22
                for col in range(2, 9):
                    ws.column_dimensions[get_column_letter(col)].width = 16
                continue
            d.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = head_fill
                cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
            ws.row_dimensions[1].height = 45
            ws.freeze_panes = "B2" if name not in ("Итоги", "Данные", "Описание метрик") else "A2"
            for j, col in enumerate(d.columns, start=1):
                letter = get_column_letter(j)
                maxlen = max([len(str(col)) * 0.6] + [len(str(v)) for v in d[col].head(50)])
                ws.column_dimensions[letter].width = min(max(10, maxlen + 2), 90 if name in ("Описание метрик", "Итоги", "Данные", "Гиперпараметры") else 48)
                if pd.api.types.is_float_dtype(d[col]):
                    fmt = '#,##0" ₽"' if "руб" in str(col) else ("0.0%" if "%" in str(col) else ("0.0000" if "Порог" in str(col) else "0.000"))
                    for r in range(2, len(d) + 2):
                        ws.cell(r, j).number_format = fmt
                elif pd.api.types.is_integer_dtype(d[col]) and "руб" in str(col):
                    for r in range(2, len(d) + 2):
                        ws.cell(r, j).number_format = '#,##0" ₽"'
                if any(k in str(col) for k in ("AUC", "F1", "Recall", "MCC", "Balanced", "Экономия, %", "Верно")) and "std" not in str(col) and len(d) > 2:
                    ws.conditional_formatting.add(f"{letter}2:{letter}{len(d) + 1}",
                                                  ColorScaleRule(start_type="min", start_color="F8696B", mid_type="percentile",
                                                                 mid_value=50, mid_color="FFEB84", end_type="max", end_color="63BE7B"))
            if "Модель" in d.columns:
                for r, v in enumerate(d["Модель"], start=2):
                    if v in (best_name, best_mc) and (("Тип" in name) == (v == best_mc) or best_name == best_mc):
                        for c in range(1, len(d.columns) + 1):
                            if ws.cell(r, c).fill.fgColor.rgb in (None, "00000000"):
                                ws.cell(r, c).fill = best_fill
                        ws.cell(r, d.columns.get_loc("Модель") + 1).font = Font(bold=True)
            if name == "Итоги":
                for r in range(2, len(d) + 2):
                    if str(ws.cell(r, 1).value).isupper() or str(ws.cell(r, 1).value).startswith(("ЗАДАЧА", "ЭКОНОМ")):
                        ws.cell(r, 1).font = Font(bold=True, size=12, color="1F3864")
            if name in ("Описание метрик", "Гиперпараметры"):
                for row in ws.iter_rows(min_row=2):
                    for cell in row:
                        cell.alignment = Alignment(wrap_text=True, vertical="top")

        ws = xw.book.create_sheet("Графики")
        ws.cell(1, 1, "Графики качества моделей").font = Font(bold=True, size=14)
        row = 3
        for f, h in (("model_comparison.png", 30), ("roc_pr_curves.png", 28), ("confusion_matrices.png", 30),
                     ("feature_importance.png", 26), ("multiclass_comparison.png", 25)):
            img = XLImage(str(FIG / f))
            img.width, img.height = img.width * 0.55, img.height * 0.55
            ws.add_image(img, f"A{row}")
            row += h


def export_demo(df, idx_te, X_te, y_te, test_df, cv_df, mc_df, biz_df, imp_df, fe_df, curves, cm_bin_df, cm_mc_df, rep_df,
                p_best, thr, best_name, mc_pred, le, best_mc, X_tr, y_tr, y_mc_tr, pos_weight):
    """JSON для интерактивной демонстрации + компактные модели LightGBM для «что-если» в браузере."""

    def r(x, n=4):
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)

    def thin(a, b, k=120):
        idx = np.unique(np.linspace(0, len(a) - 1, min(k, len(a))).astype(int))
        return [[r(a[i]), r(b[i])] for i in idx]

    top = test_df["Модель"].head(5).tolist() + (["Logistic Regression"] if "Logistic Regression" not in test_df["Модель"].head(5).tolist() else [])

    # Демо-модели для браузера: LightGBM (компактные деревья → JSON)
    demo_bin = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10, subsample=0.85,
                              subsample_freq=1, colsample_bytree=0.85, scale_pos_weight=np.sqrt(pos_weight), verbose=-1,
                              random_state=RANDOM_STATE).fit(X_tr, y_tr)
    demo_mc = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=5, class_weight="balanced",
                             verbose=-1, random_state=RANDOM_STATE).fit(X_tr, y_mc_tr)
    p_demo = demo_bin.predict_proba(X_te)[:, 1]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_demo, _, _, _ = cv_binary(demo_bin, X_tr, y_tr, skf)
    demo_thr = best_f1_threshold(y_tr, oof_demo)
    demo_quality = {"roc_auc": r(roc_auc_score(y_te, p_demo)), "pr_auc": r(average_precision_score(y_te, p_demo)),
                    "threshold": r(demo_thr),
                    "recall": r(recall_score(y_te, (p_demo >= demo_thr).astype(int))),
                    "precision": r(precision_score(y_te, (p_demo >= demo_thr).astype(int), zero_division=0)),
                    "mc_macro_f1": r(f1_score(le.transform(df.loc[idx_te, "failure_type"]), demo_mc.predict(X_te), average="macro"))}

    def dump_trees(model):
        trees = []
        for t in model.booster_.dump_model()["tree_info"]:
            nodes = []  # плоский массив: [feature, threshold, left, right] или [-1, leaf_value]

            def walk(n):
                i = len(nodes); nodes.append(None)
                if "leaf_value" in n:
                    nodes[i] = [-1, r(n["leaf_value"], 6)]
                else:
                    left = walk(n["left_child"]); right = walk(n["right_child"])
                    nodes[i] = [n["split_feature"], r(n["threshold"], 6), left, right]
                return i

            walk(t["tree_structure"])
            trees.append(nodes)
        return trees

    # Парк оборудования: тестовые записи с прогнозом лучшей модели
    fleet = df.loc[idx_te].copy()
    fleet["risk"] = p_best
    fleet["pred_type"] = le.inverse_transform(mc_pred)
    fleet = fleet.sort_values("risk", ascending=False)
    sample = pd.concat([fleet.head(60), fleet.iloc[60:].sample(140, random_state=RANDOM_STATE)]).sort_values("risk", ascending=False)

    best_row = test_df[test_df["Модель"] == best_name].iloc[0]
    demo = {
        "dataset": {"name": "AI4I 2020 Predictive Maintenance (UCI)", "rows": int(len(df)), "failures": int(df["failure"].sum()),
                    "failure_rate": r(df["failure"].mean()), "test_rows": int(len(idx_te)),
                    "type_counts": df["failure_type"].value_counts().to_dict()},
        "costs": {"unplanned": COST_UNPLANNED_FAILURE, "planned": COST_PLANNED_REPAIR, "false_alarm": COST_FALSE_ALARM},
        "best_binary": best_name, "best_multiclass": best_mc, "threshold": r(thr),
        "binary_test": [{k: (r(v) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (np.integer,)) else v))
                         for k, v in row.items()} for row in test_df.to_dict("records")],
        "binary_cv": [{k: (r(v) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()} for row in cv_df.to_dict("records")],
        "business": [{k: (r(v) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (np.integer,)) else v))
                      for k, v in row.items()} for row in biz_df.to_dict("records")],
        "multiclass": [{k: (r(v) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (np.integer,)) else v))
                        for k, v in row.items()} for row in mc_df.to_dict("records")],
        "per_class": [{k: (r(v) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()} for row in rep_df.to_dict("records")],
        "feature_importance": [{"feature": a, "label": b, "value": r(c)} for a, b, c in
                               zip(imp_df["Признак"], imp_df["Описание"], imp_df["Падение PR-AUC при перемешивании (mean)"])],
        "fe_gain": [{k: (r(v) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()} for row in fe_df.to_dict("records")],
        "curves": {n: {"roc": thin(curves[n]["fpr"], curves[n]["tpr"]), "pr": thin(curves[n]["rec"], curves[n]["prec"])} for n in top},
        "cm_binary": cm_bin_df.values.tolist(),
        "cm_multiclass": {"labels": [c.replace("Факт: ", "") for c in cm_mc_df.index], "values": cm_mc_df.values.tolist()},
        "test_probs": {"risk": [r(v, 4) for v in p_best], "y": y_te.astype(int).tolist()},
        "fleet": [{"id": str(rw["product_id"]), "type": rw["type"], "air": r(rw["air_temp_k"], 1), "proc": r(rw["process_temp_k"], 1),
                   "rpm": int(rw["rpm"]), "torque": r(rw["torque_nm"], 1), "wear": int(rw["tool_wear_min"]),
                   "risk": r(rw["risk"]), "pred_type": rw["pred_type"], "actual": int(rw["failure"]),
                   "actual_type": rw["failure_type"]} for _, rw in sample.iterrows()],
        "ranges": {f: [r(df[f].min(), 2), r(df[f].max(), 2), r(df[f].median(), 2)] for f in RAW_FEATURES},
        "demo_model": {"features": FEATURES, "quality": demo_quality,
                       "binary": {"trees": dump_trees(demo_bin)},
                       "multiclass": {"classes": le.classes_.tolist(), "trees": dump_trees(demo_mc)}},
    }
    with open(RES / "demo_data.json", "w", encoding="utf-8") as f:
        json.dump(demo, f, ensure_ascii=False, separators=(",", ":"))


if __name__ == "__main__":
    main()
