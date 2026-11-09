"""
Единое описание признаков для обучения (pdm_pipeline.py) и инференса (predict.py).

Формулы признаков существуют только здесь: обучение и прогноз вызывают одну и ту же
функцию make_features(), поэтому признаки в проде не могут разойтись с обучением.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Исходные названия колонок датасета AI4I 2020 -> короткие имена
COLUMN_MAP = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "failure",
}

TYPE_MAP = {"L": 0, "M": 1, "H": 2}

# Что нужно подать на вход: класс изделия + 5 показаний датчиков
INPUT_COLUMNS = ["type", "air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min"]
SENSOR_COLUMNS = INPUT_COLUMNS[1:]

RAW_FEATURES = ["type_code", "air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min"]
ENG_FEATURES = ["temp_diff_k", "power_w", "strain_nm_min"]
FEATURES = RAW_FEATURES + ENG_FEATURES  # порядок важен: модель ждёт именно его

# Физически допустимые значения. Вне них запись считается ошибкой датчика, прогноз не делается.
PHYSICAL_LIMITS = {
    "air_temp_k": (250.0, 350.0),
    "process_temp_k": (250.0, 400.0),
    "rpm": (0.0, 10_000.0),
    "torque_nm": (0.0, 500.0),
    "tool_wear_min": (0.0, 1_000.0),
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит исходные названия колонок AI4I к коротким. Короткие имена остаются как есть."""
    return df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Возвращает копию df с добавленными признаками type_code, temp_diff_k, power_w, strain_nm_min.

    Принимает колонки в исходном формате AI4I или в коротком (INPUT_COLUMNS).
    """
    out = standardize_columns(df).copy()
    missing = [c for c in INPUT_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Нет обязательных колонок: {missing}. Нужны: {INPUT_COLUMNS}")
    out["type"] = out["type"].astype(str).str.strip().str.upper()
    for c in SENSOR_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["type_code"] = out["type"].map(TYPE_MAP)
    out["temp_diff_k"] = out["process_temp_k"] - out["air_temp_k"]        # теплоотвод -> HDF
    out["power_w"] = out["torque_nm"] * out["rpm"] * 2 * np.pi / 60        # мощность на валу -> PWF
    out["strain_nm_min"] = out["tool_wear_min"] * out["torque_nm"]         # перегрузка -> OSF
    return out


# Библиотеки, от версии которых зависит загрузка сохранённых моделей (joblib/pickle)
CRITICAL_LIBS = ["scikit-learn", "lightgbm", "catboost", "xgboost", "numpy", "joblib"]


def runtime_versions() -> dict:
    """Версии Python и библиотек в текущем окружении."""
    import platform
    from importlib import metadata

    out = {"python": platform.python_version()}
    for lib in CRITICAL_LIBS + ["pandas"]:
        try:
            out[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            out[lib] = None
    return out


def training_ranges(df: pd.DataFrame) -> dict:
    """Диапазоны входных признаков в обучающей выборке (сохраняются вместе с моделью)."""
    return {c: [float(df[c].min()), float(df[c].max())] for c in SENSOR_COLUMNS}


def validate(feat: pd.DataFrame, train_ranges: dict | None = None) -> tuple[pd.Series, pd.Series]:
    """Проверяет записи после make_features().

    Возвращает (valid, notes):
      valid — False, если прогноз делать нельзя (пропуск, неизвестный класс, физически невозможное значение);
      notes — текст проблем; для valid-записей вне диапазона обучения — предупреждение.
    """
    notes = pd.Series([[] for _ in range(len(feat))], index=feat.index, dtype=object)
    valid = pd.Series(True, index=feat.index)

    bad_type = feat["type_code"].isna()
    valid &= ~bad_type
    for i in feat.index[bad_type]:
        notes[i].append(f"неизвестный класс изделия '{feat.at[i, 'type']}' (ожидается L/M/H)")

    for c in SENSOR_COLUMNS:
        lo, hi = PHYSICAL_LIMITS[c]
        nan = feat[c].isna()
        out = ~nan & ((feat[c] < lo) | (feat[c] > hi))
        valid &= ~(nan | out)
        for i in feat.index[nan]:
            notes[i].append(f"{c}: нет значения")
        for i in feat.index[out]:
            notes[i].append(f"{c}={feat.at[i, c]}: физически невозможно, проверьте датчик")
        if train_ranges and c in train_ranges:
            tlo, thi = train_ranges[c]
            far = ~nan & ~out & ((feat[c] < tlo) | (feat[c] > thi))
            for i in feat.index[far]:
                notes[i].append(f"{c}={feat.at[i, c]}: вне диапазона обучения [{tlo:g}; {thi:g}], прогноз менее надёжен")

    return valid, notes.map("; ".join)
