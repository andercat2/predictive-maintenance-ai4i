"""
Инференс: прогноз отказа станка и диагностика причины.

Примеры:
    # пакетный прогноз по CSV (колонки AI4I или короткие: type, air_temp_k, process_temp_k, rpm, torque_nm, tool_wear_min)
    python predict.py --input sensors.csv --output predictions.csv

    # одна запись
    python predict.py --json '{"type": "L", "air_temp_k": 300, "process_temp_k": 310, "rpm": 1320, "torque_nm": 58, "tool_wear_min": 205}'

Из кода:
    from predict import Predictor
    pred = Predictor.load("models")
    result = pred.predict(df)

Перед загрузкой моделей проверяется models/manifest.json:
  1) SHA-256 каждого файла модели. joblib — это pickle, при загрузке он может выполнить код,
     поэтому подменённый или повреждённый файл не открывается вообще;
  2) версии библиотек. Модель, сохранённая в другой версии scikit-learn/LightGBM/CatBoost,
     может не загрузиться или молча считать иначе. При расхождении загрузка прерывается
     (обойти: --allow-version-mismatch, только для отладки).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from pdm_features import CRITICAL_LIBS, make_features, runtime_versions, validate

BINARY_FILE = "best_binary_model.joblib"
TYPE_FILE = "best_failure_type_model.joblib"
NO_FAILURE = "Нет отказа"

FAILURE_RU = {
    "TWF": "износ инструмента",
    "HDF": "перегрев (теплоотвод)",
    "PWF": "мощность вне допуска",
    "OSF": "механическая перегрузка",
    "Прочее": "случайный отказ",
}


class ModelIntegrityError(RuntimeError):
    """Файл модели не совпадает с манифестом или окружение несовместимо."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_environment(manifest: dict, allow_mismatch: bool = False) -> list[str]:
    """Сравнивает версии библиотек из манифеста с текущими. Возвращает список расхождений."""
    saved, now = manifest.get("versions", {}), runtime_versions()
    issues = [f"{lib}: модель сохранена в {saved.get(lib)}, установлено {now.get(lib)}"
              for lib in CRITICAL_LIBS if saved.get(lib) != now.get(lib)]
    if issues and not allow_mismatch:
        raise ModelIntegrityError(
            "Версии библиотек не совпадают с теми, в которых сохранена модель:\n  " + "\n  ".join(issues)
            + "\nУстановите точные версии: pip install -r requirements.txt")
    for i in issues:
        warnings.warn("Несовпадение версий: " + i)
    return issues


class Predictor:
    def __init__(self, binary: dict, failure_type: dict):
        self.binary = binary
        self.failure_type = failure_type
        self.model = binary["model"]
        self.threshold = float(binary["threshold"])
        self.watch_threshold = float(binary.get("watch_threshold", self.threshold / 3))
        self.features = binary["features"]
        self.train_ranges = binary.get("train_ranges")
        le = failure_type["label_encoder"]
        self.type_classes = list(le.inverse_transform(failure_type["model"].classes_))

    @classmethod
    def load(cls, models_dir: str | Path = "models", allow_version_mismatch: bool = False) -> "Predictor":
        import joblib

        models_dir = Path(models_dir)
        man_path = models_dir / "manifest.json"
        if not man_path.exists():
            raise ModelIntegrityError(f"Нет {man_path}: без манифеста модели не загружаются")
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
        bundles = {}
        for fname in (BINARY_FILE, TYPE_FILE):
            entry = manifest.get("models", {}).get(fname)
            path = models_dir / fname
            if entry is None or not path.exists():
                raise ModelIntegrityError(f"{fname}: нет файла или записи в манифесте")
            if _sha256(path) != entry["sha256"]:
                raise ModelIntegrityError(f"{fname}: контрольная сумма не совпадает с манифестом, файл не загружен")
        check_environment(manifest, allow_version_mismatch)
        for fname in (BINARY_FILE, TYPE_FILE):
            bundles[fname] = joblib.load(models_dir / fname)  # только после проверки SHA-256
        return cls(bundles[BINARY_FILE], bundles[TYPE_FILE])

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Прогноз для таблицы показаний. Возвращает исходные колонки + результат по каждой строке."""
        feat = make_features(df).reset_index(drop=True)
        valid, notes = validate(feat, self.train_ranges)
        n = len(feat)
        risk = np.full(n, np.nan)
        cause = np.full(n, None, dtype=object)
        cause_p = np.full(n, np.nan)

        if valid.any():
            X = feat.loc[valid, self.features]
            risk[valid.values] = self.model.predict_proba(X)[:, 1]
            # Вероятная причина: среди типов отказа (без класса «Нет отказа»), нормированно
            proba = self.failure_type["model"].predict_proba(X)
            fail_idx = [i for i, c in enumerate(self.type_classes) if c != NO_FAILURE]
            pf = proba[:, fail_idx]
            total = pf.sum(axis=1)
            pf = pf / np.clip(total[:, None], 1e-12, None)
            best = pf.argmax(axis=1)
            # Причину называем только при повышенном риске и ненулевом сигнале модели типа отказа
            show = (risk[valid.values] >= self.watch_threshold) & (total > 0)
            cause[valid.values] = [self.type_classes[fail_idx[j]] if s else None for j, s in zip(best, show)]
            cause_p[valid.values] = np.where(show, pf[np.arange(len(best)), best], np.nan)

        status = np.where(~valid.values, "ошибка данных",
                          np.where(risk >= self.threshold, "тревога",
                                   np.where(risk >= self.watch_threshold, "наблюдение", "норма")))
        out = df.reset_index(drop=True).copy()
        out["risk"] = np.round(risk, 4)
        out["status"] = status
        out["alarm"] = status == "тревога"
        out["likely_cause"] = cause
        out["likely_cause_ru"] = [FAILURE_RU.get(c) if c else None for c in cause]
        out["likely_cause_p"] = np.round(cause_p, 3)
        out["notes"] = notes.values
        return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Прогноз отказа станка и вероятной причины")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="CSV с показаниями")
    src.add_argument("--json", help="одна запись или список записей в JSON")
    ap.add_argument("--output", help="куда сохранить CSV с прогнозом (по умолчанию — вывод в консоль)")
    ap.add_argument("--models", default=str(Path(__file__).resolve().parent / "models"), help="папка с моделями")
    ap.add_argument("--allow-version-mismatch", action="store_true", help="загрузить при несовпадении версий (отладка)")
    args = ap.parse_args(argv)

    try:
        pred = Predictor.load(args.models, args.allow_version_mismatch)
    except ModelIntegrityError as e:
        print(f"Ошибка загрузки модели: {e}", file=sys.stderr)
        return 2

    if args.input:
        df = pd.read_csv(args.input, encoding="utf-8-sig")
    else:
        obj = json.loads(args.json)
        df = pd.DataFrame(obj if isinstance(obj, list) else [obj])

    res = pred.predict(df)
    if args.output:
        res.to_csv(args.output, index=False, encoding="utf-8-sig")
        counts = res["status"].value_counts().to_dict()
        print(f"Сохранено {len(res)} прогнозов в {args.output}: {counts}")
    elif args.json:
        cols = ["risk", "status", "alarm", "likely_cause", "likely_cause_ru", "likely_cause_p", "notes"]
        recs = res[cols].replace({np.nan: None}).to_dict("records")
        print(json.dumps(recs[0] if len(recs) == 1 else recs, ensure_ascii=False, indent=2, default=bool))
    else:
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(res.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
