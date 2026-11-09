"""Тесты инференса: признаки, воспроизводимость метрик, проверка входа, защита загрузки модели."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pdm_features import COLUMN_MAP, FEATURES, make_features  # noqa: E402
from predict import ModelIntegrityError, Predictor, check_environment, main  # noqa: E402

RAW = pd.read_csv(ROOT / "data" / "raw" / "ai4i2020.csv", encoding="utf-8-sig")
GOOD = {"type": "L", "air_temp_k": 300.0, "process_temp_k": 310.0, "rpm": 1320, "torque_nm": 58.0, "tool_wear_min": 205}


@pytest.fixture(scope="module")
def predictor():
    return Predictor.load(ROOT / "models")


def test_feature_formulas():
    f = make_features(pd.DataFrame([{**GOOD, "type": " l "}])).iloc[0]
    assert f["type_code"] == 0
    assert f["temp_diff_k"] == pytest.approx(10.0)
    assert f["power_w"] == pytest.approx(58 * 1320 * 2 * np.pi / 60)
    assert f["strain_nm_min"] == pytest.approx(205 * 58)


def test_ai4i_and_short_column_names_give_same_features():
    a = make_features(RAW.head(200))[FEATURES]
    b = make_features(RAW.head(200).rename(columns=COLUMN_MAP))[FEATURES]
    pd.testing.assert_frame_equal(a, b)


def test_missing_column_raises():
    with pytest.raises(ValueError, match="Нет обязательных колонок"):
        make_features(pd.DataFrame([{k: v for k, v in GOOD.items() if k != "rpm"}]))


def test_predict_reproduces_test_metrics(predictor):
    """predict.py на сырых строках теста даёт ровно те TP/FP, что записаны в отчёте."""
    from sklearn.model_selection import train_test_split

    import pdm_pipeline as P

    df = P.load_data()
    _, idx_te = train_test_split(df.index, test_size=P.TEST_SIZE, stratify=df["failure_type"], random_state=P.RANDOM_STATE)
    res = predictor.predict(RAW.loc[idx_te])
    y = df.loc[idx_te, "failure"].values
    alarm = res["alarm"].values

    m = pd.read_csv(ROOT / "results" / "metrics_binary.csv", encoding="utf-8-sig")
    row = m[m["Модель"] == predictor.binary["model_name"]].iloc[0]
    assert (res["status"] != "ошибка данных").all()
    assert int((alarm & (y == 1)).sum()) == row["TP"]
    assert int((alarm & (y == 0)).sum()) == row["FP"]


def test_known_overload_case(predictor):
    r = predictor.predict(pd.DataFrame([GOOD])).iloc[0]
    assert r["status"] == "тревога"
    assert r["likely_cause"] == "OSF"


def test_no_cause_for_normal_machines(predictor):
    res = predictor.predict(RAW.head(5))
    assert (res["status"] == "норма").all()
    assert res["likely_cause"].isna().all() and res["likely_cause_p"].isna().all()


def test_invalid_rows_are_not_predicted(predictor):
    rows = pd.DataFrame([
        {**GOOD, "rpm": None},               # пропуск
        {**GOOD, "type": "X"},               # неизвестный класс
        {**GOOD, "torque_nm": -5},           # физически невозможно
        {**GOOD, "air_temp_k": 320.0},       # вне обучения, но физически возможно
    ])
    res = predictor.predict(rows)
    assert list(res["status"][:3]) == ["ошибка данных"] * 3
    assert res["risk"][:3].isna().all()
    assert res["status"][3] != "ошибка данных"
    assert "вне диапазона обучения" in res["notes"][3]


def test_tampered_model_is_not_loaded(tmp_path):
    d = tmp_path / "models"
    shutil.copytree(ROOT / "models", d)
    with open(d / "best_binary_model.joblib", "ab") as f:
        f.write(b"\0")
    with pytest.raises(ModelIntegrityError, match="контрольная сумма"):
        Predictor.load(d)


def test_missing_manifest_is_refused(tmp_path):
    d = tmp_path / "models"
    shutil.copytree(ROOT / "models", d)
    (d / "manifest.json").unlink()
    with pytest.raises(ModelIntegrityError, match="манифест"):
        Predictor.load(d)


def test_version_mismatch():
    manifest = json.loads((ROOT / "models" / "manifest.json").read_text(encoding="utf-8"))
    assert check_environment(manifest) == []
    manifest["versions"]["scikit-learn"] = "0.0.1"
    with pytest.raises(ModelIntegrityError, match="scikit-learn"):
        check_environment(manifest)
    with pytest.warns(UserWarning):
        assert len(check_environment(manifest, allow_mismatch=True)) == 1


def test_cli_json(capsys):
    assert main(["--json", json.dumps(GOOD)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "тревога" and out["likely_cause"] == "OSF"
