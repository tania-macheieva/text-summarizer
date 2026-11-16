"""Спільні фікстури: ізольовані каталоги, швидка модель на демонстраційному наборі."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config  # noqa: E402
from app.data.loader import load_dataset  # noqa: E402
from app.data.splitter import split_dataset  # noqa: E402
from app.data.validation import validate_dataframe  # noqa: E402
from app.ml.training import train_model  # noqa: E402

SAMPLE_EN = (
    "Labour plans maternity pay rise\n\n"
    "Maternity pay for new mothers is to rise by £1,400 as part of new proposals announced by the Trade "
    "and Industry Secretary. It would mean paid leave would be increased to nine months by 2007. "
    "Other plans include letting maternity pay be given to fathers. The Tories dismissed the plan as "
    "\"desperate\", while the Liberal Democrats said it was misdirected.\n\n"
    "The government said the plans would be paid for by taxpayers, not employers. Business leaders warned "
    "that many small firms could be crippled by the move. Ministers will consult on other proposals later "
    "this year. The Conservatives will announce their own plans closer to the general election."
)
SAMPLE_UK = (
    "Уряд ухвалив нову програму підтримки малого бізнесу. Програма передбачає пільгові кредити для "
    "підприємців. Міністр економіки заявив, що кошти будуть доступні вже з наступного місяця. "
    "Експерти вважають, що ініціатива допоможе створити нові робочі місця. Опозиція критикує документ "
    "за недостатню прозорість. Остаточне рішення щодо бюджету програми ухвалять у парламенті."
)


@pytest.fixture(scope="session")
def demo_splits():
    df, _ = load_dataset(config.DEMO_DIR)
    df, _ = validate_dataframe(df)
    return split_dataset(df, seed=42)


@pytest.fixture(scope="session")
def trained_model(demo_splits):
    """Мала модель: демо-набір, мінімальна сітка (секунди)."""
    return train_model(demo_splits, "demo", grid={"n_estimators": [50], "max_depth": [6], "min_samples_leaf": [2]}, seed=42)


@pytest.fixture()
def client(tmp_path, trained_model, monkeypatch):
    """TestClient з ізольованими models/artifacts і підставленою навченою моделлю."""
    from fastapi.testclient import TestClient

    from app import main
    from app.services.registry import registry

    monkeypatch.setattr(config, "MODEL_FILE", tmp_path / "model.joblib")
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "artifacts")
    (tmp_path / "artifacts").mkdir()
    old = registry.model
    registry.model = trained_model
    with TestClient(main.app) as c:
        registry.model = trained_model  # lifespan міг підмінити
        yield c
    registry.model = old
