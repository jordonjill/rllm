from projects.ecoqa import prepare_ecoqa_data as prep
from projects.ecoqa.train_ecoqa_curriculum import _infer_sql_difficulty


class _DummyDataset:
    def __init__(self, data):
        self._data = data

    def __len__(self):
        return len(self._data)

    def get_data(self):
        return self._data


def test_curriculum_requires_calculator_increases_difficulty_bucket():
    base = {
        "ground_truth_sql": "SELECT month FROM exchange_rates WHERE year = 2024 ORDER BY month DESC",
        "requires_calculator": False,
    }
    calc = dict(base)
    calc["requires_calculator"] = True

    assert _infer_sql_difficulty(base) == "easy"
    assert _infer_sql_difficulty(calc) == "medium"


def test_curriculum_structure_items_and_dims_increase_difficulty():
    base = {
        "ground_truth_sql": "SELECT month FROM exchange_rates ORDER BY month DESC",
        "requires_calculator": False,
        "ground_truth": '{"items":[{"name":"month","value":12}]}',
    }
    richer = {
        **base,
        "ground_truth": (
            '{"items":[{"name":"month","value":12,"dims":{"year":2024}},'
            '{"name":"month","value":11,"dims":{"year":2024}}]}'
        ),
    }

    assert _infer_sql_difficulty(base) == "easy"
    assert _infer_sql_difficulty(richer) == "medium"


def test_prepare_data_keeps_requires_calculator_field(monkeypatch):
    captured: dict[str, list[dict]] = {}

    def _fake_register_dataset(cls, name, data, split, source="", description="", category=""):
        captured[split] = data
        return _DummyDataset(data)

    monkeypatch.setattr(prep.DatasetRegistry, "register_dataset", classmethod(_fake_register_dataset))

    train_dataset, val_dataset, test_dataset = prep.prepare_ecoqa_data()

    assert len(train_dataset) > 0 and len(val_dataset) > 0 and len(test_dataset) > 0
    assert "train" in captured and captured["train"]
    assert all("requires_calculator" in row for row in captured["train"])
    assert all(isinstance(row["requires_calculator"], bool) for row in captured["train"])
    assert any(row["requires_calculator"] for row in captured["train"])
