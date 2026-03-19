from projects.ecoqa import prepare_ecoqa_data as prep


class _DummyDataset:
    def __init__(self, data):
        self._data = data

    def __len__(self):
        return len(self._data)

    def get_data(self):
        return self._data


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
