import pytest
import joblib
from food_intelligence.elasticity import simulate_price_change
from food_intelligence.train import atomic_dump

@pytest.mark.parametrize("args", [(-2, 1e-300, 1e300, 1), (-2, 1e300, 1e-300, 1), (-1, 0, 2, 1), (float("nan"), 1, 2, 1)])
def test_simulation_rejects_unsafe_numbers(args):
    with pytest.raises(ValueError):
        simulate_price_change(*args)

def test_atomic_dump_failure_preserves_previous_model(tmp_path, monkeypatch):
    path = tmp_path / "model.joblib"
    atomic_dump({"version": 1}, path)
    def fail(*args):
        raise OSError("disk full")
    monkeypatch.setattr(joblib, "dump", fail)
    with pytest.raises(OSError):
        atomic_dump({"version": 2}, path)
    assert joblib.load(path) == {"version": 1}
    assert list(tmp_path.iterdir()) == [path]
