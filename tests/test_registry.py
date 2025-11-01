from pathlib import Path

from neurose_vton.registry import registry


def test_registry_report_runs_without_models(tmp_path: Path):
    # Should not raise even if no models exist; paths may be empty
    rep = registry.report()
    assert isinstance(rep, dict)
    assert "insightface" in rep
