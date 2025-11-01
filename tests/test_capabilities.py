from neurose_vton.registry import registry
from neurose_vton.system import system_info


def test_system_info_stable():
    info = system_info()
    assert isinstance(info, dict)
    assert "cuda" in info


def test_registry_report_includes_aliases():
    rep = registry.report()
    assert "stableviton" in rep
    assert isinstance(rep["stableviton"], dict)

