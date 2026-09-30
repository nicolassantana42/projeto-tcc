"""An installation upgrade preserves the user's configured evidence directory."""

from epi_monitor.ui.alert_panels import reports_root


def test_explicit_evidence_directory_takes_priority(tmp_path, monkeypatch):
    current, legacy = tmp_path / "current", tmp_path / "legacy"
    monkeypatch.setenv("EPI_REPORTS_DIR", str(current))
    monkeypatch.setenv("SAFEGUARD_REPORTS_DIR", str(legacy))
    assert reports_root() == current.resolve()


def test_existing_evidence_directory_configuration_is_preserved(tmp_path, monkeypatch):
    monkeypatch.delenv("EPI_REPORTS_DIR", raising=False)
    monkeypatch.setenv("SAFEGUARD_REPORTS_DIR", str(tmp_path))
    assert reports_root() == tmp_path.resolve()


def test_default_evidence_directory_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv("EPI_REPORTS_DIR", raising=False)
    monkeypatch.delenv("SAFEGUARD_REPORTS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    assert reports_root() == tmp_path / "reports"
