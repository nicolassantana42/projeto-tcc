"""The evidence directory can be configured with EPI_REPORTS_DIR."""

from epi_monitor.ui.alert_panels import reports_root


def test_explicit_evidence_directory_takes_priority(tmp_path, monkeypatch):
    monkeypatch.setenv("EPI_REPORTS_DIR", str(tmp_path / "current"))
    assert reports_root() == (tmp_path / "current").resolve()


def test_default_evidence_directory_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv("EPI_REPORTS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    assert reports_root() == tmp_path / "reports"
