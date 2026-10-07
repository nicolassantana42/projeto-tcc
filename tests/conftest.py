"""Shared test setup: the dashboard tests exercise the app behind the login screen."""
import pytest


@pytest.fixture(autouse=True)
def _skip_login(monkeypatch, request):
    if "login" not in request.node.name:
        monkeypatch.setenv("EPI_LOGIN_DISABLED", "1")
