"""Login screen: blocks the dashboard until valid credentials are entered."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from epi_monitor.ui import auth

APP_PATH = Path(__file__).resolve().parents[1] / "src" / "epi_monitor" / "ui" / "app.py"


@pytest.fixture
def login_app(monkeypatch, tmp_path):
    monkeypatch.delenv("EPI_LOGIN_DISABLED", raising=False)
    monkeypatch.delenv("EPI_LOGIN_USER", raising=False)
    monkeypatch.delenv("EPI_LOGIN_PASSWORD", raising=False)
    monkeypatch.setenv("EPI_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setattr(auth, "_secret", lambda key: "")
    return AppTest.from_file(str(APP_PATH), default_timeout=15).run()


def sign_in(app, user, password):
    next(field for field in app.text_input if field.label == "Usuário").set_value(user)
    next(field for field in app.text_input if field.label == "Senha").set_value(password)
    next(button for button in app.button if button.label == "Entrar").click().run()


def test_login_screen_hides_dashboard(login_app):
    assert not login_app.exception
    assert not login_app.tabs
    assert any(button.label == "Entrar" for button in login_app.button)
    assert "logo.svg" not in auth.logo_uri() and auth.logo_uri().startswith("data:image/svg+xml;base64,")


def test_login_wrong_password_is_rejected(login_app):
    sign_in(login_app, auth.DEFAULT_USER, "errada")
    assert not login_app.tabs
    assert any("incorretos" in error.value for error in login_app.error)


def test_login_valid_credentials_open_dashboard(login_app):
    sign_in(login_app, auth.DEFAULT_USER, auth.DEFAULT_PASSWORD)
    assert not login_app.exception
    assert [tab.label for tab in login_app.tabs] == ["Monitor", "Ocorrências", "Configurações"]
    assert any(button.label == "Sair" for button in login_app.button)


def test_login_logout_returns_to_login_screen(monkeypatch, tmp_path):
    monkeypatch.delenv("EPI_LOGIN_DISABLED", raising=False)
    monkeypatch.setenv("EPI_REPORTS_DIR", str(tmp_path / "reports"))
    app = AppTest.from_file(str(APP_PATH), default_timeout=15)
    app.session_state["authenticated"] = True
    app.run()
    assert app.tabs
    next(button for button in app.button if button.label == "Sair").click().run()
    assert not app.exception
    assert not app.tabs
    assert any(button.label == "Entrar" for button in app.button)


def test_login_credentials_come_from_environment(monkeypatch):
    monkeypatch.setattr(auth, "_secret", lambda key: "")
    monkeypatch.setenv("EPI_LOGIN_USER", "fiscal")
    monkeypatch.setenv("EPI_LOGIN_PASSWORD", "s3nha")
    assert auth.check("fiscal", "s3nha")
    assert not auth.check(auth.DEFAULT_USER, auth.DEFAULT_PASSWORD)
