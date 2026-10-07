"""Simple local login screen for the dashboard.

Credentials come from EPI_LOGIN_USER / EPI_LOGIN_PASSWORD, then from the
[login] section of .streamlit/secrets.toml, then from the demo default below.
EPI_LOGIN_DISABLED=1 skips the screen (automated tests, kiosk use).
This gates the local demo; it is not a substitute for real access control.
"""
from __future__ import annotations

from base64 import b64encode
from functools import lru_cache
import hmac
import os
from pathlib import Path

import streamlit as st

DEFAULT_USER = "admin"
DEFAULT_PASSWORD = "epi2026"
LOGO_PATH = Path(__file__).with_name("assets") / "logo.svg"


@lru_cache(maxsize=1)
def logo_uri() -> str:
    try:
        return "data:image/svg+xml;base64," + b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    except OSError:
        return ""


def logo_img(size: int) -> str:
    uri = logo_uri()
    return f'<img src="{uri}" width="{size}" height="{size}" alt="Monitor de EPIs">' if uri else "🦺"


def _secret(key: str) -> str:
    try:
        return str(st.secrets["login"][key])
    except Exception:  # No secrets file or no [login] section.
        return ""


def credentials() -> tuple[str, str]:
    user = os.environ.get("EPI_LOGIN_USER") or _secret("user") or DEFAULT_USER
    password = os.environ.get("EPI_LOGIN_PASSWORD") or _secret("password") or DEFAULT_PASSWORD
    return user, password


def check(user: str, password: str) -> bool:
    expected_user, expected_password = credentials()
    # Constant-time comparison of both fields; evaluate both before combining.
    user_ok = hmac.compare_digest(user.strip().encode(), expected_user.encode())
    password_ok = hmac.compare_digest(password.encode(), expected_password.encode())
    return user_ok and password_ok


def logout() -> None:
    st.session_state.authenticated = False
    st.session_state.pop("login_user", None)


def require_login() -> bool:
    """Show the login card until the user signs in; True once authenticated."""
    if os.environ.get("EPI_LOGIN_DISABLED") == "1" or st.session_state.get("authenticated"):
        return True
    st.markdown("<style>[data-testid='stSidebar'], [data-testid='stSidebarCollapsedControl'] {display:none;}</style>",
                unsafe_allow_html=True)
    _, center, _ = st.columns([1, 1.1, 1])
    with center:
        st.markdown(f'<div class="login-head">{logo_img(72)}<h2>Monitor de EPIs</h2>'
                    '<p>Detecção de capacete, colete e bota com visão computacional</p></div>',
                    unsafe_allow_html=True)
        with st.form("login", border=True):
            user = st.text_input("Usuário")
            password = st.text_input("Senha", type="password")
            submitted = st.form_submit_button("Entrar", type="primary", width="stretch")
        if submitted:
            if check(user, password):
                st.session_state.authenticated = True
                st.session_state.login_user = user.strip()
                st.rerun()
            st.error("Usuário ou senha incorretos.")
        st.markdown('<p class="login-foot">Trabalho de Conclusão de Curso · YOLO11</p>', unsafe_allow_html=True)
    return False
