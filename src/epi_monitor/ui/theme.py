"""Minimal visual style: one font, one accent color, flat surfaces and status pills."""
from __future__ import annotations

from html import escape

EQUIPMENT = {"helmet": "Capacete", "vest": "Colete", "boots": "Bota"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root { --text:#111827; --muted:#6B7280; --border:#E5E7EB; --surface:#FFFFFF; --bg:#F9FAFB; --accent:#0F766E; }
html, body, .stApp, [class*="st-"], button, input, textarea, select { font-family:'Inter', system-ui, sans-serif !important; }
[data-testid="stIconMaterial"], [data-testid="stExpanderIcon"], .material-symbols-rounded { font-family:'Material Symbols Rounded' !important; }
.stApp { background: var(--bg); }
/* Live video reruns every frame; keep the page from fading while it updates. */
[data-stale="true"] { opacity: 1 !important; transition: none !important; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding: 2rem 2.5rem 3rem; max-width: 1280px; }
[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
h1 { font-size: 1.6rem !important; font-weight: 700 !important; letter-spacing: -.02em; color: var(--text) !important; }
[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: .8rem 1rem; }
[data-testid="stMetricLabel"] p { color: var(--muted) !important; font-size: .8rem !important; }
[data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 600; color: var(--text); }
[data-testid="stVerticalBlockBorderWrapper"] { background: var(--surface); border-radius: 10px !important; border-color: var(--border) !important; }
.stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button { border-radius: 8px !important; font-weight: 600 !important; }
.stButton button[kind="primary"] p, [data-testid="stFormSubmitButton"] button p { color: #fff !important; }
.stButton button[kind="primary"]:disabled { background: #E5E7EB !important; border-color: #E5E7EB !important; }
.stButton button[kind="primary"]:disabled p { color: #6B7280 !important; }
div[data-epi-preview] { border-radius: 8px !important; max-width: 100% !important; background: #111827 !important; }
.stTabs [data-baseweb="tab"] { font-weight: 600; }
.person { display:flex; justify-content:space-between; align-items:center; gap:.5rem; flex-wrap:wrap;
  padding:.65rem .2rem; border-bottom:1px solid var(--border); font-size:.9rem; color:var(--text); }
.person:last-child { border-bottom: 0; }
.items { display:flex; gap:.9rem; flex-wrap:wrap; color:#374151; }
.pill { padding:.15rem .55rem; border-radius:999px; font-size:.75rem; font-weight:600; white-space:nowrap; }
.pill.ok { background:#ECFDF5; color:#047857; } .pill.unsafe { background:#FEF2F2; color:#B91C1C; }
.pill.uncertain { background:#FFFBEB; color:#B45309; } .pill.neutral { background:#F3F4F6; color:#374151; }
.muted { color: var(--muted); font-size: .85rem; }
.brand { display:flex; align-items:center; gap:.65rem; padding:.2rem 0 1rem; border-bottom:1px solid var(--border); margin-bottom:.8rem; }
.brand b { display:block; color:var(--text); font-size:1rem; }
.brand span { display:block; color:var(--muted); font-size:.75rem; }
.page-head { display:flex; align-items:center; gap:.85rem; margin-bottom:.6rem; }
.page-head h1 { margin:0 !important; padding:0 !important; }
.page-head p { margin:.1rem 0 0; }
.login-head { text-align:center; margin:9vh 0 1.2rem; }
.login-head h2 { margin:.6rem 0 .2rem !important; font-size:1.5rem !important; color:var(--text) !important; }
.login-head p { color:var(--muted); font-size:.9rem; margin:0; }
[data-testid="stForm"] { background:var(--surface); border-radius:12px !important; box-shadow:0 8px 24px rgba(16,24,40,.08); padding:1.4rem 1.4rem .6rem !important; }
.login-foot { text-align:center; color:var(--muted); font-size:.78rem; margin-top:1rem; }
.panel { display:flex; flex-direction:column; gap:1rem; margin-bottom:.5rem; }
.kpis { display:grid; grid-template-columns:repeat(4, 1fr); gap:.75rem; }
.kpi { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:.75rem 1rem; }
.kpi span { display:block; color:var(--muted); font-size:.8rem; }
.kpi b { display:block; font-size:1.6rem; font-weight:600; color:var(--text); line-height:1.3; }
.main { display:grid; grid-template-columns:2fr 1fr; gap:1.5rem; align-items:start; }
.video p { margin:.4rem 0 0; }
.note { background:#EFF6FF; color:#1E3A8A; border-radius:8px; padding:.6rem .8rem; font-size:.9rem; }
.people { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:.8rem 1rem; }
.people-title { font-weight:600; color:var(--text); margin:0 0 .3rem; }
@media (max-width: 900px) { .kpis { grid-template-columns:repeat(2, 1fr); } .main { grid-template-columns:1fr; } }
</style>
"""


def pill(text: str, kind: str = "neutral") -> str:
    return f'<span class="pill {kind}">{escape(text)}</span>'
