"""Design system: tokens, Lucide icons and small HTML components for Streamlit.

Every color, radius and shadow comes from the CSS custom properties below, so
cards, badges and tables stay consistent. Only static markup and numbers are
interpolated; user text passes through ``escape`` first.
"""
from __future__ import annotations

from html import escape

# Lucide icons (MIT), 24px grid, 1.75px stroke. Inner SVG markup only.
ICONS = {
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/>'
             '<path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "shield": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 '
              '4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/>'
             '<path d="M12 9v4"/><path d="M12 17h.01"/>',
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "video": '<path d="m16 13 5.223 3.482a.5.5 0 0 0 .777-.416V7.87a.5.5 0 0 0-.752-.432L16 10.5"/>'
             '<rect x="2" y="6" width="14" height="12" rx="2"/>',
    "helmet": '<path d="M10 10V5a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1v5"/><path d="M14 6a6 6 0 0 1 6 6v3"/>'
              '<path d="M4 15v-3a6 6 0 0 1 6-6"/><rect x="2" y="15" width="20" height="4" rx="1"/>',
    "vest": '<path d="M20.38 3.46 16 2a4 4 0 0 1-8 0L3.62 3.46a2 2 0 0 0-1.34 2.23l.58 3.47a1 1 0 0 0 .99.84H6v10'
            'c0 1.1.9 2 2 2h8a2 2 0 0 0 2-2V10h2.15a1 1 0 0 0 .99-.84l.58-3.47a2 2 0 0 0-1.34-2.23z"/>',
    "boots": '<path d="M4 16v-2.38C4 11.5 2.97 10.5 3 8c.03-2.72 1.49-6 4.5-6C9.37 2 10 3.8 10 5.5c0 3.11-2 '
             '5.66-2 8.68V16a2 2 0 1 1-4 0Z"/><path d="M20 20v-2.38c0-2.12 1.03-3.12 1-5.62-.03-2.72-1.49-6-4.5-6'
             'C14.63 6 14 7.8 14 9.5c0 3.11 2 5.66 2 8.68V20a2 2 0 1 0 4 0Z"/><path d="M16 17h4"/><path d="M4 13h4"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "x": '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    "help": '<circle cx="12" cy="12" r="10"/><path d="M12 8v4"/><path d="M12 16h.01"/>',
    "cpu": '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/>'
           '<path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/>',
    "upload": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/>'
              '<line x1="12" x2="12" y1="3" y2="15"/>',
    "play": '<polygon points="6 3 20 12 6 21 6 3"/>',
    "clipboard": '<rect width="8" height="4" x="8" y="2" rx="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6'
                 'a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><path d="m9 14 2 2 4-4"/>',
    "image": '<rect width="18" height="18" x="3" y="3" rx="2"/><circle cx="9" cy="9" r="2"/>'
             '<path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>',
    "chart": '<path d="M3 3v16a2 2 0 0 0 2 2h16"/><path d="m19 9-5 5-4-4-3 3"/>',
    "scan": '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/>'
            '<path d="M7 21H5a2 2 0 0 1-2-2v-2"/><circle cx="12" cy="12" r="3"/>',
    "folder": '<path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4'
              'a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/>',
}

EQUIPMENT = {"helmet": "Capacete", "vest": "Colete", "boots": "Bota"}


def icon(name: str, size: int = 18) -> str:
    return (f'<svg class="ic" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICONS[name]}</svg>')


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  /* Neutrals */
  --bg: #F8F9FA; --surface: #FFFFFF; --surface-2: #F3F4F6; --border: #E5E7EB; --border-strong: #D1D5DB;
  --text: #111827; --text-2: #4B5563; --text-3: #6B7280;
  /* Brand: deep teal (trust) + safety amber (accent, PPE world) */
  --primary: #0F766E; --primary-600: #0D655E; --primary-50: #F0FDFA; --primary-100: #CCFBF1;
  --accent: #F59E0B; --accent-50: #FFFBEB;
  /* States */
  --success: #10B981; --success-50: #ECFDF5; --success-700: #047857;
  --warning: #F59E0B; --warning-50: #FFFBEB; --warning-700: #B45309;
  --danger: #EF4444;  --danger-50: #FEF2F2;  --danger-700: #B91C1C;
  --info: #3B82F6;    --info-50: #EFF6FF;    --info-700: #1D4ED8;
  /* Shape & depth */
  --r-sm: 8px; --r-md: 12px; --r-lg: 16px;
  --shadow-sm: 0 1px 2px rgba(16,24,40,.05);
  --shadow-md: 0 1px 3px rgba(16,24,40,.06), 0 4px 12px rgba(16,24,40,.05);
  --shadow-lg: 0 12px 32px rgba(16,24,40,.10);
  --ease: cubic-bezier(.2,.8,.2,1);
}

html, body, .stApp, [class*="st-"], button, input, textarea, select {
  font-family: 'Inter', -apple-system, 'Segoe UI', system-ui, sans-serif !important;
}
/* Keep Streamlit's icon font: the rule above would turn icons into plain words. */
[data-testid="stIconMaterial"], [data-testid="stExpanderIcon"], .material-symbols-rounded {
  font-family: 'Material Symbols Rounded' !important;
}
.stApp { background: var(--bg); color: var(--text); }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding: 1.75rem 2.25rem 3rem; max-width: 1440px; }
.ic { display:inline-block; vertical-align:middle; flex-shrink:0; }

/* Type scale: Display 30 / H1 24 / H2 18 / Body 14 / Caption 12 */
h1, h2, h3, h4 { color: var(--text) !important; letter-spacing: -.02em; font-weight: 700 !important; }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color: var(--text-2); }
[data-testid="stCaptionContainer"], .stCaption { color: var(--text-3) !important; font-size: .78rem !important; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] .block-container, [data-testid="stSidebarContent"] { padding-top: .5rem; }
.brand { display:flex; align-items:center; gap:.7rem; padding:.25rem 0 1.1rem; border-bottom:1px solid var(--border); margin-bottom:1.1rem; }
.brand-logo { width:38px; height:38px; border-radius:10px; display:grid; place-items:center; color:#fff;
  background: linear-gradient(135deg, var(--primary) 0%, #14B8A6 100%); box-shadow: 0 4px 12px rgba(15,118,110,.30); }
.brand-name { font-weight:800; font-size:1.02rem; color:var(--text); letter-spacing:-.02em; line-height:1.1; }
.brand-sub { font-size:.72rem; color:var(--text-3); }
.side-label { font-size:.68rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--text-3);
  margin: 1rem 0 .35rem; display:flex; align-items:center; gap:.4rem; }
.side-meta { font-size:.78rem; color:var(--text-3); display:flex; align-items:center; gap:.45rem; margin:.25rem 0; }
.dot { width:8px; height:8px; border-radius:50%; display:inline-block; }

/* ---------- Header ---------- */
.crumbs { font-size:.8rem; color:var(--text-3); display:flex; align-items:center; gap:.4rem; margin-bottom:.35rem; }
.crumbs b { color: var(--text-2); font-weight:600; }
.page-head { display:flex; justify-content:space-between; align-items:flex-end; gap:1rem; flex-wrap:wrap; margin-bottom:1.25rem; }
.page-title { font-size:1.85rem; font-weight:800; letter-spacing:-.03em; color:var(--text); margin:0; line-height:1.15; }
.page-sub { color:var(--text-3); font-size:.92rem; margin:.3rem 0 0; }
.chips { display:flex; gap:.5rem; flex-wrap:wrap; }
.chip { display:inline-flex; align-items:center; gap:.4rem; padding:.42rem .8rem; border-radius:999px; font-size:.78rem; font-weight:600;
  background:var(--surface); color:var(--text-2); border:1px solid var(--border); box-shadow:var(--shadow-sm);
  transition: all .2s var(--ease); }
.chip.on { background:var(--primary-50); color:var(--primary); border-color:var(--primary-100); }
.chip.off { color:var(--text-3); background:var(--surface-2); text-decoration:line-through; }

/* ---------- Badges ---------- */
.badge { display:inline-flex; align-items:center; gap:.35rem; padding:.22rem .6rem; border-radius:999px; font-size:.72rem; font-weight:650; line-height:1.4; }
.badge.ok { background:var(--success-50); color:var(--success-700); }
.badge.unsafe { background:var(--danger-50); color:var(--danger-700); }
.badge.uncertain { background:var(--warning-50); color:var(--warning-700); }
.badge.info { background:var(--info-50); color:var(--info-700); }
.badge.neutral { background:var(--surface-2); color:var(--text-2); }
.badge.live::before { content:""; width:7px; height:7px; border-radius:50%; background:currentColor; animation: pulse 1.6s infinite; }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(16,185,129,.55)} 70%{box-shadow:0 0 0 7px rgba(16,185,129,0)} 100%{box-shadow:0 0 0 0 rgba(16,185,129,0)} }

/* ---------- KPI cards ---------- */
.kpi { background:var(--surface); border:1px solid var(--border); border-radius:var(--r-lg); padding:1.05rem 1.15rem;
  box-shadow:var(--shadow-sm); transition: transform .2s var(--ease), box-shadow .2s var(--ease); min-height:116px; }
.kpi:hover { transform: translateY(-2px); box-shadow: var(--shadow-md); }
.kpi-top { display:flex; justify-content:space-between; align-items:center; }
.kpi-label { font-size:.8rem; font-weight:600; color:var(--text-3); }
.kpi-icon { width:34px; height:34px; border-radius:10px; display:grid; place-items:center; }
.kpi-icon.teal { background:var(--primary-50); color:var(--primary); }
.kpi-icon.green { background:var(--success-50); color:var(--success-700); }
.kpi-icon.red { background:var(--danger-50); color:var(--danger-700); }
.kpi-icon.blue { background:var(--info-50); color:var(--info-700); }
.kpi-value { font-size:1.85rem; font-weight:750; color:var(--text); letter-spacing:-.03em; margin-top:.35rem; line-height:1.1; }
.kpi-value small { font-size:.9rem; color:var(--text-3); font-weight:600; margin-left:.2rem; }
.kpi-foot { font-size:.74rem; color:var(--text-3); margin-top:.3rem; display:flex; align-items:center; gap:.35rem; }
.delta { font-weight:700; padding:.05rem .4rem; border-radius:6px; }
.delta.up { color:var(--success-700); background:var(--success-50); }
.delta.down { color:var(--danger-700); background:var(--danger-50); }
.delta.flat { color:var(--text-3); background:var(--surface-2); }

/* ---------- Cards / containers ---------- */
[data-testid="stVerticalBlockBorderWrapper"] { background:var(--surface); border:1px solid var(--border) !important;
  border-radius:var(--r-lg) !important; box-shadow:var(--shadow-sm); transition: box-shadow .2s var(--ease); }
[data-testid="stVerticalBlockBorderWrapper"]:hover { box-shadow: var(--shadow-md); }
.card-head { display:flex; align-items:center; justify-content:space-between; margin-bottom:.75rem; }
.card-title { display:flex; align-items:center; gap:.55rem; font-weight:700; font-size:1.02rem; color:var(--text); }
.card-title .ic { color: var(--primary); }

/* ---------- Person cards ---------- */
.person { border:1px solid var(--border); border-radius:var(--r-md); padding:.8rem .95rem; margin-bottom:.65rem; background:var(--surface);
  transition: all .2s var(--ease); position:relative; overflow:hidden; animation: rise .35s var(--ease) both; }
.person:hover { border-color:var(--border-strong); box-shadow:var(--shadow-md); }
.person::before { content:""; position:absolute; left:0; top:0; bottom:0; width:4px; }
.person.ok::before { background:var(--success); } .person.unsafe::before { background:var(--danger); } .person.uncertain::before { background:var(--warning); }
.person-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:.55rem; }
.avatar { width:30px; height:30px; border-radius:50%; display:inline-grid; place-items:center; font-size:.78rem; font-weight:700;
  background:var(--surface-2); color:var(--text-2); margin-right:.55rem; }
.person-name { font-weight:650; color:var(--text); font-size:.92rem; display:flex; align-items:center; }
.eq { display:grid; grid-template-columns: repeat(3, 1fr); gap:.45rem; }
.eq-item { border-radius:var(--r-sm); padding:.45rem .5rem; display:flex; flex-direction:column; align-items:center; gap:.2rem;
  font-size:.72rem; font-weight:600; border:1px solid transparent; }
.eq-item.ok { background:var(--success-50); color:var(--success-700); }
.eq-item.unsafe { background:var(--danger-50); color:var(--danger-700); border-color:#FECACA; }
.eq-item.uncertain { background:var(--warning-50); color:var(--warning-700); }
.eq-item span { color: var(--text-2); font-weight:600; }
@keyframes rise { from { opacity:0; transform: translateY(6px);} to { opacity:1; transform:none; } }

/* ---------- Camera viewport ---------- */
div[data-epi-preview] { border-radius: var(--r-md) !important; max-width: 100% !important;
  background: #0F172A !important; color:#94A3B8 !important; box-shadow: inset 0 0 0 1px rgba(255,255,255,.04); }

/* ---------- Empty state & steps ---------- */
.empty { text-align:center; padding:2.2rem 1.5rem; }
.empty-icon { width:56px; height:56px; border-radius:16px; margin:0 auto .9rem; display:grid; place-items:center;
  background:var(--primary-50); color:var(--primary); }
.empty h4 { margin:.2rem 0 .3rem; font-size:1.05rem; }
.empty p { color:var(--text-3); font-size:.88rem; max-width:420px; margin:0 auto; }
.steps { display:flex; gap:.75rem; margin-top:1.3rem; justify-content:center; flex-wrap:wrap; }
.step { display:flex; align-items:center; gap:.55rem; background:var(--surface-2); border-radius:999px; padding:.45rem .9rem .45rem .45rem;
  font-size:.8rem; color:var(--text-2); font-weight:600; }
.step b { width:24px; height:24px; border-radius:50%; display:grid; place-items:center; background:var(--primary); color:#fff; font-size:.72rem; }

/* ---------- Skeleton ---------- */
.skel { background: linear-gradient(90deg, #EEF0F3 25%, #F7F8FA 37%, #EEF0F3 63%); background-size: 400% 100%;
  animation: shimmer 1.3s ease infinite; border-radius: var(--r-sm); }
@keyframes shimmer { 0% { background-position: 100% 50%; } 100% { background-position: 0 50%; } }

/* ---------- Timeline (pipeline) ---------- */
.timeline { position:relative; padding-left:1.6rem; }
.timeline::before { content:""; position:absolute; left:.55rem; top:.3rem; bottom:.3rem; width:2px; background:var(--border); }
.tl { position:relative; margin-bottom:1rem; }
.tl::before { content:""; position:absolute; left:-1.32rem; top:.25rem; width:12px; height:12px; border-radius:50%;
  background:var(--surface); border:2px solid var(--primary); }
.tl b { color:var(--text); font-size:.9rem; } .tl p { margin:.1rem 0 0; font-size:.83rem; color:var(--text-3); }

/* ---------- Widgets ---------- */
.stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button {
  border-radius: 10px !important; font-weight: 600 !important; border:1px solid var(--border) !important;
  transition: all .18s var(--ease) !important; box-shadow: var(--shadow-sm); }
.stButton button:hover, .stDownloadButton button:hover { transform: translateY(-1px); box-shadow: var(--shadow-md); border-color: var(--border-strong) !important; }
.stButton button:active { transform: translateY(0); }
.stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"] {
  background: var(--primary) !important; color:#fff !important; border-color: var(--primary) !important; }
.stButton button[kind="primary"] p, [data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"] p { color:#fff !important; }
.stButton button[kind="primary"]:hover { background: var(--primary-600) !important; box-shadow: 0 6px 16px rgba(15,118,110,.28); }
.stButton button:disabled { opacity:.45; transform:none; box-shadow:none; }
.stButton button:focus-visible, input:focus-visible { outline: 3px solid var(--primary-100) !important; outline-offset: 1px; }
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] { border-radius: 10px !important; }
[data-testid="stFileUploaderDropzone"] { border-radius: var(--r-md); border: 1.5px dashed var(--border-strong); background: var(--surface-2); }
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--primary); background: var(--primary-50); }
[data-testid="stExpander"] details { border-radius: var(--r-md) !important; border-color: var(--border) !important; background: var(--surface); }

/* Tabs as a segmented navigation */
.stTabs [data-baseweb="tab-list"] { gap:.25rem; background:var(--surface-2); padding:.3rem; border-radius:12px; width:fit-content; border:1px solid var(--border); }
.stTabs [data-baseweb="tab"] { height:auto; padding:.45rem 1rem; border-radius:9px; font-weight:600; color:var(--text-3); transition: all .18s var(--ease); }
.stTabs [data-baseweb="tab"]:hover { color: var(--text); }
.stTabs [aria-selected="true"] { background:var(--surface) !important; color:var(--primary) !important; box-shadow:var(--shadow-sm); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }

/* Tables & alerts */
[data-testid="stDataFrame"] { border-radius: var(--r-md); overflow:hidden; border:1px solid var(--border); }
[data-testid="stAlert"] { border-radius: var(--r-md); border:1px solid var(--border); }
[data-testid="stToast"] { border-radius: var(--r-md); box-shadow: var(--shadow-lg); }
div[role="dialog"] { border-radius: var(--r-lg) !important; }

@media (max-width: 1024px) {
  .block-container { padding: 1.25rem 1rem 2rem; }
  .page-title { font-size: 1.5rem; }
}
</style>
"""


def badge(text: str, kind: str = "neutral", live: bool = False) -> str:
    return f'<span class="badge {kind}{" live" if live else ""}">{escape(text)}</span>'


def card_title(title: str, icon_name: str, right: str = "") -> str:
    return (f'<div class="card-head"><div class="card-title">{icon(icon_name)}{escape(title)}</div>'
            f'<div>{right}</div></div>')


def kpi(label: str, value: str, icon_name: str, tone: str, foot: str = "", delta: float | None = None,
        delta_good_up: bool = True) -> str:
    """KPI card; ``delta`` is the change versus the previous analyzed frame."""
    delta_html = ""
    if delta is not None:
        good = (delta > 0) == delta_good_up
        flat = abs(delta) < .05
        kind = "flat" if flat else "up" if good else "down"
        arrow = "→" if flat else "↑" if delta > 0 else "↓"
        amount = f"{abs(delta):.1f}".rstrip("0").rstrip(".")
        delta_html = f'<span class="delta {kind}">{arrow} {amount}</span>'
    return (f'<div class="kpi"><div class="kpi-top"><span class="kpi-label">{escape(label)}</span>'
            f'<span class="kpi-icon {tone}">{icon(icon_name)}</span></div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-foot">{delta_html}{escape(foot)}</div></div>')


def empty_state(title: str, text: str, icon_name: str = "scan", steps: tuple[str, ...] = ()) -> str:
    steps_html = "".join(f'<span class="step"><b>{n}</b>{escape(step)}</span>' for n, step in enumerate(steps, 1))
    return (f'<div class="empty"><div class="empty-icon">{icon(icon_name, 26)}</div><h4>{escape(title)}</h4>'
            f'<p>{escape(text)}</p>{f"<div class=steps>{steps_html}</div>" if steps else ""}</div>')


def skeleton_panel() -> str:
    """Loading placeholder shaped like the real layout (KPIs + viewport + list)."""
    kpis = "".join('<div class="skel" style="height:116px;border-radius:16px"></div>' for _ in range(4))
    rows = "".join('<div class="skel" style="height:84px;margin-bottom:.65rem;border-radius:12px"></div>' for _ in range(3))
    return (f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:1rem">{kpis}</div>'
            f'<div style="display:grid;grid-template-columns:2.2fr 1fr;gap:1.25rem">'
            f'<div class="skel" style="aspect-ratio:16/9;border-radius:16px"></div><div>{rows}</div></div>')
