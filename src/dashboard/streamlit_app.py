# ============================================================
#  src/dashboard/streamlit_app.py
#  Dashboard de Monitoramento em Tempo Real
#  TCC: Monitoramento de EPI com Visão Computacional
#
#  Como executar (da raiz do projeto):
#    streamlit run src/dashboard/streamlit_app.py
# ============================================================

import json
import sys
from datetime import datetime, date
from pathlib import Path
from collections import Counter

import pandas as pd
import streamlit as st
from PIL import Image

# ── Garante importação do config ─────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import VIOLATIONS_DIR, LOGS_DIR, MAX_VIOLATIONS_UI, MODEL_PATH, CONFIDENCE

# ── Configuração da página ────────────────────────────────────
st.set_page_config(
    page_title="Sistema EPI — TCC",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS personalizado ─────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e2a3a 0%, #243447 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2d4a6b;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4fc3f7;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #90a4ae;
        margin-top: 6px;
    }
    .violation-tag {
        background-color: #b71c1c;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        display: inline-block;
        margin: 2px;
    }
    .ok-tag {
        background-color: #1b5e20;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
    }
    .status-online {
        color: #66bb6a;
        font-weight: bold;
    }
    .demo-badge {
        background: #e65100;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Funções auxiliares ────────────────────────────────────────

@st.cache_data(ttl=3)
def load_violations() -> list:
    """Carrega violações do JSON (cache de 3s para desempenho)."""
    json_path = LOGS_DIR / "violations.json"
    if not json_path.exists():
        return []
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return []


def get_today_count(records: list) -> int:
    today = date.today().isoformat()
    return sum(1 for r in records if r.get("timestamp", "").startswith(today))


def format_ts(iso_str: str) -> str:
    try:
        return datetime.fromisoformat(iso_str).strftime("%d/%m/%Y %H:%M:%S")
    except Exception:
        return iso_str


def count_png_files() -> int:
    return len(list(VIOLATIONS_DIR.glob("*.png")))


def get_violations_by_day(records: list) -> pd.DataFrame:
    """Agrupa violações por dia para o gráfico."""
    if not records:
        return pd.DataFrame()
    counts: Counter = Counter()
    for r in records:
        ts = r.get("timestamp", "")
        if ts:
            day = ts[:10]  # "YYYY-MM-DD"
            counts[day] += 1
    if not counts:
        return pd.DataFrame()
    df = pd.DataFrame(
        sorted(counts.items()),
        columns=["Data", "Violações"]
    )
    df["Data"] = pd.to_datetime(df["Data"])
    return df


def get_violation_types(records: list) -> pd.DataFrame:
    all_viols: list = []
    for r in records:
        all_viols.extend(r.get("violations", []))
    if not all_viols:
        return pd.DataFrame()
    counts = Counter(all_viols)
    return pd.DataFrame(counts.items(), columns=["Tipo", "Ocorrências"])


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🦺 Sistema EPI")
    st.caption("TCC — Visão Computacional e IA")
    st.divider()

    st.markdown("### ⚙️ Configurações")
    auto_refresh = st.toggle("🔄 Auto-atualizar (5s)", value=True)
    show_debug   = st.toggle("🔧 Detalhes técnicos", value=False)

    st.divider()
    st.markdown("### 🤖 Modelo")
    st.code(f"""
Modelo   : {Path(MODEL_PATH).name}
Confiança: {CONFIDENCE:.0%}
    """)

    st.divider()
    st.markdown("### ℹ️ Sobre")
    st.markdown(
        "Monitoramento automático de EPIs com "
        "**YOLOv8** + **OpenCV** em tempo real."
    )
    st.markdown("---")
    now = datetime.now().strftime("%H:%M:%S")
    st.caption(f"Última atualização: {now}")


# ── Header ────────────────────────────────────────────────────
st.markdown("# 📊 Monitoramento de EPI em Tempo Real")
st.markdown("Sistema inteligente de detecção de Equipamentos de Proteção Individual")
st.divider()

# ── Carrega dados ─────────────────────────────────────────────
records         = load_violations()
today_count     = get_today_count(records)
total_count     = len(records)
png_count       = count_png_files()
recent          = sorted(records, key=lambda x: x.get("timestamp",""), reverse=True)

# Violação mais comum
all_viols_flat = []
for r in records:
    all_viols_flat.extend(r.get("violations", []))
most_common = Counter(all_viols_flat).most_common(1)
most_common_str = most_common[0][0] if most_common else "—"

# ── Métricas ──────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

with c1:
    delta = f"+{today_count}" if today_count > 0 else "0"
    color = "inverse" if today_count > 0 else "normal"
    st.metric("⚠️ Alertas Hoje", today_count, delta=delta, delta_color=color)

with c2:
    st.metric("📋 Total de Registros", total_count)

with c3:
    st.metric("🔴 Infração Mais Comum", most_common_str)

with c4:
    st.metric("📁 Imagens Salvas", png_count)

st.divider()

# ── Gráficos ──────────────────────────────────────────────────
if records:
    col_g1, col_g2 = st.columns([2, 1])

    with col_g1:
        st.markdown("### 📈 Violações por Dia")
        df_day = get_violations_by_day(records)
        if not df_day.empty:
            st.bar_chart(df_day.set_index("Data")["Violações"])
        else:
            st.info("Sem dados suficientes para o gráfico.")

    with col_g2:
        st.markdown("### 🥧 Tipos de Violação")
        df_types = get_violation_types(records)
        if not df_types.empty:
            st.dataframe(df_types, use_container_width=True, hide_index=True)
        else:
            st.info("Sem dados.")

    st.divider()

# ── Últimas Infrações ─────────────────────────────────────────
st.markdown("### 📸 Últimas Infrações Registradas")

if not recent:
    st.info(
        "🟢 **Nenhuma infração registrada ainda.**\n\n"
        "O sistema está monitorando... Execute `python main.py` para iniciar a câmera."
    )
else:
    cols = st.columns(3)
    for i, record in enumerate(recent[:9]):
        col = cols[i % 3]
        with col:
            with st.container(border=True):
                img_file = VIOLATIONS_DIR / record.get("image_file", "")
                if img_file.exists():
                    img = Image.open(img_file)
                    st.image(img, use_container_width=True)
                else:
                    st.markdown("🖼️ *Imagem não encontrada*")

                ts    = format_ts(record.get("timestamp", ""))
                viols = record.get("violations", [])
                n     = record.get("violators", 1)

                st.caption(f"🕐 {ts}")
                col_a, col_b = st.columns(2)
                col_a.caption(f"👤 {n} pessoa(s)")
                col_b.caption(f"⛔ {', '.join(viols)}")

st.divider()

# ── Tabela completa ───────────────────────────────────────────
st.markdown("### 📋 Histórico Completo de Violações")

if records:
    df = pd.DataFrame([{
        "Data/Hora":   format_ts(r.get("timestamp", "")),
        "Pessoas":     r.get("violators", 1),
        "Violações":   ", ".join(r.get("violations", [])),
        "Arquivo":     r.get("image_file", "—"),
    } for r in sorted(records, key=lambda x: x.get("timestamp",""), reverse=True)])

    st.dataframe(df, use_container_width=True, hide_index=True, height=300)

    csv = df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="⬇️ Exportar dados para CSV (TCC)",
        data=csv.encode("utf-8-sig"),
        file_name=f"violacoes_epi_{date.today().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        help="Exporta todos os registros de violações para análise no TCC"
    )
else:
    st.info("Nenhum dado disponível ainda.")

st.divider()

# ── Tabela de métricas do TCC ─────────────────────────────────
st.markdown("### 📊 Métricas de Desempenho — Dados para o TCC")

tab1, tab2 = st.tabs(["⚡ Benchmark de Modelos", "🧪 Cenários de Teste"])

with tab1:
    st.markdown("**Comparação YOLOv8: Nano vs Small vs Medium**")
    st.caption("Preencha após executar `B` no main.py para cada modelo")
    bench_df = pd.DataFrame({
        "Modelo":        ["yolov8n", "yolov8s", "yolov8m"],
        "Parâmetros":    ["3.2M",    "11.2M",   "25.9M"],
        "Tempo médio":   ["— ms",    "— ms",    "— ms"],
        "FPS equiv.":    ["—",       "—",       "—"],
        "mAP@0.5":       ["—",       "—",       "—"],
        "RAM aprox.":    ["~300MB",  "~450MB",  "~700MB"],
    })
    st.dataframe(bench_df, use_container_width=True, hide_index=True)

with tab2:
    st.markdown("**Resultados por Cenário Experimental**")
    test_df = pd.DataFrame({
        "Cenário": [
            "1 — 1 pessoa com capacete",
            "2 — 1 pessoa sem capacete",
            "3 — 2 pessoas (1 com, 1 sem)",
            "4 — Iluminação baixa",
            "5 — Oclusão parcial",
            "6 — Distância longa",
        ],
        "Esperado":  ["Sem alerta ✅", "Alerta ⛔", "Alerta seletivo ✅", "—", "—", "—"],
        "Obtido":    ["—", "—", "—", "—", "—", "—"],
        "Precisão":  ["—", "—", "—", "—", "—", "—"],
        "Observação": ["", "", "", "", "", ""],
    })
    st.dataframe(test_df, use_container_width=True, hide_index=True)
    st.caption("Preencha durante os testes experimentais para o capítulo de resultados.")

# ── Debug técnico ─────────────────────────────────────────────
if show_debug:
    st.divider()
    st.markdown("### 🔧 Informações Técnicas")
    c_d1, c_d2 = st.columns(2)
    with c_d1:
        st.code(f"""
Dir. violações : {VIOLATIONS_DIR}
Dir. logs      : {LOGS_DIR}
JSON log       : {LOGS_DIR / 'violations.json'}
PNGs salvos    : {png_count}
Modelo         : {MODEL_PATH}
Confiança      : {CONFIDENCE}
        """, language="text")
    with c_d2:
        if recent:
            st.markdown("**Último registro JSON:**")
            st.json(recent[0])

# ── Auto-refresh ──────────────────────────────────────────────
if auto_refresh:
    import time
    time.sleep(5)
    st.rerun()
