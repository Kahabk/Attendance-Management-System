from __future__ import annotations

import streamlit as st


def inject_global_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0b1220;
            --panel: rgba(17, 24, 39, 0.88);
            --panel-border: rgba(148, 163, 184, 0.18);
            --text: #e5e7eb;
            --muted: #94a3b8;
            --accent: #22c55e;
            --accent-2: #38bdf8;
            --danger: #ef4444;
            --warning: #f59e0b;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(56, 189, 248, 0.18), transparent 30%),
                radial-gradient(circle at top right, rgba(34, 197, 94, 0.12), transparent 28%),
                linear-gradient(180deg, #07111f 0%, #0b1220 50%, #0f172a 100%);
            color: var(--text);
        }
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
        }
        .glass-card {
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: 20px;
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.18);
            padding: 1rem 1.1rem;
            backdrop-filter: blur(18px);
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin-bottom: 0.1rem;
        }
        .hero-subtitle {
            color: var(--muted);
            margin-bottom: 0.75rem;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 0.75rem;
        }
        .metric-card {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.95), rgba(15, 23, 42, 0.8));
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 18px;
            padding: 1rem;
        }
        .metric-label {
            font-size: 0.85rem;
            color: #cbd5e1;
        }
        .metric-value {
            font-size: 1.7rem;
            font-weight: 800;
            margin-top: 0.2rem;
        }
        .metric-note {
            color: var(--muted);
            font-size: 0.8rem;
        }
        .section-header {
            font-size: 1.15rem;
            font-weight: 700;
            margin: 0.4rem 0 0.6rem;
        }
        .stButton > button {
            border-radius: 12px;
            font-weight: 700;
        }
        .stTextInput input, .stNumberInput input, .stDateInput input, .stSelectbox div[data-baseweb="select"] {
            border-radius: 12px !important;
        }
        .status-pill {
            padding: 0.25rem 0.6rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str = "") -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """


def section_title(text: str) -> None:
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)

