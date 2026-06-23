from __future__ import annotations

import html
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from air_quality_llm import ask_air_quality, summarize_air_quality
from pipeline import collapse_consecutive_values, run_manual_pipeline, scrape_frames_equal


ROOT = Path(__file__).resolve().parents[1]
SCRAPED_DIR = ROOT / "data" / "scraped"
PREDICTIONS_DIR = ROOT / "predictions"
STATIONS_PATH = ROOT / "data" / "estaciones_valencia.csv"
MODEL_REGISTRY_PATH = ROOT / "models" / "builded" / "registry.json"

POLLUTANTS = ["NO2", "O3", "SO2", "PM-10", "PM-2.5"]
NAME_MAP = {
    "AVDA.FRANCIA": "València - Av. França",
    "BULEVARD SUD": "València - Bulevard Sud",
    "MOLÍ DEL SOL": "València - Molí del Sol",
    "PISTA DE SILLA": "València - Pista de Silla",
    "POLITÈCNIC": "València - Politècnic",
    "VIVERS": "València - Vivers",
    "VALÈNCIA CENTRE": "València - Centre",
    "OLIVERETA": "València Olivereta",
}
EXTRA_COORDS = {
    "DR.LLUCH": (39.4664, -0.3283),
    "CABANYAL": (39.4700, -0.3320),
    "PATRAIX": (39.4623, -0.3958),
}
HISTORICAL_SCRAPER_STATIONS = set(NAME_MAP)
QUALITY_LEVELS = [
    "Buena",
    "Razonablemente buena",
    "Regular",
    "Desfavorable",
    "Muy desfavorable",
    "Extremadamente desfavorable",
]
QUALITY_COLORS = {
    "Buena": "#50f0e6",
    "Razonablemente buena": "#50ccaa",
    "Regular": "#f0e641",
    "Desfavorable": "#ff5050",
    "Muy desfavorable": "#960032",
    "Extremadamente desfavorable": "#7d2181",
    "No hay datos": "#64748b",
}
QUALITY_THRESHOLDS = {
    "SO2": [100, 200, 350, 500, 750, 1250],
    "NO2": [40, 90, 120, 230, 340, 1000],
    "O3": [50, 100, 130, 240, 380, 800],
    "PM-10": [20, 40, 50, 100, 150, 1200],
    "PM-2.5": [10, 20, 25, 50, 75, 800],
}
QUALITY_RANK = {level: index for index, level in enumerate(QUALITY_LEVELS)}
POLLUTANT_NAMES = {
    "SO2": "Dioxido de Azufre",
    "NO2": "Dioxido de Nitrogeno",
    "O3": "Ozono",
    "PM-10": "Particulas < 10 micras",
    "PM-2.5": "Particulas < 2.5 micras",
}
AI_SUMMARY_VERSION = 2


st.set_page_config(page_title="Valencia Respira", layout="wide", initial_sidebar_state="collapsed")


@st.cache_data
def station_coords() -> dict[str, tuple[float, float]]:
    stations = pd.read_csv(STATIONS_PATH)
    coords = {
        row["Estación"]: (float(row["Latitud"]), float(row["Longitud"]))
        for _, row in stations.iterrows()
    }
    coords.update(EXTRA_COORDS)
    return coords


@st.cache_data
def load_snapshot(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.empty:
        fallback = latest_valid_history_file(path.parent / "history")
        if fallback is not None:
            df = pd.read_csv(fallback, encoding="utf-8-sig")
    df["station_display"] = df["µg/m3"]
    df["station"] = df["µg/m3"].map(NAME_MAP).fillna(df["µg/m3"])
    for col in POLLUTANTS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data
def model_training_cutoff() -> str:
    registry = json.loads(MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
    source_ends = []
    for entry in registry:
        model = json.loads((ROOT / entry["model_path"]).read_text(encoding="utf-8"))
        source_ends.append(pd.to_datetime(model["source_end"]))
    if not source_ends:
        return "sin fecha"
    return min(source_ends).strftime("%d/%m/%Y")


def latest_valid_history_file(history_dir: Path) -> Path | None:
    for path in sorted(history_dir.glob("*.csv"), reverse=True):
        try:
            frame = pd.read_csv(path, encoding="utf-8-sig")
        except (OSError, pd.errors.EmptyDataError):
            continue
        if not frame.empty:
            return path
    return None


@st.cache_data
def load_scraped_history() -> pd.DataFrame:
    rows = []
    previous_snapshot: pd.DataFrame | None = None
    history_dir = SCRAPED_DIR / "history"
    for path in sorted(history_dir.glob("*.csv")):
        timestamp = pd.to_datetime(path.stem, format="%Y-%m-%d_%H-%M", errors="coerce")
        if pd.isna(timestamp):
            continue
        try:
            frame = pd.read_csv(path, encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        if frame.empty:
            continue
        if previous_snapshot is not None and scrape_frames_equal(frame, previous_snapshot):
            continue
        previous_snapshot = frame
        frame["timestamp"] = timestamp
        rows.append(frame)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "station_display", "station", *POLLUTANTS])

    history = pd.concat(rows, ignore_index=True)
    history["station_display"] = history["µg/m3"]
    history["station"] = history["µg/m3"].map(NAME_MAP).fillna(history["µg/m3"])
    for col in POLLUTANTS:
        history[col] = pd.to_numeric(history[col], errors="coerce")
    return history.dropna(subset=["timestamp"]).sort_values("timestamp")


def quality_for_value(pollutant: str, value: float | None) -> str:
    if pd.isna(value):
        return "No hay datos"
    for level, upper in zip(QUALITY_LEVELS, QUALITY_THRESHOLDS[pollutant]):
        if float(value) <= upper:
            return level
    return "Extremadamente desfavorable"


def format_timestamp(value: pd.Timestamp) -> str:
    return pd.to_datetime(value).strftime("%d/%m/%Y %H:%M")


def threshold_range(pollutant: str, index: int) -> str:
    lower = 0 if index == 0 else QUALITY_THRESHOLDS[pollutant][index - 1] + 1
    upper = QUALITY_THRESHOLDS[pollutant][index]
    return f"{lower}-{upper}"


def color_for_value(pollutant: str, value: float | None) -> str:
    return QUALITY_COLORS[quality_for_value(pollutant, value)]


def radius_for_value(pollutant: str, value: float | None) -> int:
    if pd.isna(value):
        return 24
    max_reasonable = QUALITY_THRESHOLDS[pollutant][3]
    return int(24 + min(float(value) / max_reasonable, 1.0) * 22)


def map_points(df: pd.DataFrame, pollutant: str, mode: str) -> list[dict]:
    coords = station_coords()
    points = []
    for _, row in df.iterrows():
        if row["station"] not in coords:
            continue
        value = row[pollutant]
        missing_prediction = mode == "Prediccion" and row["station_display"] not in HISTORICAL_SCRAPER_STATIONS
        if missing_prediction or pd.isna(value):
            continue
        lat, lon = coords[row["station"]]
        quality = quality_for_value(pollutant, value)
        points.append(
            {
                "label": row["station_display"],
                "station": row["station"],
                "lat": lat,
                "lon": lon,
                "value_label": f"{float(value):.1f} ug/m3 · {quality}",
                "color": color_for_value(pollutant, value),
                "radius": radius_for_value(pollutant, value),
            }
        )
    return points


def style_page() -> None:
    st.markdown(
        """
        <style>
          .stApp {
            background:
              radial-gradient(circle at 8% 10%, rgba(20, 184, 166, .15), transparent 25%),
              radial-gradient(circle at 92% 8%, rgba(59, 130, 246, .14), transparent 26%),
              linear-gradient(135deg, #050816 0%, #07111f 46%, #041114 100%);
            color: #e5f8ff;
          }
          html, body, [data-testid="stAppViewContainer"], .stApp {
            width: 100%;
            min-width: 0;
            overflow-x: clip;
          }
          .block-container {
            width: 100%;
            max-width: 100vw;
            min-width: 0;
            padding: clamp(.65rem, 1.4vw, 1.1rem) clamp(.65rem, 1.7vw, 1.25rem) .8rem;
          }
          [data-testid="stHeader"] { background: transparent; }
          .access-shell {
            min-height: 78vh;
            display: flex;
            align-items: center;
            justify-content: center;
          }
          .access-panel {
            width: min(760px, 94vw);
            padding: 38px;
            border-radius: 28px;
            border: 1px solid rgba(125,249,255,.22);
            background:
              radial-gradient(circle at 20% 0%, rgba(34,211,238,.18), transparent 30%),
              linear-gradient(135deg, rgba(2,6,23,.86), rgba(7,17,31,.92));
            box-shadow: 0 34px 100px rgba(0,0,0,.38), inset 0 1px 0 rgba(255,255,255,.06);
            text-align: center;
          }
          .access-brand {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 14px;
          }
          .access-brand-copy { text-align: left; }
          .access-brand-kicker {
            display: block;
            margin-bottom: 3px;
            color: #67e8f9;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: .16em;
            text-transform: uppercase;
          }
          .access-panel h1 {
            margin: 0;
            color: #f8feff;
            font-size: clamp(38px, 4vw, 48px);
            font-weight: 800;
            letter-spacing: -.045em;
            line-height: .98;
          }
          .access-panel h1 span,
          .app-heading h1 span {
            color: #67e8f9;
            background: linear-gradient(105deg, #f8feff 0%, #67e8f9 48%, #50ccaa 100%);
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
          }
          .brand-pulse {
            position: relative;
            display: inline-block;
            flex: 0 0 auto;
            width: 32px;
            height: 32px;
            border: 1px solid rgba(103, 232, 249, .48);
            border-radius: 50%;
            background: radial-gradient(circle, #67e8f9 0 18%, rgba(80,204,170,.28) 20% 42%, transparent 44%);
            box-shadow: 0 0 24px rgba(34,211,238,.24), inset 0 0 16px rgba(103,232,249,.12);
          }
          .brand-pulse::after {
            content: "";
            position: absolute;
            inset: 6px;
            border: 1px solid rgba(248, 254, 255, .55);
            border-radius: 50%;
          }
          .access-panel p { color: #cbd5e1; font-size: 16px; line-height: 1.55; margin: 14px auto 0; max-width: 620px; }
          div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: rgba(125, 249, 255, .18);
            border-radius: 22px;
            background: rgba(2, 6, 23, .62);
            box-shadow: inset 0 1px 0 rgba(255,255,255,.04);
          }
          div[data-testid="stChatMessage"] {
            border: 1px solid rgba(148, 163, 184, .16);
            border-radius: 14px;
            background: rgba(15, 23, 42, .72);
            padding: 8px 10px;
          }
          div[data-testid="stChatMessage"] p,
          div[data-testid="stChatMessage"] li { font-size: 13px; line-height: 1.45; }
          .valencia-resizable-row {
            align-items: stretch !important;
            gap: 0 !important;
            width: 100% !important;
            min-width: 0 !important;
          }
          .valencia-main-column,
          .valencia-chat-column { min-width: 0; }
          .valencia-main-column { container-type: inline-size; }
          .valencia-main-column iframe,
          .valencia-chat-column iframe { max-width: 100%; }
          .valencia-chat-resizer {
            position: relative;
            flex: 0 0 32px;
            width: 32px;
            min-height: 220px;
            cursor: col-resize;
            touch-action: none;
            user-select: none;
            outline: none;
            z-index: 20;
          }
          .valencia-chat-resizer::before {
            content: "";
            position: absolute;
            inset: 0 14px;
            border-radius: 999px;
            background: rgba(103, 232, 249, .26);
            box-shadow: 0 0 0 1px rgba(125, 249, 255, .16), 0 0 16px rgba(34, 211, 238, .12);
            transition: background .16s ease, box-shadow .16s ease, inset .16s ease;
          }
          .valencia-chat-resizer:hover::before,
          .valencia-chat-resizer:focus-visible::before,
          body.valencia-chat-resizing .valencia-chat-resizer::before {
            inset: 0 12px;
            background: rgba(103, 232, 249, .78);
            box-shadow: 0 0 0 1px rgba(224, 242, 254, .72), 0 0 22px rgba(34, 211, 238, .42);
          }
          body.valencia-chat-resizing,
          body.valencia-chat-resizing * { cursor: col-resize !important; user-select: none !important; }
          body.valencia-dashboard-active section[data-testid="stMain"] { overflow-y: hidden !important; }
          section[data-testid="stMain"]:has(#valencia-main-anchor) { overflow-y: hidden !important; }
          section[data-testid="stMain"]:has(#valencia-main-anchor) .block-container { padding-bottom: 0 !important; }
          body.valencia-dashboard-active .block-container { padding-bottom: 0 !important; }
          body.valencia-dashboard-active [data-testid="stLayoutWrapper"]:has(.valencia-resizable-row)
            + [data-testid="stElementContainer"]:has(.stHtml) {
            position: absolute !important;
            width: 0 !important;
            height: 0 !important;
            overflow: hidden !important;
          }
          body.valencia-dashboard-active .valencia-resizable-row,
          body.valencia-dashboard-active .valencia-main-column,
          body.valencia-dashboard-active .valencia-chat-column { overflow: hidden; }
          [data-testid="stElementContainer"]:has(#valencia-main-scroll-mode) {
            position: absolute !important;
            width: 0 !important;
            height: 0 !important;
            overflow: hidden !important;
          }
          body.valencia-dashboard-active .valencia-main-column.valencia-main-scrollable {
            overflow-x: hidden !important;
            overflow-y: auto !important;
            overscroll-behavior: contain;
            scrollbar-gutter: stable;
          }
          .valencia-main-scrollable::-webkit-scrollbar { width: 9px; }
          .valencia-main-scrollable::-webkit-scrollbar-track { background: rgba(2, 6, 23, .35); }
          .valencia-main-scrollable::-webkit-scrollbar-thumb {
            border: 2px solid transparent;
            border-radius: 999px;
            background: rgba(103, 232, 249, .42);
            background-clip: padding-box;
          }
          .side-title {
            color: #67e8f9;
            text-transform: uppercase;
            font-size: 11px;
            margin-bottom: 10px;
          }
          .st-key-control_bar {
            padding: 12px 14px;
            margin-bottom: 8px;
            border: 1px solid rgba(125, 249, 255, .18);
            border-radius: 20px;
            background: rgba(2, 6, 23, .70);
            box-shadow: inset 0 1px 0 rgba(255,255,255,.05), 0 18px 42px rgba(0,0,0,.18);
            overflow-x: auto;
            overflow-y: hidden;
          }
          .st-key-control_bar div[data-testid="stHorizontalBlock"] {
            flex-wrap: nowrap !important;
            align-items: flex-start;
          }
          .st-key-control_bar div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child {
            flex: 0 1 430px !important;
            width: auto !important;
            min-width: 390px !important;
          }
          .st-key-control_bar div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:last-child {
            flex: 0 1 560px !important;
            width: auto !important;
            min-width: 455px !important;
            margin-left: auto;
          }
          .st-key-control_bar [data-testid="stSegmentedControl"] {
            min-width: 455px;
          }
          .st-key-control_bar [data-testid="stSegmentedControl"] [role="radiogroup"] {
            flex-wrap: nowrap !important;
          }
          .st-key-control_bar [data-testid="stSegmentedControl"] button {
            min-width: 0 !important;
            white-space: nowrap !important;
            padding-inline: clamp(8px, 1vw, 18px) !important;
          }
          .app-heading {
            display: flex;
            align-items: center;
            gap: 18px;
            padding: 0 2px 8px;
          }
          .app-brand {
            display: inline-flex;
            align-items: center;
            flex: 0 0 auto;
            gap: 9px;
          }
          .app-brand .brand-pulse {
            width: 22px;
            height: 22px;
          }
          .app-brand .brand-pulse::after { inset: 4px; }
          .app-heading h1 {
            flex: 0 0 auto;
            margin: 0;
            padding: 0;
            color: #f8feff;
            font-size: clamp(24px, 1.7vw, 30px);
            font-weight: 800;
            letter-spacing: -.04em;
            line-height: 1.05;
          }
          .app-heading p { margin: 0; color: #94a3b8; font-size: 13px; }
          .control-label {
            color: #67e8f9;
            text-transform: uppercase;
            font-size: 10px;
            margin: 0 0 7px 0;
          }
          .status-strip {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
            gap: 8px;
            margin-bottom: 10px;
          }
          .status-strip.history-kpis { grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); }
          .status-item {
            border: 1px solid rgba(125, 249, 255, .12);
            border-radius: 14px;
            background: rgba(8, 47, 73, .42);
            padding: 9px 11px;
            min-height: 64px;
          }
          .status-item span { display: block; color: #94a3b8; font-size: 10px; text-transform: uppercase; }
          .status-item b { display: block; margin-top: 2px; color: #f8feff; font-size: 15px; overflow-wrap: anywhere; }
          .quality-dot {
            display: inline-block;
            width: 9px;
            height: 9px;
            border-radius: 999px;
            margin-right: 6px;
            box-shadow: 0 0 14px currentColor;
          }
          .history-card {
            padding: 20px;
            margin-bottom: 10px;
            border: 1px solid rgba(125,249,255,.30);
            border-radius: 24px;
            background:
              radial-gradient(circle at 14% 8%, rgba(34,211,238,.16), transparent 28%),
              linear-gradient(135deg, rgba(2,6,23,.76), rgba(7,17,31,.88));
            box-shadow: 0 30px 90px rgba(0,0,0,.34), inset 0 1px 0 rgba(255,255,255,.06);
          }
          .info-panel {
            min-height: 815px;
            padding: 20px;
            border: 1px solid rgba(125,249,255,.30);
            border-radius: 24px;
            background:
              radial-gradient(circle at 14% 8%, rgba(34,211,238,.16), transparent 28%),
              linear-gradient(135deg, rgba(2,6,23,.76), rgba(7,17,31,.88));
            box-shadow: 0 30px 90px rgba(0,0,0,.34), inset 0 1px 0 rgba(255,255,255,.06);
          }
          .history-title {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 12px;
          }
          .history-title h2 { margin: 0; color: #f8feff; font-size: 28px; }
          .history-title p { margin: 7px 0 0; color: #cbd5e1; font-size: 13px; }
          .station-picker {
            margin-bottom: 10px;
            padding: 14px 16px;
            border: 1px solid rgba(125,249,255,.20);
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(8,47,73,.58), rgba(2,6,23,.62));
            box-shadow: inset 0 1px 0 rgba(255,255,255,.05);
          }
          .station-picker strong {
            display: block;
            color: #f8feff;
            font-size: 15px;
            margin-bottom: 3px;
          }
          .station-picker span {
            display: block;
            color: #94a3b8;
            font-size: 12px;
          }
          div[data-testid="stSelectbox"] label {
            color: #67e8f9 !important;
            text-transform: uppercase;
            font-size: 11px !important;
          }
          .info-grid {
            display: grid;
            grid-template-columns: 1.15fr .85fr;
            gap: 14px;
            margin-top: 12px;
          }
          .info-box {
            border: 1px solid rgba(125,249,255,.16);
            border-radius: 16px;
            background: rgba(2,6,23,.46);
            padding: 14px;
          }
          .quality-table {
            width: 100%;
            border-collapse: collapse;
            color: #dbeafe;
            font-size: 12px;
          }
          .quality-table th, .quality-table td {
            border-bottom: 1px solid rgba(148,163,184,.16);
            padding: 9px 8px;
            text-align: left;
          }
          .quality-table th {
            color: #67e8f9;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0;
          }
          .abbr-list {
            display: grid;
            gap: 9px;
            color: #cbd5e1;
            font-size: 13px;
            line-height: 1.38;
          }
          .threshold-note {
            color: #94a3b8;
            font-size: 12px;
            line-height: 1.45;
            margin-top: 15px;
          }
          @media (min-width: 901px) {
            .valencia-main-column { min-width: 420px; }
            .valencia-chat-column { min-width: 280px; }
          }
          @media (max-width: 900px) {
            .block-container { padding: .7rem .65rem .9rem; }
            .valencia-resizable-row {
              flex-direction: column !important;
              gap: .85rem !important;
            }
            .valencia-main-column,
            .valencia-chat-column {
              flex: 1 1 auto !important;
              width: 100% !important;
              min-width: 0 !important;
            }
            .valencia-chat-resizer { display: none !important; }
            .history-title { flex-direction: column; }
            .history-title h2 { font-size: clamp(22px, 6vw, 28px); }
            .app-heading { align-items:flex-start; flex-direction:column; gap:4px; }
            .status-strip,
            .status-strip.history-kpis { grid-template-columns: repeat(auto-fit, minmax(135px, 1fr)); }
          }
          @media (max-width: 560px) {
            .block-container { padding-inline: .45rem; }
            .status-strip,
            .status-strip.history-kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .status-item { padding: 8px 9px; }
            .status-item b { font-size: 14px; }
          }
          @container (max-width: 650px) {
            .valencia-main-column div[data-testid="stHorizontalBlock"] {
              flex-direction: column !important;
              gap: .65rem !important;
            }
            .valencia-main-column div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
              flex: 1 1 auto !important;
              width: 100% !important;
              min-width: 0 !important;
            }
            .valencia-main-column .st-key-control_bar div[data-testid="stHorizontalBlock"] {
              flex-direction: row !important;
              flex-wrap: nowrap !important;
              gap: .65rem !important;
            }
            .valencia-main-column .st-key-control_bar div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child {
              flex: 0 0 390px !important;
              width: 390px !important;
              min-width: 390px !important;
            }
            .valencia-main-column .st-key-control_bar div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:last-child {
              flex: 0 0 455px !important;
              width: 455px !important;
              min-width: 455px !important;
              margin-left: 0;
            }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def leaflet_map(points: list[dict], pollutant: str, mode: str) -> None:
    component = f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
          html, body {{ margin:0; padding:0; background:#020617; font-family:Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; }}
          .map-frame {{ position:relative; height:100vh; box-sizing:border-box; overflow:hidden; border:1px solid rgba(125,249,255,.30); border-radius:24px; box-shadow:0 30px 90px rgba(0,0,0,.42), inset 0 1px 0 rgba(255,255,255,.08); background:#07111f; }}
          #map {{ position:absolute; inset:0; z-index:1; background:#07111f; }}
          .map-frame::after {{ content:""; pointer-events:none; position:absolute; inset:0; z-index:600; background:radial-gradient(circle at 20% 18%, rgba(34,211,238,.18), transparent 28%), radial-gradient(circle at 84% 72%, rgba(59,130,246,.15), transparent 28%), linear-gradient(180deg, rgba(2,6,23,.10), rgba(2,6,23,.22)); mix-blend-mode:screen; }}
          .legend {{ position:absolute; right:18px; bottom:18px; z-index:1000; display:flex; gap:10px; flex-wrap:wrap; max-width:520px; padding:11px 13px; border:1px solid rgba(125,249,255,.22); border-radius:999px; background:rgba(2,6,23,.76); color:#e2e8f0; font-size:12px; backdrop-filter:blur(10px); pointer-events:none; }}
          .legend span {{ display:inline-flex; align-items:center; gap:6px; }}
          .legend i {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
          .leaflet-control-zoom a {{ background:rgba(2,6,23,.82) !important; color:#e0f2fe !important; border-color:rgba(125,249,255,.22) !important; }}
          .leaflet-popup-content-wrapper, .leaflet-popup-tip {{ background:rgba(2,6,23,.94); color:#e5f8ff; border:1px solid rgba(125,249,255,.22); box-shadow:0 18px 45px rgba(0,0,0,.45); }}
          .popup-title {{ color:#fff; font-weight:800; font-size:15px; margin-bottom:4px; }}
          .popup-subtitle {{ color:#93c5fd; font-size:12px; margin-bottom:10px; }}
          .popup-value {{ color:#a7f3d0; font-weight:800; font-size:22px; }}
          .pulse-ring {{ filter: drop-shadow(0 0 16px rgba(103,232,249,.35)); }}
          @media (max-width: 600px) {{
            .legend {{ left:10px; right:10px; bottom:10px; max-width:none; border-radius:18px; gap:7px 10px; }}
            .leaflet-control-zoom {{ display:none; }}
          }}
        </style>
      </head>
      <body>
        <div class="map-frame">
          <div id="map"></div>
          <div class="legend">
            <span><i style="background:#50f0e6"></i>buena</span>
            <span><i style="background:#50ccaa"></i>raz. buena</span>
            <span><i style="background:#f0e641"></i>regular</span>
            <span><i style="background:#ff5050"></i>desfavorable</span>
            <span><i style="background:#960032"></i>muy desf.</span>
            <span><i style="background:#7d2181"></i>extrema</span>
          </div>
        </div>
        <script>
          const points = {json.dumps(points, ensure_ascii=False)};
          const map = L.map('map', {{ zoomControl:true, scrollWheelZoom:true, preferCanvas:true }}).setView([39.4699, -0.3763], 12.35);
          L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{ maxZoom:19, attribution:'&copy; OpenStreetMap' }}).addTo(map);
          const bounds = [];
          points.forEach((point) => {{
            const marker = L.circleMarker([point.lat, point.lon], {{
              radius: point.radius, color:'#f8fafc', weight:2.5, fillColor:point.color, fillOpacity:.82, opacity:.98, className:'pulse-ring'
            }}).addTo(map);
            marker.bindPopup(`<div class="popup-title">${{point.label}}</div><div class="popup-subtitle">${{point.station}}</div><div class="popup-value">${{point.value_label}}</div>`);
            bounds.push([point.lat, point.lon]);
          }});
        </script>
      </body>
    </html>
    """
    st.iframe(component, height=845, width="stretch")


def access_screen() -> None:
    if "pipeline_error" in st.session_state:
        st.error(st.session_state["pipeline_error"])
    st.markdown(
        """
        <div class="access-shell">
          <div class="access-panel">
            <div class="access-brand">
              <i class="brand-pulse" aria-hidden="true"></i>
              <div class="access-brand-copy">
                <span class="access-brand-kicker">Observatorio urbano</span>
                <h1>Valencia <span>Respira</span></h1>
              </div>
            </div>
            <p>Al acceder se comprueba la medicion actual. Solo se crea una nueva instantanea historica cuando los datos han cambiado.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, center, _ = st.columns([1.2, 1, 1.2])
    with center:
        if st.button("ACCEDER", type="primary", width="stretch"):
            with st.spinner("Comprobando si hay una nueva medicion..."):
                try:
                    result = run_manual_pipeline()
                except Exception as exc:
                    st.session_state["pipeline_error"] = f"No se ha podido actualizar ahora: {exc}"
                    st.rerun()
            st.cache_data.clear()
            st.session_state.pop("pipeline_error", None)
            st.session_state.pop("ai_summary", None)
            st.session_state.pop("ai_summary_error", None)
            st.session_state.pop("chat_messages", None)
            st.session_state["access_granted"] = True
            st.session_state["last_pipeline_result"] = result
            st.rerun()


def chat_panel() -> None:
    if st.session_state.get("ai_summary_version") != AI_SUMMARY_VERSION:
        st.session_state.pop("ai_summary", None)
        st.session_state.pop("ai_summary_error", None)
        st.session_state["ai_summary_version"] = AI_SUMMARY_VERSION

    summary_tab, chat_tab = st.tabs(["Resumen IA", "Chat"])

    with summary_tab:
        st.markdown('<div class="side-title">Resumen IA</div>', unsafe_allow_html=True)
        if "ai_summary" not in st.session_state:
            with st.spinner("Generando resumen..."):
                try:
                    st.session_state["ai_summary"] = summarize_air_quality()
                except Exception as exc:
                    st.session_state["ai_summary_error"] = str(exc)

        if "ai_summary_error" in st.session_state:
            st.error(f"No se ha podido generar el resumen: {st.session_state['ai_summary_error']}")
            if st.button("Reintentar resumen", width="stretch"):
                st.session_state.pop("ai_summary", None)
                st.session_state.pop("ai_summary_error", None)
                st.rerun()
        else:
            with st.container(height=662, border=True):
                st.markdown(st.session_state["ai_summary"])

    with chat_tab:
        st.markdown('<div class="side-title">Chat</div>', unsafe_allow_html=True)
        if "chat_messages" not in st.session_state:
            st.session_state["chat_messages"] = [
                ("bot", "Pregunta sobre valores actuales, predicciones, zonas o contaminantes.")
            ]

        with st.container(height=662, border=True):
            for role, message in st.session_state["chat_messages"][-8:]:
                author = "user" if role == "user" else "assistant"
                with st.chat_message(author):
                    st.markdown(message)

        with st.form("ai_chat_form", clear_on_submit=True):
            prompt = st.text_input("Mensaje", placeholder="Escribe una pregunta")
            submitted = st.form_submit_button("Enviar", width="stretch")
            if submitted and prompt.strip():
                question = prompt.strip()
                st.session_state["chat_messages"].append(("user", question))
                with st.spinner("Consultando Mistral..."):
                    try:
                        answer = ask_air_quality(question)
                    except Exception as exc:
                        answer = f"No se ha podido responder ahora: {exc}"
                st.session_state["chat_messages"].append(("bot", answer))
                st.rerun()


def install_chat_resizer() -> None:
    st.html(
        """
        <script>
          (() => {
            const host = window.parent;
            const doc = host.document;
            const storageKey = "valencia-air-chat-width";
            const defaultWidth = 30;

            const readWidth = () => {
              try {
                const rawValue = host.localStorage.getItem(storageKey);
                if (rawValue === null) return defaultWidth;
                const stored = Number(rawValue);
                return Number.isFinite(stored) ? stored : defaultWidth;
              } catch (_) {
                return defaultWidth;
              }
            };

            const saveWidth = (value) => {
              try { host.localStorage.setItem(storageKey, String(value)); } catch (_) {}
            };

            const initialize = (attempt = 0) => {
              const mainAnchor = doc.getElementById("valencia-main-anchor");
              const chatAnchor = doc.getElementById("valencia-chat-anchor");
              const mainColumn = mainAnchor?.closest('[data-testid="stColumn"]');
              const chatColumn = chatAnchor?.closest('[data-testid="stColumn"]');
              const row = mainColumn?.closest('[data-testid="stHorizontalBlock"]');

              if (!mainColumn || !chatColumn || !row || !row.contains(chatColumn)) {
                if (attempt < 40) host.setTimeout(() => initialize(attempt + 1), 50);
                return;
              }

              if (typeof host.__valenciaChatResizerCleanup === "function") {
                host.__valenciaChatResizerCleanup();
              }

              row.classList.add("valencia-resizable-row");
              mainColumn.classList.add("valencia-main-column");
              chatColumn.classList.add("valencia-chat-column");

              let handle = row.querySelector(":scope > .valencia-chat-resizer");
              if (!handle) {
                handle = doc.createElement("div");
                handle.className = "valencia-chat-resizer";
                handle.setAttribute("role", "separator");
                handle.setAttribute("aria-orientation", "vertical");
                handle.setAttribute("aria-label", "Cambiar ancho del chat");
                handle.setAttribute("title", "Arrastra para cambiar el ancho del chat. Doble clic para restablecer.");
                handle.tabIndex = 0;
                row.insertBefore(handle, chatColumn);
              }

              let currentWidth = readWidth();
              let dragging = false;

              const limits = () => {
                const available = Math.max(row.getBoundingClientRect().width - handle.offsetWidth, 1);
                return {
                  min: Math.max(22, 280 / available * 100),
                  max: Math.min(48, (available - 420) / available * 100),
                };
              };

              const applyWidth = (requested, persist = false) => {
                const range = limits();
                const safeMax = Math.max(range.min, range.max);
                currentWidth = Math.min(safeMax, Math.max(range.min, requested));
                chatColumn.style.setProperty("flex", `0 0 ${currentWidth}%`);
                chatColumn.style.setProperty("width", `${currentWidth}%`);
                mainColumn.style.setProperty("flex", "1 1 0%");
                mainColumn.style.setProperty("width", "auto");
                handle.setAttribute("aria-valuemin", String(Math.round(range.min)));
                handle.setAttribute("aria-valuemax", String(Math.round(safeMax)));
                handle.setAttribute("aria-valuenow", String(Math.round(currentWidth)));
                if (persist) saveWidth(currentWidth);
              };

              const fitViewport = () => {
                const mobile = host.matchMedia("(max-width: 900px)").matches;
                const mainIframe = mainColumn.querySelector("iframe");
                const scrollMarker = mainColumn.querySelector("#valencia-main-scroll-mode");
                const scrollMode = scrollMarker?.dataset.mode || "fixed";
                const mainShouldScroll = scrollMarker?.dataset.scroll === "true";
                const chatScrollers = [...chatColumn.querySelectorAll('[data-testid="stVerticalBlock"]')]
                  .filter((element) =>
                    host.getComputedStyle(element).overflowY === "auto"
                    && element.getBoundingClientRect().height > 0
                  );

                if (mobile) {
                  doc.body.classList.remove("valencia-dashboard-active");
                  mainColumn.classList.remove("valencia-main-scrollable");
                  delete mainColumn.dataset.scrollMode;
                  row.style.removeProperty("height");
                  row.style.removeProperty("max-height");
                  mainColumn.style.removeProperty("height");
                  chatColumn.style.removeProperty("height");
                  if (mainIframe) {
                    mainIframe.style.removeProperty("height");
                    mainIframe.closest('[data-testid="stElementContainer"]')?.style.removeProperty("height");
                  }
                  chatScrollers.forEach((scroller) => {
                    scroller.style.removeProperty("height");
                    scroller.style.removeProperty("min-height");
                    scroller.style.removeProperty("max-height");
                    if (scroller.parentElement) {
                      scroller.parentElement.style.removeProperty("height");
                      scroller.parentElement.style.removeProperty("min-height");
                      scroller.parentElement.style.removeProperty("max-height");
                      scroller.parentElement.style.removeProperty("flex");
                    }
                  });
                  return;
                }

                doc.body.classList.add("valencia-dashboard-active");
                if (mainColumn.dataset.scrollMode !== scrollMode) {
                  mainColumn.scrollTop = 0;
                  mainColumn.dataset.scrollMode = scrollMode;
                }
                mainColumn.classList.toggle("valencia-main-scrollable", mainShouldScroll);
                const viewportBottom = host.innerHeight - 8;
                const rowTop = row.getBoundingClientRect().top;
                const availableHeight = Math.max(320, viewportBottom - rowTop);
                row.style.setProperty("height", `${availableHeight}px`);
                row.style.setProperty("max-height", `${availableHeight}px`);
                mainColumn.style.setProperty("height", `${availableHeight}px`);
                chatColumn.style.setProperty("height", `${availableHeight}px`);

                if (mainIframe) {
                  const iframeHeight = Math.max(260, viewportBottom - mainIframe.getBoundingClientRect().top);
                  mainIframe.style.setProperty("height", `${iframeHeight}px`);
                  mainIframe.closest('[data-testid="stElementContainer"]')?.style.setProperty("height", `${iframeHeight}px`);
                }

                chatScrollers.forEach((scroller) => {
                  const panel = scroller.closest('[role="tabpanel"]');
                  const form = panel?.querySelector('[data-testid="stForm"]');
                  const formRect = form?.getBoundingClientRect();
                  const reserve = formRect && formRect.height > 0 ? formRect.height + 12 : 0;
                  const scrollerHeight = Math.max(
                    220,
                    viewportBottom - scroller.getBoundingClientRect().top - reserve,
                  );
                  scroller.style.setProperty("height", `${scrollerHeight}px`, "important");
                  scroller.style.setProperty("min-height", "0", "important");
                  scroller.style.setProperty("max-height", `${scrollerHeight}px`, "important");
                  if (scroller.parentElement) {
                    scroller.parentElement.style.setProperty("height", `${scrollerHeight}px`, "important");
                    scroller.parentElement.style.setProperty("min-height", "0", "important");
                    scroller.parentElement.style.setProperty("max-height", `${scrollerHeight}px`, "important");
                    scroller.parentElement.style.setProperty("flex", `0 0 ${scrollerHeight}px`, "important");
                  }
                });
              };

              const onPointerDown = (event) => {
                if (host.matchMedia("(max-width: 900px)").matches) return;
                dragging = true;
                doc.body.classList.add("valencia-chat-resizing");
                handle.setPointerCapture?.(event.pointerId);
                event.preventDefault();
              };

              const onPointerMove = (event) => {
                if (!dragging) return;
                const rect = row.getBoundingClientRect();
                const available = Math.max(rect.width - handle.offsetWidth, 1);
                applyWidth((rect.right - event.clientX) / available * 100);
                host.requestAnimationFrame(fitViewport);
                event.preventDefault();
              };

              const onPointerUp = () => {
                if (!dragging) return;
                dragging = false;
                doc.body.classList.remove("valencia-chat-resizing");
                saveWidth(currentWidth);
                fitViewport();
                host.dispatchEvent(new Event("resize"));
              };

              const onKeyDown = (event) => {
                if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
                applyWidth(currentWidth + (event.key === "ArrowLeft" ? 2 : -2), true);
                event.preventDefault();
              };

              const onDoubleClick = () => {
                applyWidth(defaultWidth, true);
                fitViewport();
              };
              const onResize = () => {
                applyWidth(currentWidth);
                fitViewport();
              };
              const onChatClick = () => host.setTimeout(fitViewport, 0);
              const layoutObserver = new MutationObserver(() => host.setTimeout(fitViewport, 0));
              layoutObserver.observe(row, { childList: true, subtree: true });

              handle.addEventListener("pointerdown", onPointerDown);
              handle.addEventListener("keydown", onKeyDown);
              handle.addEventListener("dblclick", onDoubleClick);
              doc.addEventListener("pointermove", onPointerMove);
              doc.addEventListener("pointerup", onPointerUp);
              host.addEventListener("resize", onResize);
              chatColumn.addEventListener("click", onChatClick);

              host.__valenciaChatResizerCleanup = () => {
                handle.removeEventListener("pointerdown", onPointerDown);
                handle.removeEventListener("keydown", onKeyDown);
                handle.removeEventListener("dblclick", onDoubleClick);
                doc.removeEventListener("pointermove", onPointerMove);
                doc.removeEventListener("pointerup", onPointerUp);
                host.removeEventListener("resize", onResize);
                chatColumn.removeEventListener("click", onChatClick);
                layoutObserver.disconnect();
                doc.body.classList.remove("valencia-chat-resizing");
                doc.body.classList.remove("valencia-dashboard-active");
              };

              applyWidth(currentWidth);
              host.requestAnimationFrame(fitViewport);
              host.setTimeout(fitViewport, 120);
              host.setTimeout(fitViewport, 600);
            };

            initialize();
          })();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def toolbar() -> tuple[str, str]:
    with st.container(key="control_bar"):
        col1, col2 = st.columns([1, 1.35], gap="small")
        with col1:
            st.markdown('<div class="control-label">Contaminante</div>', unsafe_allow_html=True)
            pollutant = st.pills("Contaminante", POLLUTANTS, default="NO2", label_visibility="collapsed")
        with col2:
            st.markdown('<div class="control-label">Vista</div>', unsafe_allow_html=True)
            mode = st.segmented_control(
                "Vista",
                ["Actual", "Prediccion", "Historico", "KPIs", "Info"],
                default="Actual",
                label_visibility="collapsed",
            )
    return pollutant or "NO2", mode or "Actual"


def status_strip(df: pd.DataFrame, pollutant: str, mode: str) -> None:
    values = df[pollutant].dropna()
    total_stations = len(df)
    no_data = total_stations - len(values)
    mean = f"{values.mean():.1f} ug/m3" if len(values) else "sin datos"
    if len(values):
        max_index = values.idxmax()
        maximum = f"{values.max():.1f} ug/m3"
        max_station = df.loc[max_index, "station_display"]
        worst_quality = quality_for_value(pollutant, values.max())
        qualities = [quality_for_value(pollutant, value) for value in values]
        desired_count = sum(level in {"Buena", "Razonablemente buena"} for level in qualities)
        regular_or_worse = len(values) - desired_count
        desired_share = f"{desired_count}/{len(values)}"
    else:
        maximum = "sin datos"
        max_station = "-"
        worst_quality = "No hay datos"
        desired_count = 0
        regular_or_worse = 0
        desired_share = "0/0"
    quality_color = QUALITY_COLORS[worst_quality]
    st.markdown(
        f"""
        <div class="status-strip">
          <div class="status-item"><span>Media</span><b>{html.escape(mean)}</b></div>
          <div class="status-item"><span>Maximo</span><b>{html.escape(maximum)}</b><span>{html.escape(str(max_station))}</span></div>
          <div class="status-item"><span>Peor calidad</span><b><i class="quality-dot" style="background:{quality_color}; color:{quality_color};"></i>{html.escape(worst_quality)}</b></div>
          <div class="status-item"><span>Calidad deseada</span><b>{desired_share}</b><span>buena o razonable</span></div>
          <div class="status-item"><span>Regular o peor</span><b>{regular_or_worse}</b><span>sobre {len(values)} con dato</span></div>
          <div class="status-item"><span>Sin datos</span><b>{no_data}</b><span>no aparecen en mapa</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def measurement_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, station in df.iterrows():
        for pollutant in POLLUTANTS:
            value = station[pollutant]
            if pd.isna(value):
                continue
            quality = quality_for_value(pollutant, value)
            rank = QUALITY_RANK[quality]
            upper = QUALITY_THRESHOLDS[pollutant][rank]
            rows.append(
                {
                    "station": station["station_display"],
                    "pollutant": pollutant,
                    "value": float(value),
                    "quality": quality,
                    "rank": rank,
                    "severity": rank + min(float(value) / upper, 1.5),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["station", "pollutant", "value", "quality", "rank", "severity"],
    )


def forecast_frame(current: pd.DataFrame, prediction: pd.DataFrame) -> pd.DataFrame:
    current_rows = measurement_frame(current).rename(
        columns={
            "value": "current_value",
            "quality": "current_quality",
            "rank": "current_rank",
            "severity": "current_severity",
        }
    )
    predicted_rows = measurement_frame(prediction).rename(
        columns={
            "value": "predicted_value",
            "quality": "predicted_quality",
            "rank": "predicted_rank",
            "severity": "predicted_severity",
        }
    )
    if current_rows.empty:
        return current_rows
    comparison = current_rows.merge(predicted_rows, on=["station", "pollutant"], how="left")
    comparison["delta"] = comparison["predicted_value"] - comparison["current_value"]
    comparison["rank_change"] = comparison["predicted_rank"] - comparison["current_rank"]
    comparison["normalized_delta"] = comparison.apply(
        lambda row: row["delta"] / QUALITY_THRESHOLDS[row["pollutant"]][0]
        if not pd.isna(row["delta"])
        else np.nan,
        axis=1,
    )
    return comparison


def current_kpis_panel(current: pd.DataFrame, prediction: pd.DataFrame) -> None:
    records = measurement_frame(current)
    comparison = forecast_frame(current, prediction)
    if records.empty:
        st.warning("No hay mediciones disponibles para calcular los KPIs.")
        return

    attention = records[records["rank"] >= QUALITY_RANK["Regular"]]
    worst_record = records.sort_values(["rank", "severity"], ascending=False).iloc[0]
    worst_global = worst_record["quality"]
    worst_color = QUALITY_COLORS[worst_global]

    monitored_stations = records["station"].nunique()
    affected_stations = attention["station"].nunique()

    comparable = comparison.dropna(subset=["predicted_value"])
    worsening = comparable[comparable["rank_change"] > 0]
    improving = comparable[comparable["rank_change"] < 0]
    rising = comparable[comparable["delta"] > 0]
    if rising.empty:
        rise_value = "Sin subidas"
        rise_detail = "en las predicciones disponibles"
    else:
        largest_rise = rising.sort_values(["rank_change", "normalized_delta"], ascending=False).iloc[0]
        rise_value = largest_rise["station"]
        rise_detail = f'{largest_rise["pollutant"]} {largest_rise["delta"]:+.1f} ug/m3'

    st.markdown(
        f"""
        <section class="history-card">
          <div class="history-title" style="margin-bottom:0;">
            <div>
              <h2>Indicadores para actuar</h2>
              <p>Las cuatro señales esenciales: peor situación actual, alcance y cambios previstos a +8 horas.</p>
            </div>
            <div style="color:#67e8f9;font-size:12px;text-transform:uppercase;">{len(comparable)} comparaciones disponibles</div>
          </div>
        </section>
        <div class="status-strip">
          <div class="status-item"><span>Situación más desfavorable</span><b><i class="quality-dot" style="background:{worst_color}; color:{worst_color};"></i>{html.escape(worst_global)}</b><span>Peor dato actual: {html.escape(str(worst_record['station']))} · {worst_record['pollutant']} {worst_record['value']:.1f} ug/m3</span></div>
          <div class="status-item"><span>Zonas con nivel regular o peor</span><b>{affected_stations}/{monitored_stations}</b><span>Zonas afectadas sobre el total con mediciones</span></div>
          <div class="status-item"><span>Empeoran de categoría en +8 h</span><b>{len(worsening)}/{len(comparable)}</b><span>Comparaciones que pasan a una categoría peor · {len(improving)} mejoran</span></div>
          <div class="status-item"><span>Mayor aumento previsto en +8 h</span><b>{html.escape(str(rise_value))}</b><span>{html.escape(rise_detail)}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    summaries = []
    for pollutant in POLLUTANTS:
        pollutant_rows = records[records["pollutant"] == pollutant]
        pollutant_comparison = comparable[comparable["pollutant"] == pollutant]
        if pollutant_rows.empty:
            continue
        worst = pollutant_rows.sort_values(["rank", "severity"], ascending=False).iloc[0]
        change = pollutant_comparison["delta"].mean() if not pollutant_comparison.empty else np.nan
        summaries.append(
            {
                "Contaminante": pollutant,
                "Situación actual": worst["quality"],
                "Zona más alta": worst["station"],
                "Máximo": f'{pollutant_rows["value"].max():.1f} ug/m3',
                "Zonas a vigilar": int((pollutant_rows["rank"] >= QUALITY_RANK["Regular"]).sum()),
                "Cambio medio +8h": "sin prediccion" if pd.isna(change) else f"{change:+.1f} ug/m3",
                "Empeoran categoría": int((pollutant_comparison["rank_change"] > 0).sum()),
            }
        )
    st.markdown("#### Lectura por contaminante")
    st.dataframe(pd.DataFrame(summaries), hide_index=True, width="stretch", height=210)

    priority = comparison[
        (comparison["current_rank"] >= QUALITY_RANK["Regular"])
        | (comparison["predicted_rank"] >= QUALITY_RANK["Regular"])
        | (comparison["rank_change"] > 0)
    ].copy()
    if priority.empty:
        st.success("No hay valores regulares o peores ni deterioros de categoria previstos.")
        return

    priority["priority_rank"] = priority[["current_rank", "predicted_rank"]].max(axis=1, skipna=True)
    priority = priority.sort_values(["priority_rank", "rank_change", "current_severity"], ascending=False)
    priority["Actual"] = priority["current_value"].map(lambda value: f"{value:.1f} ug/m3")
    priority["Prediccion +8h"] = priority["predicted_value"].map(
        lambda value: "sin dato" if pd.isna(value) else f"{value:.1f} ug/m3"
    )
    priority["Cambio"] = priority["delta"].map(
        lambda value: "sin dato" if pd.isna(value) else f"{value:+.1f} ug/m3"
    )
    priority["Calidad +8h"] = priority["predicted_quality"].fillna("Sin prediccion")
    st.markdown("#### Prioridades de seguimiento")
    st.dataframe(
        priority.rename(
            columns={
                "station": "Zona",
                "pollutant": "Contaminante",
                "current_quality": "Calidad actual",
            }
        )[["Zona", "Contaminante", "Actual", "Calidad actual", "Prediccion +8h", "Calidad +8h", "Cambio"]],
        hide_index=True,
        width="stretch",
        height="auto",
    )
    st.markdown(
        """
        <div class="threshold-note">
          <b>Criterio de prioridad:</b> se incluyen las combinaciones de zona y contaminante que
          actualmente están en categoría regular o peor, que se prevé que estén en regular o peor
          dentro de 8 horas, o que empeoran de categoría. Se ordenan primero por la peor categoría
          entre la actual y la prevista, después por el mayor deterioro de categoría y, por último,
          por la severidad del valor actual. Una fila sin predicción se mantiene si su situación
          actual ya requiere seguimiento.
        </div>
        """,
        unsafe_allow_html=True,
    )


def historical_panel(pollutant: str) -> None:
    history = load_scraped_history()
    station_options = sorted(history["station_display"].dropna().unique()) if not history.empty else []

    st.markdown(
        f"""
        <section class="history-card">
        <div class="history-title" style="margin-bottom:0;">
          <div>
            <h2>Historico scrapeado</h2>
            <p>Evolucion de los valores publicados. Los valores consecutivos iguales se agrupan en un solo punto.</p>
          </div>
          <div style="color:#67e8f9;font-size:12px;text-transform:uppercase;">{html.escape(POLLUTANT_NAMES[pollutant])}</div>
        </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if not station_options:
        st.warning("Todavia no hay historico scrapeado disponible.")
        return

    st.markdown(
        """
        <div class="station-picker">
          <strong>Zona monitorizada</strong>
          <span>Selecciona una estacion para consultar la evolucion temporal del contaminante activo.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected_station = st.selectbox(
        "Estación",
        station_options,
        label_visibility="visible",
        key=f"station_{pollutant}",
    )
    observed_history = history[
        history["station_display"] == selected_station
    ][["timestamp", pollutant]].dropna()
    observed_history = observed_history.sort_values("timestamp")

    if observed_history.empty:
        st.info("No hay datos de este contaminante para la estacion seleccionada.")
        return

    station_history = collapse_consecutive_values(observed_history, pollutant)
    grouped_observations = len(observed_history) - len(station_history)
    latest_value = float(observed_history[pollutant].iloc[-1])
    latest_timestamp = observed_history["timestamp"].iloc[-1]
    latest_change_value = float(station_history[pollutant].iloc[-1])
    previous_value = float(station_history[pollutant].iloc[-2]) if len(station_history) > 1 else np.nan
    delta_value = latest_change_value - previous_value if not pd.isna(previous_value) else np.nan
    latest_quality = quality_for_value(pollutant, latest_value)
    latest_color = QUALITY_COLORS[latest_quality]
    period_max = float(observed_history[pollutant].max())
    latest_change_timestamp = station_history["timestamp"].iloc[-1]
    period_max_position = int(observed_history[pollutant].to_numpy().argmax())
    period_max_timestamp = observed_history["timestamp"].iloc[period_max_position]
    delta_label = "Sin dato previo" if pd.isna(delta_value) else f"{delta_value:+.1f} ug/m3"
    previous_detail = (
        "No existe un valor diferente anterior"
        if pd.isna(previous_value)
        else f"De {previous_value:.1f} a {latest_change_value:.1f} ug/m3 · {format_timestamp(latest_change_timestamp)}"
    )

    if grouped_observations:
        st.caption(
            f"{grouped_observations} observaciones consecutivas sin cambio se agrupan para evitar puntos repetidos. "
            f"El valor actual se observó por última vez el {format_timestamp(latest_timestamp)}."
        )

    st.markdown(
        f"""
        <div class="status-strip history-kpis">
          <div class="status-item"><span>Último valor observado</span><b>{latest_value:.1f} ug/m3</b><span>{html.escape(format_timestamp(latest_timestamp))}</span></div>
          <div class="status-item"><span>Categoría del último dato</span><b><i class="quality-dot" style="background:{latest_color}; color:{latest_color};"></i>{html.escape(latest_quality)}</b><span>Clasificación del índice de calidad del aire</span></div>
          <div class="status-item"><span>Último cambio del valor</span><b>{html.escape(delta_label)}</b><span>{html.escape(previous_detail)}</span></div>
          <div class="status-item"><span>Máximo del periodo disponible</span><b>{period_max:.1f} ug/m3</b><span>Registrado el {html.escape(format_timestamp(period_max_timestamp))}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chart_data = station_history.rename(columns={"timestamp": "Fecha", pollutant: "Valor"}).copy()
    chart_data["Calidad"] = chart_data["Valor"].map(lambda value: quality_for_value(pollutant, value))
    chart_data["FechaTexto"] = chart_data["Fecha"].map(format_timestamp)
    chart_upper = max(period_max * 1.18, QUALITY_THRESHOLDS[pollutant][0] * 1.12)
    rule_data = pd.DataFrame(
        [
            {
                "Valor": limit,
                "Umbral": f"{level}: {limit} ug/m3",
                "Color": QUALITY_COLORS[level],
            }
            for level, limit in zip(QUALITY_LEVELS, QUALITY_THRESHOLDS[pollutant])
            if limit <= chart_upper
        ]
    )

    line = (
        alt.Chart(chart_data)
        .mark_line(color="#67e8f9", strokeWidth=3)
        .encode(
            x=alt.X("Fecha:T", title="Fecha del cambio detectado", axis=alt.Axis(format="%d/%m %H:%M", labelAngle=-30)),
            y=alt.Y("Valor:Q", title=f"{pollutant} (ug/m3)", scale=alt.Scale(zero=True)),
            tooltip=[
                alt.Tooltip("Fecha:T", title="Fecha", format="%d/%m/%Y %H:%M"),
                alt.Tooltip("Valor:Q", title=f"{pollutant} ug/m3", format=".1f"),
                alt.Tooltip("Calidad:N", title="Calidad"),
            ],
        )
    )
    points = (
        alt.Chart(chart_data)
        .mark_circle(size=82, stroke="#f8fafc", strokeWidth=1.4)
        .encode(
            x="Fecha:T",
            y="Valor:Q",
            color=alt.Color(
                "Calidad:N",
                scale=alt.Scale(domain=QUALITY_LEVELS, range=[QUALITY_COLORS[level] for level in QUALITY_LEVELS]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Fecha:T", title="Fecha", format="%d/%m/%Y %H:%M"),
                alt.Tooltip("Valor:Q", title=f"{pollutant} ug/m3", format=".1f"),
                alt.Tooltip("Calidad:N", title="Calidad"),
            ],
        )
    )
    if rule_data.empty:
        chart = line + points
    else:
        rules = (
            alt.Chart(rule_data)
            .mark_rule(strokeDash=[5, 5], opacity=0.58)
            .encode(
                y="Valor:Q",
                color=alt.Color("Color:N", scale=None, legend=None),
                tooltip=[alt.Tooltip("Umbral:N", title="Umbral")],
            )
        )
        chart = rules + line + points
    st.altair_chart(chart.properties(height=500), width="stretch")
    recent = chart_data.tail(8).sort_values("Fecha", ascending=False).copy()
    recent["Fecha"] = recent["Fecha"].map(format_timestamp)
    recent["Valor"] = recent["Valor"].map(lambda value: f"{value:.1f} ug/m3")
    st.dataframe(recent[["Fecha", "Valor", "Calidad"]], hide_index=True, width="stretch", height=300)
    st.markdown(
        """
        <div class="threshold-note">
          Umbrales de color basados en el Indice Nacional de Calidad del Aire
          (Orden TEC/351/2019 y Resolucion de 2 de septiembre de 2020).
          Elaboracion propia con datos del
          <a href="https://www.valencia.es/val/qualitataire/contaminacio-atmosferica" target="_blank" style="color:#67e8f9;">Servicio de mejora climatica</a>.
          ug/m3 = microgramos por metro cubico; PM-10 y PM-2.5 = particulas en suspension.
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_panel() -> None:
    header_cells = "".join(f"<th>{html.escape(pollutant)}</th>" for pollutant in ["SO2", "NO2", "O3", "PM-10", "PM-2.5"])
    rows = []
    for index, level in enumerate(QUALITY_LEVELS):
        color = QUALITY_COLORS[level]
        values = "".join(
            f"<td>{html.escape(threshold_range(pollutant, index))}</td>"
            for pollutant in ["SO2", "NO2", "O3", "PM-10", "PM-2.5"]
        )
        rows.append(
            f"""
            <tr>
              <td><i class="quality-dot" style="background:{color}; color:{color};"></i>{html.escape(level)}</td>
              {values}
            </tr>
            """
        )

    info_html = f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <style>
          body {{ margin:0; background:transparent; font-family:Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; color:#e5f8ff; }}
          .panel {{ min-height:100vh; box-sizing:border-box; padding:20px; border:1px solid rgba(125,249,255,.30); border-radius:24px; background:radial-gradient(circle at 14% 8%, rgba(34,211,238,.16), transparent 28%), linear-gradient(135deg, rgba(2,6,23,.86), rgba(7,17,31,.94)); box-shadow:0 30px 90px rgba(0,0,0,.34), inset 0 1px 0 rgba(255,255,255,.06); }}
          .title {{ display:flex; align-items:flex-start; justify-content:space-between; gap:16px; margin-bottom:16px; }}
          h2 {{ margin:0; color:#f8feff; font-size:28px; }}
          p {{ margin:7px 0 0; color:#cbd5e1; font-size:13px; line-height:1.45; }}
          .tag {{ color:#67e8f9; font-size:12px; text-transform:uppercase; white-space:nowrap; }}
          .grid {{ display:grid; grid-template-columns:1.15fr .85fr; gap:14px; }}
          .box {{ border:1px solid rgba(125,249,255,.16); border-radius:16px; background:rgba(2,6,23,.46); padding:14px; }}
          table {{ width:100%; border-collapse:collapse; color:#dbeafe; font-size:12px; }}
          th, td {{ border-bottom:1px solid rgba(148,163,184,.16); padding:9px 8px; text-align:left; }}
          th {{ color:#67e8f9; font-size:10px; text-transform:uppercase; }}
          .dot, .quality-dot {{ display:inline-block; width:9px; height:9px; border-radius:999px; margin-right:6px; box-shadow:0 0 14px currentColor; }}
          .abbr {{ display:grid; gap:9px; color:#cbd5e1; font-size:13px; line-height:1.38; }}
          .note {{ color:#94a3b8; font-size:12px; line-height:1.45; margin-top:15px; }}
          a {{ color:#67e8f9; }}
          @media (max-width: 700px) {{
            .panel {{ padding:14px; min-height:100vh; }}
            .title {{ flex-direction:column; }}
            .grid {{ grid-template-columns:1fr; }}
            table {{ font-size:11px; }}
            th, td {{ padding:7px 5px; }}
          }}
        </style>
      </head>
      <body>
        <section class="panel">
          <div class="title">
            <div>
              <h2>Info calidad del aire</h2>
              <p>Los colores del mapa, las predicciones, los KPIs y el historico se calculan con estos rangos de calidad del aire.</p>
            </div>
            <div class="tag">umbrales oficiales</div>
          </div>
          <div class="grid">
            <div class="box">
              <table>
                <thead>
                  <tr><th>Calidad del aire</th>{header_cells}</tr>
                </thead>
                <tbody>{''.join(rows)}</tbody>
              </table>
              <div class="note">Rangos en µg/m3. ND significa que no se han recabado suficientes datos para establecer criterio. Las estaciones sin dato para un contaminante no se muestran en el mapa.</div>
            </div>
            <div class="box">
              <div class="abbr">
                <div><b>SO2</b> Dioxido de Azufre</div>
                <div><b>NO2</b> Dioxido de Nitrogeno</div>
                <div><b>O3</b> Ozono</div>
                <div><b>PM-10</b> Particulas en suspension inferiores a 10 micras</div>
                <div><b>PM-2.5</b> Particulas en suspension inferiores a 2.5 micras</div>
                <div><b>µg/m3</b> Microgramos por metro cubico</div>
                <div><b>mg/m3</b> Miligramos por metro cubico</div>
              </div>
              <div class="note">Umbrales basados en el Indice Nacional de Calidad del Aire (Orden TEC/351/2019, de 18 de marzo) y Resolucion de 2 de septiembre de 2020. Elaboracion propia con datos proporcionados por el <a href="https://www.valencia.es/val/qualitataire/contaminacio-atmosferica" target="_blank">Servicio de mejora climatica</a>.</div>
            </div>
          </div>
        </section>
      </body>
    </html>
    """
    st.iframe(info_html, height=835, width="stretch")


def dashboard() -> None:
    st.markdown(
        f"""
        <header class="app-heading">
          <div class="app-brand">
            <i class="brand-pulse" aria-hidden="true"></i>
            <h1>Valencia <span>Respira</span></h1>
          </div>
          <p>Calidad del aire urbano · Modelos entrenados con histórico hasta {model_training_cutoff()}.</p>
        </header>
        """,
        unsafe_allow_html=True,
    )
    main, chat = st.columns([7, 3], gap="small")

    with main:
        st.markdown('<span id="valencia-main-anchor"></span>', unsafe_allow_html=True)
    with chat:
        st.markdown('<span id="valencia-chat-anchor"></span>', unsafe_allow_html=True)

    install_chat_resizer()

    current = load_snapshot(SCRAPED_DIR / "latest.csv")
    prediction = load_snapshot(PREDICTIONS_DIR / "latest.csv")

    with main:
        pollutant, mode = toolbar()
        scroll_enabled = mode in {"Historico", "KPIs"}
        st.markdown(
            f'<span id="valencia-main-scroll-mode" data-mode="{html.escape(mode)}" '
            f'data-scroll="{str(scroll_enabled).lower()}"></span>',
            unsafe_allow_html=True,
        )
        if mode == "Historico":
            historical_panel(pollutant)
        elif mode == "KPIs":
            current_kpis_panel(current, prediction)
        elif mode == "Info":
            info_panel()
        else:
            shown = current if mode == "Actual" else prediction
            status_strip(shown, pollutant, mode)
            leaflet_map(map_points(shown, pollutant, mode), pollutant, mode)

    with chat:
        chat_panel()


def main() -> None:
    style_page()
    if not st.session_state.get("access_granted", False):
        access_screen()
    else:
        dashboard()


if __name__ == "__main__":
    main()
