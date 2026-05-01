# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import streamlit as st
import streamlit.components.v1 as components


# =========================================================
# 기본 설정
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

PRED_CSV_PATH = BASE_DIR / "gru_test_predictions_simple.csv"
META_JSON_PATH = BASE_DIR / "meta.json"
AREA_EXCEL_PATH = BASE_DIR / "area_info.xlsx"
SHP_PATH = BASE_DIR / "UPIS_SHP_ZON100.shp"

SOURCE_EPSG = 5174
TARGET_EPSG = 4326

START_DATETIME = pd.Timestamp("2025-01-01 00:00:00")
TIME_UNIT_MINUTES = 30

DEFAULT_DATE = "2025-11-25"
DEFAULT_TIME = "18:00"

PANEL_HEIGHT = 760
MAP_HEIGHT = 620

# 챗봇 패널 내부 높이 조정값
CHAT_GUIDE_HEIGHT = 118
CHAT_SCROLL_HEIGHT = 405


# =========================================================
# Streamlit 설정
# =========================================================
st.set_page_config(
    page_title="E-Vlog EV Demand Assistant",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# CSS
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif;
    }

    .stApp {
        background: #EEF3F8;
    }

    .block-container {
        padding-top: 0.7rem;
        padding-bottom: 1.5rem;
        max-width: 1720px;
    }

    header[data-testid="stHeader"] {
        background: rgba(238, 243, 248, 0);
        height: 0rem;
    }

    div[data-testid="stToolbar"] {
        visibility: hidden;
        height: 0%;
        position: fixed;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    section[data-testid="stSidebar"] {
        display: none;
    }

    div[data-testid="stVerticalBlock"] {
        gap: 0.56rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF;
        border-radius: 24px;
        box-shadow: 0 10px 30px rgba(35, 55, 80, 0.07);
        border: 1px solid rgba(220, 228, 238, 0.95);
    }

    .panel-title {
        color: #202532;
        font-size: 23px;
        font-weight: 900;
        letter-spacing: -0.045em;
        margin-bottom: 2px;
        line-height: 1.2;
    }

    .panel-subtitle {
        color: #7C8594;
        font-size: 13px;
        font-weight: 650;
        margin-bottom: 8px;
        line-height: 1.35;
    }

    .chat-guide-compact {
        background: #F8FAFD;
        border: 1px solid #E3EAF3;
        border-radius: 16px;
        padding: 10px 12px;
        color: #5E6878;
        font-size: 12px;
        line-height: 1.35;
        font-weight: 650;
        margin-bottom: 8px;
    }

    .status-pill {
        display: inline-flex;
        border-radius: 999px;
        padding: 4px 9px;
        background: #EAF2FF;
        color: #1F6FE5;
        font-size: 11px;
        font-weight: 900;
        margin-bottom: 6px;
    }

    .chat-example-row {
        display: flex;
        flex-direction: column;
        gap: 5px;
        margin-top: 7px;
    }

    .chat-example-small {
        background: #FFFFFF;
        border: 1px solid #E6EDF6;
        border-radius: 11px;
        padding: 6px 8px;
        color: #2F3747;
        font-size: 11px;
        font-weight: 800;
        line-height: 1.3;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .legend-wrap {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 8px;
        color: #788395;
        font-weight: 800;
        font-size: 13px;
    }

    .legend-bar {
        height: 10px;
        flex: 1;
        border-radius: 999px;
        background: linear-gradient(90deg, #D9E9FF, #76A8FF, #2E5BEA, #20145C);
    }

    .alert-card {
        background: #F8FAFD;
        border: 1px solid #E3EAF3;
        border-radius: 18px;
        padding: 15px 16px;
        margin-bottom: 12px;
    }

    .alert-top {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 12px;
    }

    .alert-title {
        color: #222633;
        font-size: 15px;
        font-weight: 900;
        line-height: 1.35;
        margin-bottom: 6px;
    }

    .alert-meta {
        color: #7C8594;
        font-size: 12px;
        font-weight: 700;
        line-height: 1.45;
    }

    .alert-value {
        color: #222633;
        font-size: 13px;
        font-weight: 900;
        margin-top: 7px;
    }

    .alert-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        padding: 5px 10px;
        font-size: 11px;
        font-weight: 900;
        color: #FFFFFF;
        background: #2E6BEA;
        white-space: nowrap;
    }

    .alert-badge.hot {
        background: #E74C5B;
    }

    .alert-badge.watch {
        background: #F59E0B;
    }

    .alert-badge.monitor {
        background: #64748B;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E8EEF5;
        border-radius: 16px;
        padding: 12px 14px;
        min-height: 95px;
        box-shadow: 0 6px 18px rgba(35, 55, 80, 0.045);
    }

    div[data-testid="stMetricLabel"] {
        color: #8A93A3;
        font-size: 12px;
        font-weight: 850;
    }

    div[data-testid="stMetricValue"] {
        color: #202532;
        font-size: 22px;
        font-weight: 900;
        letter-spacing: -0.04em;
    }

    .detail-box {
        background: #F8FAFD;
        border: 1px solid #E3EAF3;
        border-radius: 18px;
        padding: 16px 16px;
        margin-bottom: 12px;
    }

    .detail-title {
        color: #202532;
        font-size: 22px;
        font-weight: 900;
        letter-spacing: -0.045em;
        margin-bottom: 4px;
    }

    .detail-id {
        color: #2E8B55;
        font-size: 13px;
        font-weight: 900;
        margin-bottom: 10px;
    }

    .detail-label {
        color: #8A93A3;
        font-size: 12px;
        font-weight: 900;
        margin-top: 10px;
        margin-bottom: 4px;
    }

    .detail-text {
        color: #4F5A6A;
        font-size: 13px;
        font-weight: 700;
        line-height: 1.5;
    }

    .small-info {
        color: #7C8594;
        font-size: 12px;
        font-weight: 700;
        line-height: 1.5;
    }

    iframe {
        border-radius: 18px;
    }

    .mapboxgl-control-container {
        opacity: 0.92;
    }

    div[data-testid="stTextInput"] {
        margin-bottom: 0rem;
    }

    div[data-testid="stTextInput"] input {
        border-radius: 14px;
        min-height: 42px;
        font-size: 13px;
    }

    div[data-testid="stForm"] {
        border: 0;
        padding: 0;
    }

    div[data-testid="stFormSubmitButton"] button {
        border-radius: 14px;
        min-height: 42px;
        font-weight: 900;
        background: #2E6BEA;
        color: white;
        border: 0;
        margin-top: -0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 화면 보조 함수
# =========================================================
def panel_title(title: str, subtitle: str | None = None) -> None:
    if subtitle:
        st.markdown(
            f"""
            <div class="panel-title">{title}</div>
            <div class="panel-subtitle">{subtitle}</div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="panel-title">{title}</div>
            """,
            unsafe_allow_html=True,
        )


def render_legend() -> None:
    st.markdown(
        """
        <div class="legend-wrap">
            <span>낮음</span>
            <div class="legend-bar"></div>
            <span>높음</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def escape_html(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )


def render_chat_messages(messages: list[dict]) -> None:
    bubbles = ""

    for msg in messages:
        role = msg.get("role", "assistant")
        content = escape_html(msg.get("content", ""))
        role_class = "user" if role == "user" else "assistant"

        bubbles += f"""
        <div class="chat-bubble-row {role_class}">
            <div class="chat-bubble {role_class}">{content}</div>
        </div>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8" />
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            background: transparent;
            font-family: Inter, -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Noto Sans KR", sans-serif;
        }}

        .chat-scroll-box {{
            height: {CHAT_SCROLL_HEIGHT}px;
            overflow-y: auto;
            padding: 8px 4px 8px 2px;
            box-sizing: border-box;
            border-top: 1px solid #EEF2F7;
            border-bottom: 1px solid #EEF2F7;
            background: transparent;
        }}

        .chat-scroll-box::-webkit-scrollbar {{
            width: 8px;
        }}

        .chat-scroll-box::-webkit-scrollbar-thumb {{
            background: #CDD7E5;
            border-radius: 999px;
        }}

        .chat-scroll-box::-webkit-scrollbar-track {{
            background: transparent;
        }}

        .chat-bubble-row {{
            display: flex;
            margin-bottom: 12px;
        }}

        .chat-bubble-row.user {{
            justify-content: flex-end;
        }}

        .chat-bubble-row.assistant {{
            justify-content: flex-start;
        }}

        .chat-bubble {{
            max-width: 88%;
            border-radius: 16px;
            padding: 11px 13px;
            font-size: 13px;
            font-weight: 700;
            line-height: 1.55;
            word-break: keep-all;
            box-sizing: border-box;
        }}

        .chat-bubble.user {{
            background: #2E6BEA;
            color: #FFFFFF;
            border-bottom-right-radius: 5px;
        }}

        .chat-bubble.assistant {{
            background: #F8FAFD;
            color: #2F3747;
            border: 1px solid #E3EAF3;
            border-bottom-left-radius: 5px;
        }}
    </style>
    </head>
    <body>
        <div class="chat-scroll-box" id="chatbox">
            {bubbles}
        </div>
        <script>
            const box = document.getElementById("chatbox");
            box.scrollTop = box.scrollHeight;
        </script>
    </body>
    </html>
    """

    components.html(html, height=CHAT_SCROLL_HEIGHT + 8, scrolling=False)


# =========================================================
# 유틸 함수
# =========================================================
def normalize_id(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def clean_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value).strip().lower())


def clean_label(label: str) -> str:
    if not isinstance(label, str):
        return ""
    return (
        label.replace("지역생활권(", "")
        .replace("권역생활권(", "")
        .replace(")", "")
        .replace("_", " · ")
    )


def get_prediction_column(df: pd.DataFrame) -> str:
    candidates = ["y_pred_kwh", "predicted_kwh", "pred_kwh", "prediction_kwh"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError("CSV에 y_pred_kwh 또는 predicted_kwh 컬럼이 필요합니다.")


def get_true_column(df: pd.DataFrame) -> str | None:
    candidates = ["y_true_kwh", "true_kwh", "actual_kwh"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def kwh_to_color(value: float, vmin: float, vmax: float) -> list[int]:
    if pd.isna(value):
        return [220, 225, 232, 90]

    if vmax <= vmin:
        ratio = 0.5
    else:
        ratio = (value - vmin) / (vmax - vmin)
        ratio = float(np.clip(ratio, 0.0, 1.0))

    r = int(205 - 155 * ratio)
    g = int(222 - 135 * ratio)
    b = int(247 - 25 * ratio)
    a = 190

    return [r, g, b, a]


def add_datetime_to_predictions(pred: pd.DataFrame, meta: Dict) -> pd.DataFrame:
    val_end = int(meta["val_end"])
    look_back = int(meta["look_back"])
    test_time_offset = val_end + look_back

    pred = pred.copy()
    pred["global_time_idx"] = pred["sample_idx"].astype(int) + test_time_offset
    pred["datetime"] = START_DATETIME + pd.to_timedelta(
        pred["global_time_idx"] * TIME_UNIT_MINUTES,
        unit="m",
    )

    pred["date_str"] = pred["datetime"].dt.strftime("%Y-%m-%d")
    pred["time_str"] = pred["datetime"].dt.strftime("%H:%M")
    pred["hour"] = pred["datetime"].dt.hour
    pred["minute"] = pred["datetime"].dt.minute
    pred["daily_slot"] = pred["hour"] * 2 + (pred["minute"] // 30)

    return pred


# =========================================================
# 데이터 로딩
# =========================================================
@st.cache_data(show_spinner="meta.json 로딩 중...")
def load_meta(meta_path: Path) -> Dict:
    encodings = ["utf-8", "utf-8-sig", "cp949", "euc-kr"]
    last_error = None

    for enc in encodings:
        try:
            with open(meta_path, "r", encoding=enc) as f:
                meta = json.load(f)

            if "zone_ids" not in meta:
                raise ValueError("meta.json에 zone_ids가 없습니다.")

            return meta

        except UnicodeDecodeError as e:
            last_error = e
            continue

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"meta.json을 읽지 못했습니다. 마지막 오류: {last_error}",
    )


@st.cache_data(show_spinner="예측 결과 로딩 중...")
def load_predictions(pred_path: Path, meta: Dict) -> Tuple[pd.DataFrame, str, str | None]:
    pred = pd.read_csv(pred_path)

    required_cols = {"sample_idx", "zone_idx"}
    missing = required_cols - set(pred.columns)
    if missing:
        raise ValueError(f"예측 CSV에 필수 컬럼이 없습니다: {missing}")

    pred_col = get_prediction_column(pred)
    true_col = get_true_column(pred)

    pred[pred_col] = pd.to_numeric(pred[pred_col], errors="coerce")
    pred["sample_idx"] = pd.to_numeric(pred["sample_idx"], errors="coerce").astype(int)
    pred["zone_idx"] = pd.to_numeric(pred["zone_idx"], errors="coerce").astype(int)
    pred[pred_col] = pred[pred_col].clip(lower=0)

    zone_ids = meta["zone_ids"]
    zone_map = pd.DataFrame(
        {
            "zone_idx": list(range(len(zone_ids))),
            "생활권역ID": zone_ids,
        }
    )

    pred = pred.merge(zone_map, on="zone_idx", how="left")

    if pred["생활권역ID"].isna().any():
        bad = pred[pred["생활권역ID"].isna()]["zone_idx"].unique()
        raise ValueError(f"meta.json과 매칭되지 않는 zone_idx가 있습니다: {bad}")

    pred = add_datetime_to_predictions(pred, meta)
    pred = pred.rename(columns={pred_col: "predicted_kwh"})

    if true_col is not None and true_col != "y_true_kwh":
        pred = pred.rename(columns={true_col: "y_true_kwh"})

    return pred, "predicted_kwh", true_col


@st.cache_data(show_spinner="생활권역 정보 로딩 중...")
def load_area_info(excel_path: Path) -> pd.DataFrame:
    area = pd.read_excel(excel_path)

    if "생활권역ID" not in area.columns:
        raise ValueError("엑셀 파일에 생활권역ID 컬럼이 없습니다.")

    area["생활권역ID"] = area["생활권역ID"].apply(normalize_id)

    keep_cols = []
    for col in [
        "생활권역ID",
        "생활권역OBJECTID",
        "생활권역코드",
        "생활권역구분",
        "생활권역라벨",
        "행정동수",
        "행정동명목록",
    ]:
        if col in area.columns:
            keep_cols.append(col)

    area = area[keep_cols].copy()

    if "생활권역라벨" in area.columns:
        area["생활권역표시명"] = area["생활권역라벨"].apply(clean_label)
    else:
        area["생활권역표시명"] = area["생활권역ID"]

    if "행정동명목록" not in area.columns:
        area["행정동명목록"] = ""

    area["search_text"] = (
        area["생활권역ID"].astype(str)
        + " "
        + area["생활권역표시명"].astype(str)
        + " "
        + area.get("생활권역라벨", "").astype(str)
        + " "
        + area["행정동명목록"].astype(str)
    )
    area["search_text_clean"] = area["search_text"].apply(clean_text)

    return area


@st.cache_data(show_spinner="생활권역 경계 로딩 중...")
def load_living_area_gdf(shp_path: Path, meta_zone_ids: list[str]) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(shp_path, encoding="cp949")

    if "ID" not in gdf.columns:
        raise ValueError(f"SHP에 ID 컬럼이 없습니다. 현재 컬럼: {list(gdf.columns)}")

    gdf["ID"] = gdf["ID"].apply(normalize_id)

    if "ENT_NAME" in gdf.columns:
        gdf = gdf[gdf["ENT_NAME"] == "지역생활권"].copy()

    gdf = gdf[gdf["ID"].isin(meta_zone_ids)].copy()

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=SOURCE_EPSG)

    gdf = gdf.to_crs(epsg=TARGET_EPSG)
    gdf["geometry"] = gdf["geometry"].simplify(0.00012, preserve_topology=True)

    centroid = gdf.geometry.centroid
    gdf["lon"] = centroid.x
    gdf["lat"] = centroid.y

    return gdf


# =========================================================
# 자연어 질의 처리
# =========================================================
def parse_date_from_text(text: str, available_dates: list[str]) -> Optional[str]:
    m = re.search(r"(20\d{2})[-년./\s]*(\d{1,2})[-월./\s]*(\d{1,2})", text)
    if m:
        y, mo, d = map(int, m.groups())
        candidate = f"{y:04d}-{mo:02d}-{d:02d}"
        if candidate in available_dates:
            return candidate

    m = re.search(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일", text)
    if m:
        mo, d = map(int, m.groups())
        candidate = f"2025-{mo:02d}-{d:02d}"
        if candidate in available_dates:
            return candidate

    m = re.search(r"(\d{1,2})[-/.](\d{1,2})", text)
    if m:
        mo, d = map(int, m.groups())
        candidate = f"2025-{mo:02d}-{d:02d}"
        if candidate in available_dates:
            return candidate

    return None


def parse_time_from_text(text: str, available_times: list[str]) -> Optional[str]:
    m = re.search(r"(\d{1,2})\s*:\s*(\d{1,2})", text)
    if m:
        h, mi = map(int, m.groups())
        if mi >= 45:
            h += 1
            mi = 0
        elif mi >= 15:
            mi = 30
        else:
            mi = 0
        candidate = f"{h:02d}:{mi:02d}"
        if candidate in available_times:
            return candidate

    m = re.search(r"(오전|오후)\s*(\d{1,2})\s*시\s*(반)?", text)
    if m:
        ampm, h_str, half = m.groups()
        h = int(h_str)
        if ampm == "오후" and h < 12:
            h += 12
        if ampm == "오전" and h == 12:
            h = 0
        mi = 30 if half else 0
        candidate = f"{h:02d}:{mi:02d}"
        if candidate in available_times:
            return candidate

    m = re.search(r"(\d{1,2})\s*시\s*(반)?", text)
    if m:
        h = int(m.group(1))
        mi = 30 if m.group(2) else 0

        candidates = []
        if 0 <= h <= 23:
            candidates.append(f"{h:02d}:{mi:02d}")
        if 1 <= h <= 11:
            candidates.append(f"{h + 12:02d}:{mi:02d}")

        for candidate in candidates:
            if candidate in available_times:
                return candidate

    return None


def find_zone_by_location(text: str, area_info: pd.DataFrame) -> Optional[str]:
    q = clean_text(text)

    id_candidates = re.findall(r"생활권경계[_\s-]*\d{1,3}", text)
    for cand in id_candidates:
        cand_norm = cand.replace(" ", "").replace("-", "_")
        if "_" not in cand_norm:
            cand_norm = cand_norm.replace("생활권경계", "생활권경계_")
        match = area_info[area_info["생활권역ID"] == cand_norm]
        if not match.empty:
            return match["생활권역ID"].iloc[0]

    best_zone = None
    best_score = 0

    for _, row in area_info.iterrows():
        zone_id = row["생활권역ID"]
        label = str(row.get("생활권역표시명", ""))
        dongs = str(row.get("행정동명목록", ""))
        search_text = str(row.get("search_text_clean", ""))

        score = 0
        tokens = [t for t in re.split(r"[,\s·/]+", clean_text(text)) if len(t) >= 2]

        if search_text and any(token in search_text for token in tokens):
            score += 5

        for part in re.split(r"[,·/\s]+", label):
            part_clean = clean_text(part)
            if len(part_clean) >= 2 and part_clean in q:
                score += 10

        for dong in re.split(r"[,·/\s]+", dongs):
            dong_clean = clean_text(dong)
            if len(dong_clean) >= 2 and dong_clean in q:
                score += 18

        if clean_text(zone_id) in q:
            score += 30

        m = re.search(r"([가-힣]+구)", text)
        if m:
            gu = clean_text(m.group(1))
            if gu in clean_text(label) or gu in clean_text(dongs):
                score += 6

        if score > best_score:
            best_score = score
            best_zone = zone_id

    if best_score > 0:
        return best_zone

    return None


def parse_user_query(
    text: str,
    pred: pd.DataFrame,
    area_info: pd.DataFrame,
    fallback_date: str,
    fallback_time: str,
    fallback_zone_id: str,
) -> Dict:
    available_dates = sorted(pred["date_str"].unique())
    parsed_date = parse_date_from_text(text, available_dates) or fallback_date

    available_times = sorted(pred[pred["date_str"] == parsed_date]["time_str"].unique())
    parsed_time = parse_time_from_text(text, available_times) or fallback_time

    parsed_zone_id = find_zone_by_location(text, area_info) or fallback_zone_id

    return {
        "date": parsed_date,
        "time": parsed_time,
        "zone_id": parsed_zone_id,
    }


# =========================================================
# 지도 생성
# =========================================================
def prepare_map_gdf(
    boundary_gdf: gpd.GeoDataFrame,
    area_info: pd.DataFrame,
    pred_filtered: pd.DataFrame,
    focus_zone_id: Optional[str] = None,
) -> gpd.GeoDataFrame:
    gdf = boundary_gdf.copy()

    gdf = gdf.merge(
        area_info,
        left_on="ID",
        right_on="생활권역ID",
        how="left",
    )

    gdf = gdf.merge(
        pred_filtered[
            [
                "생활권역ID",
                "zone_idx",
                "sample_idx",
                "global_time_idx",
                "date_str",
                "time_str",
                "daily_slot",
                "predicted_kwh",
            ]
        ],
        on="생활권역ID",
        how="left",
    )

    if "생활권역표시명" not in gdf.columns:
        gdf["생활권역표시명"] = gdf["ID"]

    if "행정동명목록" not in gdf.columns:
        gdf["행정동명목록"] = ""

    gdf["is_focus"] = gdf["ID"] == focus_zone_id

    for col in ["zone_idx", "sample_idx", "global_time_idx", "daily_slot"]:
        if col in gdf.columns:
            gdf[col] = gdf[col].astype("Int64").astype(str)

    if "predicted_kwh" in gdf.columns:
        gdf["predicted_kwh"] = gdf["predicted_kwh"].astype(float).round(3)

    return gdf


def make_polygon_layer(map_gdf: gpd.GeoDataFrame) -> pdk.Layer:
    vmin = float(map_gdf["predicted_kwh"].quantile(0.05))
    vmax = float(map_gdf["predicted_kwh"].quantile(0.95))

    map_gdf = map_gdf.copy()

    map_gdf["fill_color"] = map_gdf.apply(
        lambda row: [255, 130, 80, 220]
        if bool(row.get("is_focus", False))
        else kwh_to_color(row["predicted_kwh"], vmin, vmax),
        axis=1,
    )

    map_gdf["line_color"] = map_gdf["is_focus"].apply(
        lambda x: [255, 60, 60, 255] if bool(x) else [255, 255, 255, 185]
    )

    map_gdf["line_width"] = map_gdf["is_focus"].apply(
        lambda x: 90 if bool(x) else 20
    )

    return pdk.Layer(
        "GeoJsonLayer",
        data=json.loads(map_gdf.to_json()),
        pickable=True,
        stroked=True,
        filled=True,
        extruded=False,
        get_fill_color="properties.fill_color",
        get_line_color="properties.line_color",
        get_line_width="properties.line_width",
        line_width_min_pixels=1.2,
        auto_highlight=True,
    )


def make_column_layer(map_gdf: gpd.GeoDataFrame) -> pdk.Layer:
    df = pd.DataFrame(map_gdf.drop(columns="geometry")).copy()

    vmin = float(df["predicted_kwh"].quantile(0.05))
    vmax = float(df["predicted_kwh"].quantile(0.95))

    df["fill_color"] = df.apply(
        lambda row: [255, 90, 70, 240]
        if bool(row.get("is_focus", False))
        else kwh_to_color(row["predicted_kwh"], vmin, vmax),
        axis=1,
    )

    df["elevation"] = df.apply(
        lambda row: float(row["predicted_kwh"]) * 110
        if bool(row.get("is_focus", False))
        else float(row["predicted_kwh"]) * 70,
        axis=1,
    )

    return pdk.Layer(
        "ColumnLayer",
        data=df,
        get_position=["lon", "lat"],
        get_elevation="elevation",
        elevation_scale=1,
        radius=330,
        get_fill_color="fill_color",
        pickable=True,
        auto_highlight=True,
    )


def make_deck(
    map_gdf: gpd.GeoDataFrame,
    use_3d_column: bool,
    focus_zone_id: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    zoom: Optional[float] = None,
    transition_ms: int = 0,
) -> pdk.Deck:
    layers = [make_polygon_layer(map_gdf)]

    if use_3d_column:
        layers.append(make_column_layer(map_gdf))

    focus = map_gdf[map_gdf["ID"] == focus_zone_id] if focus_zone_id else pd.DataFrame()

    if latitude is not None and longitude is not None and zoom is not None:
        view_lat = latitude
        view_lon = longitude
        view_zoom = zoom
    elif not focus.empty:
        view_lat = float(focus["lat"].iloc[0])
        view_lon = float(focus["lon"].iloc[0])
        view_zoom = 12.0
    else:
        view_lat = 37.5665
        view_lon = 126.9780
        view_zoom = 10.05

    view_kwargs = dict(
        latitude=view_lat,
        longitude=view_lon,
        zoom=view_zoom,
        pitch=42 if use_3d_column else 0,
        bearing=0,
    )

    # deck.gl 자체 transition 기능 사용
    # Python 반복 렌더링보다 훨씬 부드럽게 보임
    if transition_ms > 0:
        view_kwargs["transition_duration"] = transition_ms
        view_kwargs["transition_interpolator"] = pdk.types.String("FlyToInterpolator")

    view_state = pdk.ViewState(**view_kwargs)

    tooltip = {
        "html": """
        <div style="font-family: Inter, sans-serif;">
            <b>{생활권역표시명}</b><br/>
            <span style="color:#B7C1D3;">생활권ID:</span> {ID}<br/>
            <span style="color:#B7C1D3;">예측 충전량:</span> <b>{predicted_kwh} kWh</b><br/>
            <span style="color:#B7C1D3;">행정동:</span> {행정동명목록}
        </div>
        """,
        "style": {
            "backgroundColor": "rgba(20, 30, 45, 0.92)",
            "color": "white",
            "fontSize": "12px",
            "borderRadius": "12px",
            "padding": "10px",
        },
    }

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip=tooltip,
    )


# =========================================================
# 표시 함수
# =========================================================
def draw_alerts(top_df: pd.DataFrame, selected_time: str):
    if top_df.empty:
        st.info("수요 알림을 생성할 수 없습니다.")
        return

    alert_rows = top_df.head(4).copy()
    cards_html = ""

    for i, row in enumerate(alert_rows.itertuples(), start=1):
        label = getattr(row, "생활권역표시명", getattr(row, "생활권역ID"))
        value = float(getattr(row, "predicted_kwh"))

        if i == 1:
            badge = "최고수요"
            badge_class = "hot"
            message = f"{label} — 선택 시각 최고 수요 예상"
            guide = "충전기 가용 상태와 대기 가능성을 우선 확인하세요."
        elif i == 2:
            badge = "집중권역"
            badge_class = ""
            message = f"{label} — 충전 수요 집중 권역"
            guide = "인근 생활권과의 수요 분산 가능성을 확인하세요."
        elif i == 3:
            badge = "주의"
            badge_class = "watch"
            message = f"{label} — 운영 여유 확인 필요"
            guide = "수요 증가 가능성이 있으므로 운영 여유를 모니터링하세요."
        else:
            badge = "모니터링"
            badge_class = "monitor"
            message = f"{label} — 추가 모니터링 권장"
            guide = "피크 시간대 전후의 수요 변화를 함께 확인하세요."

        cards_html += f"""
        <div class="alert-card">
            <div class="alert-top">
                <div>
                    <div class="alert-title">{message}</div>
                    <div class="alert-meta">{guide}</div>
                    <div class="alert-value">예측 {value:,.1f} kWh · {selected_time}</div>
                </div>
                <div class="alert-badge {badge_class}">{badge}</div>
            </div>
        </div>
        """

    st.markdown(cards_html, unsafe_allow_html=True)


def draw_selected_detail_native(
    selected_label: str,
    selected_zone_id: str,
    selected_dongs: str,
    zone_pred_kwh: float,
    zone_rank: int,
    n_zones: int,
    total_day_kwh: float,
    peak_time: str,
    peak_kwh: float,
    selected_date: str,
    selected_time: str,
):
    st.markdown(
        f"""
        <div class="detail-box">
            <div class="detail-title">{selected_label}</div>
            <div class="detail-id">{selected_zone_id}</div>
            <div class="detail-label">조회 시각</div>
            <div class="detail-text">{selected_date} {selected_time}</div>
            <div class="detail-label">포함 행정동</div>
            <div class="detail-text">{selected_dongs if selected_dongs else "행정동 정보 없음"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2 = st.columns(2)
    m3, m4 = st.columns(2)

    with m1:
        st.metric("현재 예측", f"{zone_pred_kwh:.1f} kWh")

    with m2:
        st.metric("수요 순위", f"{int(zone_rank)} / {n_zones}")

    with m3:
        st.metric("일일 총량", f"{total_day_kwh:.0f} kWh")

    with m4:
        st.metric("피크 시간", peak_time, f"{peak_kwh:.1f} kWh")


def build_answer(
    selected_date: str,
    selected_time: str,
    selected_label: str,
    selected_zone_id: str,
    zone_pred_kwh: float,
    zone_rank: int,
    n_zones: int,
    peak_time: str,
    peak_kwh: float,
    total_day_kwh: float,
) -> str:
    return (
        f"{selected_date} {selected_time} 기준으로 요청하신 위치는 "
        f"{selected_label} 생활권에 포함됩니다.\n\n"
        f"생활권ID: {selected_zone_id}\n"
        f"선택 시각 예측 충전 수요: {zone_pred_kwh:.1f} kWh\n"
        f"전체 {n_zones}개 생활권 중 수요 순위: {int(zone_rank)}위\n"
        f"선택 날짜 총 예측 충전량: {total_day_kwh:.0f} kWh\n"
        f"피크 시간: {peak_time}, 피크 예측값 {peak_kwh:.1f} kWh\n\n"
        f"지도는 해당 생활권으로 확대되며, 오른쪽 패널에 상세 정보를 표시했습니다."
    )


# =========================================================
# 데이터 로딩
# =========================================================
try:
    meta = load_meta(META_JSON_PATH)
    pred, pred_col, true_col = load_predictions(PRED_CSV_PATH, meta)
    area_info = load_area_info(AREA_EXCEL_PATH)
    boundary_gdf = load_living_area_gdf(SHP_PATH, meta["zone_ids"])
except Exception as e:
    st.error("데이터 로딩 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()


# =========================================================
# 세션 상태
# =========================================================
available_dates_all = sorted(pred["date_str"].unique())

if "selected_date" not in st.session_state:
    st.session_state.selected_date = DEFAULT_DATE if DEFAULT_DATE in available_dates_all else available_dates_all[0]

available_times_default = sorted(
    pred[pred["date_str"] == st.session_state.selected_date]["time_str"].unique()
)

if "selected_time" not in st.session_state:
    st.session_state.selected_time = DEFAULT_TIME if DEFAULT_TIME in available_times_default else available_times_default[0]

zone_candidates = area_info[area_info["생활권역ID"].isin(meta["zone_ids"])].sort_values("생활권역ID")

if "selected_zone_id" not in st.session_state:
    st.session_state.selected_zone_id = zone_candidates["생활권역ID"].iloc[0]

if "use_3d_column" not in st.session_state:
    st.session_state.use_3d_column = False

if "has_query" not in st.session_state:
    st.session_state.has_query = False

if "animate_zoom" not in st.session_state:
    st.session_state.animate_zoom = False

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "안녕하세요. 저는 E-Vlog 충전수요 분석 LLM입니다.\n\n"
                "보고 싶은 날짜, 시간, 위치를 자연어로 입력하면 "
                "지도에서 해당 생활권을 확대하고 예측 충전수요를 알려드립니다.\n\n"
                "예: 2025년 11월 25일 오후 6시에 청운효자동 수요 보여줘"
            ),
        }
    ]


# =========================================================
# 현재 선택 상태 계산
# =========================================================
selected_date = st.session_state.selected_date
selected_time = st.session_state.selected_time
selected_zone_id = st.session_state.selected_zone_id

pred_filtered = pred[
    (pred["date_str"] == selected_date)
    & (pred["time_str"] == selected_time)
].copy()

if pred_filtered.empty:
    st.warning("선택한 날짜/시간에 해당하는 예측 데이터가 없습니다.")
    st.stop()

selected_dt = pred_filtered["datetime"].iloc[0]
daily_slot = int(pred_filtered["daily_slot"].iloc[0])
global_time_idx = int(pred_filtered["global_time_idx"].iloc[0])
n_zones = pred["zone_idx"].nunique()

zone_now = pred_filtered[pred_filtered["생활권역ID"] == selected_zone_id].copy()

if zone_now.empty:
    selected_zone_id = pred_filtered["생활권역ID"].iloc[0]
    st.session_state.selected_zone_id = selected_zone_id
    zone_now = pred_filtered[pred_filtered["생활권역ID"] == selected_zone_id].copy()

zone_pred_kwh = float(zone_now["predicted_kwh"].iloc[0])
zone_rank = (
    pred_filtered["predicted_kwh"]
    .rank(method="min", ascending=False)
    .loc[zone_now.index[0]]
)

selected_area = area_info[area_info["생활권역ID"] == selected_zone_id].copy()

if selected_area.empty:
    selected_label = selected_zone_id
    selected_dongs = ""
else:
    selected_label = selected_area["생활권역표시명"].iloc[0]
    selected_dongs = (
        selected_area["행정동명목록"].iloc[0]
        if "행정동명목록" in selected_area.columns
        else ""
    )

day_zone = pred[
    (pred["date_str"] == selected_date)
    & (pred["생활권역ID"] == selected_zone_id)
].copy()

if not day_zone.empty:
    peak_row = day_zone.loc[day_zone["predicted_kwh"].idxmax()]
    total_day_kwh = float(day_zone["predicted_kwh"].sum())
    peak_time = str(peak_row["time_str"])
    peak_kwh = float(peak_row["predicted_kwh"])
else:
    total_day_kwh = 0.0
    peak_time = "-"
    peak_kwh = 0.0

top10 = pred_filtered.sort_values("predicted_kwh", ascending=False).head(10)
top10 = top10.merge(
    area_info[["생활권역ID", "생활권역라벨", "생활권역표시명"]],
    on="생활권역ID",
    how="left",
)

focus_zone_id = selected_zone_id if st.session_state.has_query else None

map_gdf = prepare_map_gdf(
    boundary_gdf=boundary_gdf,
    area_info=area_info,
    pred_filtered=pred_filtered,
    focus_zone_id=focus_zone_id,
)


# =========================================================
# 메인 3분할 레이아웃
# =========================================================
chat_col, map_col, alert_col = st.columns([0.78, 1.42, 0.86], gap="small")


# =========================================================
# 1. 챗봇 LLM
# =========================================================
with chat_col:
    with st.container(border=True, height=PANEL_HEIGHT):
        panel_title(
            "챗봇 LLM",
            "자연어로 날짜, 시간, 위치를 입력하세요.",
        )

        st.markdown(
            """
            <div class="chat-guide-compact">
                <div class="status-pill">자연어 질의</div>
                <div>질문을 입력하면 날짜·시간·위치를 해석하고, 오른쪽 지도와 알림 패널을 갱신합니다.</div>
                <div class="chat-example-row">
                    <div class="chat-example-small">2025년 11월 25일 오후 6시에 청운효자동 수요 보여줘</div>
                    <div class="chat-example-small">11월 25일 18시에 마포구 성산생활권 알려줘</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_chat_messages(st.session_state.messages)

        with st.form("chat_form", clear_on_submit=True):
            user_text = st.text_input(
                "질문 입력",
                placeholder="예: 2025년 11월 25일 18시에 청운효자동 수요",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("질문하기", use_container_width=True)

        if submitted and user_text.strip():
            st.session_state.messages.append({"role": "user", "content": user_text.strip()})

            parsed = parse_user_query(
                text=user_text,
                pred=pred,
                area_info=area_info,
                fallback_date=st.session_state.selected_date,
                fallback_time=st.session_state.selected_time,
                fallback_zone_id=st.session_state.selected_zone_id,
            )

            st.session_state.selected_date = parsed["date"]
            st.session_state.selected_time = parsed["time"]
            st.session_state.selected_zone_id = parsed["zone_id"]
            st.session_state.has_query = True
            st.session_state.animate_zoom = True

            st.rerun()


# =========================================================
# 챗봇 답변 자동 생성
# =========================================================
if st.session_state.messages[-1]["role"] == "user":
    answer = build_answer(
        selected_date=selected_date,
        selected_time=selected_time,
        selected_label=selected_label,
        selected_zone_id=selected_zone_id,
        zone_pred_kwh=zone_pred_kwh,
        zone_rank=int(zone_rank),
        n_zones=n_zones,
        peak_time=peak_time,
        peak_kwh=peak_kwh,
        total_day_kwh=total_day_kwh,
    )
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()


# =========================================================
# 2. 지도
# =========================================================
with map_col:
    with st.container(border=True, height=PANEL_HEIGHT):
        panel_title(
            "충전수요지도",
            (
                f"{selected_dt:%Y-%m-%d %H:%M} · Daily Slot {daily_slot} / 47 · "
                f"전체 Time Index {global_time_idx}"
            ),
        )

        map_placeholder = st.empty()

        focus = map_gdf[map_gdf["ID"] == focus_zone_id] if focus_zone_id else pd.DataFrame()

        if (
            st.session_state.animate_zoom
            and st.session_state.has_query
            and not focus.empty
        ):
            start_lat = 37.5665
            start_lon = 126.9780
            start_zoom = 10.05

            target_lat = float(focus["lat"].iloc[0])
            target_lon = float(focus["lon"].iloc[0])
            target_zoom = 12.0

            # 1단계: 먼저 서울 전체 지도를 즉시 표시
            start_deck = make_deck(
                map_gdf=map_gdf,
                use_3d_column=st.session_state.use_3d_column,
                focus_zone_id=focus_zone_id,
                latitude=start_lat,
                longitude=start_lon,
                zoom=start_zoom,
                transition_ms=0,
            )
            map_placeholder.pydeck_chart(
                start_deck,
                use_container_width=True,
                height=MAP_HEIGHT,
            )

            # 아주 짧게 대기 후 deck.gl 자체 FlyTo 애니메이션으로 확대
            time.sleep(0.15)

            # 2단계: deck.gl transition으로 2.3초 동안 부드럽게 확대
            target_deck = make_deck(
                map_gdf=map_gdf,
                use_3d_column=st.session_state.use_3d_column,
                focus_zone_id=focus_zone_id,
                latitude=target_lat,
                longitude=target_lon,
                zoom=target_zoom,
                transition_ms=2300,
            )
            map_placeholder.pydeck_chart(
                target_deck,
                use_container_width=True,
                height=MAP_HEIGHT,
            )

            # 애니메이션이 끝난 뒤 상태 해제
            time.sleep(2.35)
            st.session_state.animate_zoom = False

        else:
            deck = make_deck(
                map_gdf=map_gdf,
                use_3d_column=st.session_state.use_3d_column,
                focus_zone_id=focus_zone_id,
            )

            map_placeholder.pydeck_chart(deck, use_container_width=True, height=MAP_HEIGHT)

        render_legend()

        use_3d = st.toggle(
            "3D 막대 표시",
            value=st.session_state.use_3d_column,
        )

        if use_3d != st.session_state.use_3d_column:
            st.session_state.use_3d_column = use_3d
            st.rerun()


# =========================================================
# 3. 수요 급증 알림 / 선택 생활권 상세
# =========================================================
with alert_col:
    with st.container(border=True, height=PANEL_HEIGHT):
        if st.session_state.has_query:
            panel_title(
                "선택 생활권 상세",
                "챗봇이 해석한 위치의 예측 결과입니다.",
            )

            draw_selected_detail_native(
                selected_label=selected_label,
                selected_zone_id=selected_zone_id,
                selected_dongs=selected_dongs,
                zone_pred_kwh=zone_pred_kwh,
                zone_rank=int(zone_rank),
                n_zones=n_zones,
                total_day_kwh=total_day_kwh,
                peak_time=peak_time,
                peak_kwh=peak_kwh,
                selected_date=selected_date,
                selected_time=selected_time,
            )

            st.markdown(
                """
                <div class="small-info">
                지도에서는 선택된 생활권을 주황색으로 강조하고, 해당 생활권 중심으로 확대 표시합니다.
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:
            panel_title(
                "수요 급증 알림",
                "기본 상태에서는 선택 시각 기준 수요가 높은 권역을 표시합니다.",
            )

            draw_alerts(top10, selected_time)

            st.markdown(
                """
                <div class="small-info">
                챗봇에 날짜, 시간, 위치를 입력하면 이 영역은 선택 생활권 상세 정보로 전환됩니다.
                </div>
                """,
                unsafe_allow_html=True,
            )


# =========================================================
# 하단 안내
# =========================================================
with st.expander("데이터 해석 안내"):
    st.markdown(
        """
        ### 서비스 동작 방식

        기본 상태에서는 서울시 전체 생활권별 충전수요지도를 보여줍니다.  
        사용자가 챗봇 LLM에 날짜, 시간, 위치를 자연어로 입력하면 다음 과정이 수행됩니다.

        ```text
        자연어 질의
        → 날짜/시간/위치 추출
        → 위치가 포함된 생활권역 탐색
        → 해당 시각의 예측 충전수요 조회
        → 지도 확대 및 생활권 강조
        → 오른쪽 패널에 선택 생활권 상세 정보 표시
        ```

        현재 자연어 해석은 규칙 기반으로 구현되어 있으며, 추후 OpenAI API 또는 별도 LLM API를 연결할 수 있습니다.

        ### 시간 인덱스

        현재 GRU 예측 CSV의 `sample_idx`는 test set 내부 인덱스입니다.

        ```text
        global_time_idx = sample_idx + val_end + look_back
        datetime = 2025-01-01 00:00 + global_time_idx × 30분
        ```

        현재 `meta.json` 기준 첫 예측 시각은 다음과 같습니다.

        ```text
        sample_idx 0 = 2025-11-25 18:00
        ```
        """
    )
