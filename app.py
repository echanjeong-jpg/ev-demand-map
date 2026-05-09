# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import streamlit.components.v1 as components

try:
    from google import genai
except Exception:
    genai = None


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

# Framer iframe 1440 × 685~730 대응
PANEL_HEIGHT = 625
MAP_HEIGHT = 485
CHAT_SCROLL_HEIGHT = 430

# 서울시 전체 생활권이 기본 화면에 더 잘 들어오도록 조정
OVERVIEW_LATITUDE = 37.5555
OVERVIEW_LONGITUDE = 126.9860
OVERVIEW_ZOOM = 9.45


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
        background:
            linear-gradient(180deg, #FFFFFF 0%, #FBFCFE 48%, #F7F9FC 100%) !important;
        box-shadow:
            inset 0 10px 24px rgba(20, 40, 70, 0.035),
            inset 0 -10px 24px rgba(20, 40, 70, 0.025);
    }

    .block-container {
        padding-top: 0.35rem;
        padding-bottom: 0.35rem;
        max-width: 1440px;
    }

    header[data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0);
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
        gap: 0.42rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF;
        border-radius: 22px;
        border: 1px solid rgba(218, 226, 236, 0.95);
        box-shadow:
            0 10px 26px rgba(25, 45, 70, 0.075),
            0 1px 0 rgba(255, 255, 255, 1) inset;
        backdrop-filter: blur(6px);
    }

    .panel-title {
        color: #172033;
        font-size: 21px;
        font-weight: 900;
        letter-spacing: -0.045em;
        margin-bottom: 2px;
        line-height: 1.18;
    }

    .panel-subtitle {
        color: #6E7B8D;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 8px;
        line-height: 1.34;
    }

    .legend-wrap {
        display: flex;
        align-items: center;
        gap: 9px;
        margin-top: 8px;
        color: #677589;
        font-weight: 850;
        font-size: 12px;
    }

    .legend-bar {
        height: 9px;
        flex: 1;
        border-radius: 999px;
        background: linear-gradient(90deg, #DDEBFA, #76A8FF, #2E6BEA, #132D6B);
        box-shadow: 0 4px 12px rgba(34, 93, 180, 0.18);
    }

    .alert-card {
        background: linear-gradient(180deg, #FFFFFF 0%, #F9FBFE 100%);
        border: 1px solid #DDE7F2;
        border-radius: 17px;
        padding: 15px 16px;
        box-shadow: 0 8px 18px rgba(24, 55, 90, 0.055);
    }

    .alert-top {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 10px;
    }

    .alert-title {
        color: #20283A;
        font-size: 13.8px;
        font-weight: 900;
        line-height: 1.35;
        margin-bottom: 6px;
    }

    .alert-meta {
        color: #6F7C8D;
        font-size: 11.7px;
        font-weight: 750;
        line-height: 1.45;
    }

    .alert-value {
        color: #172033;
        font-size: 12.2px;
        font-weight: 900;
        margin-top: 7px;
    }

    .alert-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        padding: 4px 8px;
        font-size: 10px;
        font-weight: 900;
        color: #FFFFFF;
        background: #1F6FE5;
        white-space: nowrap;
        margin-top: 1px;
        box-shadow: 0 6px 14px rgba(31, 111, 229, 0.22);
    }

    .alert-badge.hot {
        background: #E74756;
        box-shadow: 0 6px 14px rgba(231, 71, 86, 0.22);
    }

    .alert-badge.watch {
        background: #F59E0B;
        box-shadow: 0 6px 14px rgba(245, 158, 11, 0.20);
    }

    .alert-badge.monitor {
        background: #56657A;
        box-shadow: 0 6px 14px rgba(86, 101, 122, 0.18);
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #F9FBFE 100%);
        border: 1px solid #DDE7F2;
        border-radius: 16px;
        padding: 10px 12px;
        min-height: 82px;
        box-shadow: 0 8px 18px rgba(24, 55, 90, 0.055);
    }

    div[data-testid="stMetricLabel"] {
        color: #7A8797;
        font-size: 11px;
        font-weight: 850;
    }

    div[data-testid="stMetricValue"] {
        color: #172033;
        font-size: 20px;
        font-weight: 900;
        letter-spacing: -0.04em;
    }

    div[data-testid="stMetricDelta"] {
        font-size: 11px;
    }

    .detail-box {
        background: linear-gradient(180deg, #FFFFFF 0%, #F9FBFE 100%);
        border: 1px solid #DDE7F2;
        border-radius: 16px;
        padding: 13px 14px;
        margin-bottom: 9px;
        box-shadow: 0 8px 18px rgba(24, 55, 90, 0.055);
    }

    .detail-title {
        color: #172033;
        font-size: 20px;
        font-weight: 900;
        letter-spacing: -0.045em;
        margin-bottom: 4px;
    }

    .detail-id {
        color: #178554;
        font-size: 12px;
        font-weight: 900;
        margin-bottom: 8px;
    }

    .detail-label {
        color: #7A8797;
        font-size: 11px;
        font-weight: 900;
        margin-top: 8px;
        margin-bottom: 3px;
    }

    .detail-text {
        color: #4D5A6B;
        font-size: 12px;
        font-weight: 750;
        line-height: 1.42;
    }

    .small-info {
        color: #6F7C8D;
        font-size: 11.5px;
        font-weight: 750;
        line-height: 1.45;
        margin-top: 8px;
    }

    iframe {
        border-radius: 16px;
        box-shadow: 0 10px 26px rgba(22, 50, 82, 0.08);
    }

    div[data-testid="stTextInput"] {
        margin-bottom: 0rem;
    }

    div[data-testid="stTextInput"] > div {
        background: #FFFFFF !important;
        border-radius: 14px !important;
        border: 1px solid #D4E0ED !important;
        box-shadow: 0 8px 20px rgba(24, 55, 90, 0.09) !important;
    }

    div[data-testid="stTextInput"] input {
        background: #FFFFFF !important;
        color: #172033 !important;
        border-radius: 14px !important;
        min-height: 40px;
        font-size: 12px;
        font-weight: 750;
        border: none !important;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #8A96A7 !important;
        opacity: 1 !important;
        font-weight: 650;
    }

    div[data-testid="stForm"] {
        border: 0;
        padding: 0;
        background: transparent;
    }

    div[data-testid="stFormSubmitButton"] button {
        border-radius: 13px;
        min-height: 40px;
        font-weight: 900;
        background: linear-gradient(135deg, #1E63D6 0%, #2E7CF6 100%);
        color: white;
        border: 0;
        margin-top: -0.15rem;
        box-shadow: 0 8px 18px rgba(31, 111, 229, 0.26);
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background: linear-gradient(135deg, #184FAE 0%, #256CE6 100%);
        color: white;
        border: 0;
    }

    div[data-testid="stToggle"] label {
        color: #263246 !important;
        font-weight: 800 !important;
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
            width: 7px;
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
            margin-bottom: 10px;
        }}

        .chat-bubble-row.user {{
            justify-content: flex-end;
        }}

        .chat-bubble-row.assistant {{
            justify-content: flex-start;
        }}

        .chat-bubble {{
            max-width: 88%;
            border-radius: 15px;
            padding: 10px 12px;
            font-size: 12px;
            font-weight: 700;
            line-height: 1.52;
            word-break: keep-all;
            box-sizing: border-box;
        }}

        .chat-bubble.user {{
            background: #2E6BEA;
            color: #FFFFFF;
            border-bottom-right-radius: 5px;
        }}

        .chat-bubble.assistant {{
            background: linear-gradient(180deg, #FFFFFF 0%, #F9FBFE 100%);
            color: #2F3747;
            border: 1px solid #DDE7F2;
            border-bottom-left-radius: 5px;
            box-shadow: 0 6px 14px rgba(24, 55, 90, 0.05);
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


def safe_json_loads(text: str) -> dict:
    if not text:
        return {}

    raw = text.strip()

    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()

    try:
        return json.loads(raw)
    except Exception:
        pass

    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}

    return {}


# =========================================================
# Gemini API 함수
# =========================================================
def get_gemini_client() -> Optional[Any]:
    if genai is None:
        return None

    api_key = None

    try:
        api_key = st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        api_key = None

    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        return None

    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        st.session_state.last_llm_error = str(e)
        return None


def get_gemini_model() -> str:
    try:
        return st.secrets.get("GEMINI_MODEL", "gemini-1.5-flash")
    except Exception:
        return os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")


def call_gemini_text(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
) -> Optional[str]:
    client = get_gemini_client()
    if client is None:
        return None

    model = get_gemini_model()

    prompt = f"""
[System]
{system_prompt}

[User]
{user_prompt}
"""

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "temperature": temperature,
            },
        )
        return getattr(response, "text", None)
    except Exception as e:
        st.session_state.last_llm_error = str(e)
        return None


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


def llm_extract_query(
    text: str,
    pred: pd.DataFrame,
    area_info: pd.DataFrame,
) -> dict:
    """
    Gemini를 이용해 사용자 질의에서 날짜/시간/위치 표현만 추출.
    API 키가 없거나 실패하면 빈 dict 반환.
    """
    available_dates = sorted(pred["date_str"].unique())
    min_date = available_dates[0]
    max_date = available_dates[-1]

    sample_zones = area_info[
        ["생활권역ID", "생활권역표시명", "행정동명목록"]
    ].head(50).to_dict(orient="records")

    system_prompt = """
너는 전기차 충전수요 지도 서비스의 자연어 질의 해석기다.
사용자의 한국어 질의에서 날짜, 시간, 위치 표현을 추출한다.

반드시 JSON만 출력한다.
설명 문장은 출력하지 않는다.

출력 형식:
{
  "date_text": "사용자가 말한 날짜 표현 또는 null",
  "time_text": "사용자가 말한 시간 표현 또는 null",
  "location_text": "사용자가 말한 위치/동/구/생활권 표현 또는 null",
  "intent": "demand_lookup | other"
}

규칙:
- 날짜가 없으면 null.
- 시간이 없으면 null.
- 위치가 없으면 null.
- 사용자가 충전수요, 예측, 지도, 생활권, 알려줘, 보여줘 등을 말하면 demand_lookup.
- 임의로 예측값을 만들지 마라.
"""

    user_prompt = f"""
사용자 질의:
{text}

예측 데이터 날짜 범위:
{min_date} ~ {max_date}

생활권 예시:
{json.dumps(sample_zones, ensure_ascii=False)}

JSON만 출력하라.
"""

    llm_text = call_gemini_text(system_prompt, user_prompt, temperature=0.0)
    parsed = safe_json_loads(llm_text or "")

    return {
        "date_text": parsed.get("date_text"),
        "time_text": parsed.get("time_text"),
        "location_text": parsed.get("location_text"),
        "intent": parsed.get("intent", "demand_lookup"),
        "raw": llm_text,
    }


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
        raw_label = str(row.get("생활권역라벨", ""))
        dongs = str(row.get("행정동명목록", ""))
        search_text = str(row.get("search_text_clean", ""))

        score = 0
        tokens = [t for t in re.split(r"[,\s·/]+", clean_text(text)) if len(t) >= 2]

        if search_text and any(token in search_text for token in tokens):
            score += 5

        for part in re.split(r"[,·/\s]+", label):
            part_clean = clean_text(part)
            if len(part_clean) >= 2 and part_clean in q:
                score += 14

        for part in re.split(r"[,·/\s()_]+", raw_label):
            part_clean = clean_text(part)
            if len(part_clean) >= 2 and part_clean in q:
                score += 12

        for dong in re.split(r"[,·/\s]+", dongs):
            dong_clean = clean_text(dong)
            if len(dong_clean) >= 2 and dong_clean in q:
                score += 20

        if clean_text(zone_id) in q:
            score += 30

        m = re.search(r"([가-힣]+구)", text)
        if m:
            gu = clean_text(m.group(1))
            if gu in clean_text(label) or gu in clean_text(dongs) or gu in clean_text(raw_label):
                score += 8

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

    # 1차: Gemini로 날짜/시간/위치 표현 추출
    llm_result = llm_extract_query(text, pred, area_info)

    # 2차: 추출된 표현을 기존 규칙 기반 파서로 실제 데이터 범위에 맞게 변환
    date_source = llm_result.get("date_text") or text
    parsed_date = parse_date_from_text(str(date_source), available_dates)

    if parsed_date is None:
        parsed_date = parse_date_from_text(text, available_dates)

    if parsed_date is None:
        parsed_date = fallback_date

    available_times = sorted(pred[pred["date_str"] == parsed_date]["time_str"].unique())

    time_source = llm_result.get("time_text") or text
    parsed_time = parse_time_from_text(str(time_source), available_times)

    if parsed_time is None:
        parsed_time = parse_time_from_text(text, available_times)

    if parsed_time is None:
        parsed_time = fallback_time

    location_source = " ".join(
        [
            str(llm_result.get("location_text") or ""),
            text,
        ]
    )

    parsed_zone_id = find_zone_by_location(location_source, area_info) or fallback_zone_id

    return {
        "date": parsed_date,
        "time": parsed_time,
        "zone_id": parsed_zone_id,
        "llm_extract": llm_result,
    }


# =========================================================
# 지도 데이터 생성
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

    if "predicted_kwh" in gdf.columns:
        gdf["predicted_kwh"] = gdf["predicted_kwh"].astype(float).round(3)

    return gdf


def get_zone_view(gdf: gpd.GeoDataFrame, zone_id: Optional[str], use_3d_column: bool) -> dict:
    if zone_id:
        focus = gdf[gdf["ID"] == zone_id]
    else:
        focus = pd.DataFrame()

    if not focus.empty:
        return {
            "latitude": float(focus["lat"].iloc[0]),
            "longitude": float(focus["lon"].iloc[0]),
            "zoom": 12.0,
            "pitch": 42 if use_3d_column else 0,
            "bearing": 0,
        }

    return {
        "latitude": OVERVIEW_LATITUDE,
        "longitude": OVERVIEW_LONGITUDE,
        "zoom": OVERVIEW_ZOOM,
        "pitch": 40 if use_3d_column else 0,
        "bearing": 0,
    }


def prepare_map_payload(
    map_gdf: gpd.GeoDataFrame,
    use_3d_column: bool,
    focus_zone_id: Optional[str],
    previous_focus_zone_id: Optional[str],
) -> dict:
    gdf = map_gdf.copy()

    vmin = float(gdf["predicted_kwh"].quantile(0.05))
    vmax = float(gdf["predicted_kwh"].quantile(0.95))
    max_kwh = float(gdf["predicted_kwh"].max()) if len(gdf) else 1.0
    max_kwh = max(max_kwh, 1.0)

    gdf["fill_color"] = gdf.apply(
        lambda row: [255, 130, 80, 220]
        if bool(row.get("is_focus", False))
        else kwh_to_color(row["predicted_kwh"], vmin, vmax),
        axis=1,
    )

    gdf["line_color"] = gdf["is_focus"].apply(
        lambda x: [255, 60, 60, 255] if bool(x) else [255, 255, 255, 185]
    )

    gdf["line_width"] = gdf["is_focus"].apply(
        lambda x: 90 if bool(x) else 20
    )

    for col in ["zone_idx", "sample_idx", "global_time_idx", "daily_slot"]:
        if col in gdf.columns:
            gdf[col] = gdf[col].astype("Int64").astype(str)

    geojson = json.loads(gdf.to_json())

    columns_df = pd.DataFrame(gdf.drop(columns="geometry")).copy()

    columns_df["fill_color"] = columns_df.apply(
        lambda row: [255, 90, 70, 235]
        if bool(row.get("is_focus", False))
        else row["fill_color"],
        axis=1,
    )

    def calc_elevation(row) -> float:
        value = float(row["predicted_kwh"]) if pd.notna(row["predicted_kwh"]) else 0.0
        ratio = np.sqrt(np.clip(value / max_kwh, 0.0, 1.0))

        base_height = 220.0
        scaled_height = 2450.0 * ratio

        if bool(row.get("is_focus", False)):
            scaled_height += 450.0

        return float(np.clip(base_height + scaled_height, 220.0, 3100.0))

    columns_df["elevation"] = columns_df.apply(calc_elevation, axis=1)

    columns = columns_df[
        [
            "ID",
            "생활권역표시명",
            "행정동명목록",
            "predicted_kwh",
            "lon",
            "lat",
            "is_focus",
            "fill_color",
            "elevation",
        ]
    ].to_dict(orient="records")

    overview_view = {
        "latitude": OVERVIEW_LATITUDE,
        "longitude": OVERVIEW_LONGITUDE,
        "zoom": OVERVIEW_ZOOM,
        "pitch": 40 if use_3d_column else 0,
        "bearing": 0,
    }

    previous_view = get_zone_view(
        gdf=gdf,
        zone_id=previous_focus_zone_id,
        use_3d_column=use_3d_column,
    )

    target_view = get_zone_view(
        gdf=gdf,
        zone_id=focus_zone_id,
        use_3d_column=use_3d_column,
    )

    has_focus = bool(focus_zone_id and not gdf[gdf["ID"] == focus_zone_id].empty)
    has_previous_focus = bool(
        previous_focus_zone_id
        and previous_focus_zone_id != focus_zone_id
        and not gdf[gdf["ID"] == previous_focus_zone_id].empty
    )

    return {
        "geojson": geojson,
        "columns": columns,
        "overview_view": overview_view,
        "previous_view": previous_view,
        "target_view": target_view,
        "use_3d_column": bool(use_3d_column),
        "has_focus": has_focus,
        "has_previous_focus": has_previous_focus,
    }


def render_deck_map_html(payload: dict, animate: bool, height: int) -> None:
    payload_json = json.dumps(payload, ensure_ascii=False)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <script src="https://unpkg.com/deck.gl@8.9.36/dist.min.js"></script>
        <script src="https://api.tiles.mapbox.com/mapbox-gl-js/v2.15.0/mapbox-gl.js"></script>
        <link href="https://api.tiles.mapbox.com/mapbox-gl-js/v2.15.0/mapbox-gl.css" rel="stylesheet" />
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                width: 100%;
                height: {height}px;
                overflow: hidden;
                background: #FFFFFF;
                font-family: Inter, -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Noto Sans KR", sans-serif;
            }}

            #map {{
                position: relative;
                width: 100%;
                height: {height}px;
                border-radius: 16px;
                overflow: hidden;
                background: #FFFFFF;
            }}

            .deck-tooltip {{
                font-size: 12px !important;
                border-radius: 12px !important;
                padding: 10px !important;
                background: rgba(20, 30, 45, 0.92) !important;
                color: #FFFFFF !important;
            }}
        </style>
    </head>
    <body>
        <div id="map"></div>

        <script>
            const payload = {payload_json};
            const shouldAnimate = {str(animate).lower()};

            const geojsonData = payload.geojson;
            const columnData = payload.columns;
            const overviewView = payload.overview_view;
            const previousView = payload.previous_view;
            const targetView = payload.target_view;
            const use3d = payload.use_3d_column;
            const hasFocus = payload.has_focus;
            const hasPreviousFocus = payload.has_previous_focus;

            function smootherstep(t) {{
                return t * t * t * (t * (t * 6 - 15) + 10);
            }}

            function lerp(a, b, t) {{
                return a + (b - a) * t;
            }}

            function makeView(start, target, t) {{
                const e = smootherstep(t);
                return {{
                    latitude: lerp(start.latitude, target.latitude, e),
                    longitude: lerp(start.longitude, target.longitude, e),
                    zoom: lerp(start.zoom, target.zoom, e),
                    pitch: lerp(start.pitch || 0, target.pitch || 0, e),
                    bearing: lerp(start.bearing || 0, target.bearing || 0, e)
                }};
            }}

            function polygonLayer() {{
                return new deck.GeoJsonLayer({{
                    id: "living-area-polygons",
                    data: geojsonData,
                    pickable: true,
                    stroked: true,
                    filled: true,
                    extruded: false,
                    getFillColor: f => f.properties.fill_color || [220, 225, 232, 90],
                    getLineColor: f => f.properties.line_color || [255, 255, 255, 180],
                    getLineWidth: f => f.properties.line_width || 20,
                    lineWidthMinPixels: 1.2,
                    autoHighlight: true
                }});
            }}

            function columnLayer() {{
                return new deck.ColumnLayer({{
                    id: "living-area-columns",
                    data: columnData,
                    diskResolution: 32,
                    radius: 250,
                    extruded: true,
                    pickable: true,
                    getPosition: d => [d.lon, d.lat],
                    getFillColor: d => d.fill_color || [100, 140, 230, 190],
                    getElevation: d => d.elevation || 0,
                    elevationScale: 1,
                    autoHighlight: true
                }});
            }}

            function makeLayers() {{
                const base = [polygonLayer()];
                if (use3d) {{
                    base.push(columnLayer());
                }}
                return base;
            }}

            let initialView;

            if (shouldAnimate && hasFocus && hasPreviousFocus) {{
                initialView = previousView;
            }} else if (shouldAnimate && hasFocus) {{
                initialView = overviewView;
            }} else if (hasFocus) {{
                initialView = targetView;
            }} else {{
                initialView = overviewView;
            }}

            let currentView = initialView;

            const deckgl = new deck.DeckGL({{
                container: "map",
                mapStyle: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                viewState: currentView,
                controller: true,
                layers: makeLayers(),
                onViewStateChange: e => {{
                    currentView = e.viewState;
                    deckgl.setProps({{ viewState: currentView }});
                }},
                getTooltip: info => {{
                    if (!info.object) return null;

                    const p = info.object.properties || info.object;
                    const label = p["생활권역표시명"] || p.ID || "";
                    const id = p.ID || "";
                    const kwh = p.predicted_kwh || "";
                    const dongs = p["행정동명목록"] || "";

                    return {{
                        html: `
                            <div style="font-family: Inter, sans-serif;">
                                <b>${{label}}</b><br/>
                                <span style="color:#B7C1D3;">생활권ID:</span> ${{id}}<br/>
                                <span style="color:#B7C1D3;">예측 충전량:</span> <b>${{kwh}} kWh</b><br/>
                                <span style="color:#B7C1D3;">행정동:</span> ${{dongs}}
                            </div>
                        `
                    }};
                }}
            }});

            function animateBetween(startView, endView, duration) {{
                return new Promise(resolve => {{
                    const startTime = performance.now();

                    function step(now) {{
                        const raw = (now - startTime) / duration;
                        const t = Math.min(Math.max(raw, 0), 1);
                        const nextView = makeView(startView, endView, t);

                        deckgl.setProps({{
                            viewState: nextView,
                            layers: makeLayers()
                        }});

                        if (t < 1) {{
                            requestAnimationFrame(step);
                        }} else {{
                            deckgl.setProps({{
                                viewState: endView,
                                layers: makeLayers()
                            }});
                            resolve();
                        }}
                    }}

                    requestAnimationFrame(step);
                }});
            }}

            async function runAnimation() {{
                if (!(shouldAnimate && hasFocus)) return;

                await new Promise(resolve => setTimeout(resolve, 220));

                if (hasPreviousFocus) {{
                    await animateBetween(previousView, overviewView, 1050);
                    await new Promise(resolve => setTimeout(resolve, 120));
                    await animateBetween(overviewView, targetView, 1250);
                }} else {{
                    await animateBetween(overviewView, targetView, 2200);
                }}
            }}

            runAnimation();
        </script>
    </body>
    </html>
    """

    components.html(html, height=height, scrolling=False)


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
            badge_class = "focus"
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
                <div class="alert-text-wrap">
                    <div class="alert-title">{message}</div>
                    <div class="alert-meta">{guide}</div>
                    <div class="alert-value">예측 {value:,.1f} kWh · {selected_time}</div>
                </div>
                <div class="alert-badge {badge_class}">{badge}</div>
            </div>
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

        .alert-panel-inner {{
            height: 500px;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            padding: 8px 0 8px 0;
        }}

        .alert-card {{
            background: linear-gradient(180deg, #FFFFFF 0%, #F9FBFE 100%);
            border: 1px solid #DDE7F2;
            border-radius: 17px;
            padding: 15px 16px;
            box-sizing: border-box;
            box-shadow: 0 8px 18px rgba(24, 55, 90, 0.055);
        }}

        .alert-top {{
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 10px;
        }}

        .alert-text-wrap {{
            min-width: 0;
            flex: 1;
        }}

        .alert-title {{
            color: #20283A;
            font-size: 13.8px;
            font-weight: 900;
            line-height: 1.35;
            margin-bottom: 6px;
            word-break: keep-all;
        }}

        .alert-meta {{
            color: #6F7C8D;
            font-size: 11.7px;
            font-weight: 750;
            line-height: 1.45;
            word-break: keep-all;
        }}

        .alert-value {{
            color: #172033;
            font-size: 12.2px;
            font-weight: 900;
            margin-top: 7px;
        }}

        .alert-badge {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            padding: 4px 8px;
            font-size: 10px;
            font-weight: 900;
            color: #FFFFFF;
            background: #1F6FE5;
            white-space: nowrap;
            margin-top: 1px;
            flex-shrink: 0;
            box-shadow: 0 6px 14px rgba(31, 111, 229, 0.22);
        }}

        .alert-badge.hot {{
            background: #E74756;
            box-shadow: 0 6px 14px rgba(231, 71, 86, 0.22);
        }}

        .alert-badge.focus {{
            background: #1F6FE5;
            box-shadow: 0 6px 14px rgba(31, 111, 229, 0.22);
        }}

        .alert-badge.watch {{
            background: #F59E0B;
            box-shadow: 0 6px 14px rgba(245, 158, 11, 0.20);
        }}

        .alert-badge.monitor {{
            background: #56657A;
            box-shadow: 0 6px 14px rgba(86, 101, 122, 0.18);
        }}
    </style>
    </head>
    <body>
        <div class="alert-panel-inner">
            {cards_html}
        </div>
    </body>
    </html>
    """

    components.html(html, height=508, scrolling=False)


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


def build_fallback_answer(
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
        f"지도는 해당 생활권으로 확대되며, 왼쪽 패널에 상세 정보를 표시했습니다."
    )


def build_llm_answer(
    user_text: str,
    selected_date: str,
    selected_time: str,
    selected_label: str,
    selected_zone_id: str,
    selected_dongs: str,
    zone_pred_kwh: float,
    zone_rank: int,
    n_zones: int,
    peak_time: str,
    peak_kwh: float,
    total_day_kwh: float,
    top10: pd.DataFrame,
) -> str:
    top_items = []
    for i, row in enumerate(top10.head(5).itertuples(), start=1):
        top_items.append(
            {
                "rank": i,
                "zone": getattr(row, "생활권역표시명", getattr(row, "생활권역ID")),
                "predicted_kwh": round(float(getattr(row, "predicted_kwh")), 1),
            }
        )

    facts = {
        "user_query": user_text,
        "date": selected_date,
        "time": selected_time,
        "zone_label": selected_label,
        "zone_id": selected_zone_id,
        "included_dongs": selected_dongs,
        "predicted_kwh": round(zone_pred_kwh, 1),
        "rank": int(zone_rank),
        "num_zones": int(n_zones),
        "total_day_kwh": round(total_day_kwh, 1),
        "peak_time": peak_time,
        "peak_kwh": round(peak_kwh, 1),
        "top5_at_selected_time": top_items,
    }

    system_prompt = """
너는 E-Vlog 서비스의 전기차 충전수요 분석 챗봇 '모도리'다.
반드시 제공된 facts 데이터만 근거로 답변한다.
없는 정보는 추측하지 않는다.

답변 스타일:
- 한국어
- 사용자가 물어본 날짜, 시간, 위치를 먼저 확인
- 예측 충전수요, 순위, 일일 총량, 피크시간을 자연스럽게 설명
- 운영 관점에서 한 문장 조언 포함
- 너무 길지 않게 5~8문장 정도
- 수치는 facts에 있는 값 그대로 사용
"""

    user_prompt = f"""
사용자 질문:
{user_text}

facts:
{json.dumps(facts, ensure_ascii=False, indent=2)}

위 facts만 사용해서 자연어로 답변하라.
"""

    llm_answer = call_gemini_text(system_prompt, user_prompt, temperature=0.25)

    if llm_answer:
        return llm_answer.strip()

    return build_fallback_answer(
        selected_date=selected_date,
        selected_time=selected_time,
        selected_label=selected_label,
        selected_zone_id=selected_zone_id,
        zone_pred_kwh=zone_pred_kwh,
        zone_rank=zone_rank,
        n_zones=n_zones,
        peak_time=peak_time,
        peak_kwh=peak_kwh,
        total_day_kwh=total_day_kwh,
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
    st.session_state.selected_date = (
        DEFAULT_DATE if DEFAULT_DATE in available_dates_all else available_dates_all[0]
    )

available_times_default = sorted(
    pred[pred["date_str"] == st.session_state.selected_date]["time_str"].unique()
)

if "selected_time" not in st.session_state:
    st.session_state.selected_time = (
        DEFAULT_TIME if DEFAULT_TIME in available_times_default else available_times_default[0]
    )

zone_candidates = area_info[
    area_info["생활권역ID"].isin(meta["zone_ids"])
].sort_values("생활권역ID")

if "selected_zone_id" not in st.session_state:
    st.session_state.selected_zone_id = zone_candidates["생활권역ID"].iloc[0]

if "previous_focus_zone_id" not in st.session_state:
    st.session_state.previous_focus_zone_id = None

if "use_3d_column" not in st.session_state:
    st.session_state.use_3d_column = True

if "has_query" not in st.session_state:
    st.session_state.has_query = False

if "animate_zoom" not in st.session_state:
    st.session_state.animate_zoom = False

if "pending_user_query" not in st.session_state:
    st.session_state.pending_user_query = None

if "last_llm_error" not in st.session_state:
    st.session_state.last_llm_error = None

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "안녕하세요. 저는 모도리입니다.\n\n"
                "보고 싶은 날짜, 시간, 위치를 자연어로 입력하면 "
                "예측 결과 파일을 조회해 해당 생활권의 충전수요를 알려드립니다.\n\n"
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
previous_focus_zone_id = st.session_state.previous_focus_zone_id

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
# LLM 답변 생성
# =========================================================
if st.session_state.pending_user_query:
    answer = build_llm_answer(
        user_text=st.session_state.pending_user_query,
        selected_date=selected_date,
        selected_time=selected_time,
        selected_label=selected_label,
        selected_zone_id=selected_zone_id,
        selected_dongs=selected_dongs,
        zone_pred_kwh=zone_pred_kwh,
        zone_rank=int(zone_rank),
        n_zones=n_zones,
        peak_time=peak_time,
        peak_kwh=peak_kwh,
        total_day_kwh=total_day_kwh,
        top10=top10,
    )
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.pending_user_query = None


# =========================================================
# 메인 3분할 레이아웃
# 왼쪽: 알림/상세, 가운데: 지도, 오른쪽: 모도리
# =========================================================
alert_col, map_col, chat_col = st.columns([0.86, 1.42, 0.78], gap="small")


# =========================================================
# 1. 왼쪽: 수요 급증 알림 / 선택 생활권 상세
# =========================================================
with alert_col:
    with st.container(border=True, height=PANEL_HEIGHT):
        if st.session_state.has_query:
            panel_title(
                "선택 생활권 상세",
                "모도리가 해석한 위치의 예측 결과입니다.",
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


# =========================================================
# 2. 가운데: 지도
# =========================================================
with map_col:
    with st.container(border=True, height=PANEL_HEIGHT):
        header_left, header_right = st.columns([0.72, 0.28], gap="small")

        with header_left:
            panel_title(
                "충전수요지도",
                (
                    f"{selected_dt:%Y-%m-%d %H:%M} · Daily Slot {daily_slot} / 47 · "
                    f"전체 Time Index {global_time_idx}"
                ),
            )

        with header_right:
            use_2d = st.toggle(
                "2D 지도 보기",
                value=not st.session_state.use_3d_column,
                key="map_2d_toggle",
            )

            next_use_3d = not use_2d

            if next_use_3d != st.session_state.use_3d_column:
                st.session_state.use_3d_column = next_use_3d
                st.rerun()

        map_payload = prepare_map_payload(
            map_gdf=map_gdf,
            use_3d_column=st.session_state.use_3d_column,
            focus_zone_id=focus_zone_id,
            previous_focus_zone_id=previous_focus_zone_id,
        )

        should_animate = bool(st.session_state.animate_zoom and st.session_state.has_query)

        render_deck_map_html(
            payload=map_payload,
            animate=should_animate,
            height=MAP_HEIGHT,
        )

        if st.session_state.animate_zoom:
            st.session_state.animate_zoom = False

        render_legend()


# =========================================================
# 3. 오른쪽: 모도리
# =========================================================
with chat_col:
    with st.container(border=True, height=PANEL_HEIGHT):
        panel_title(
            "모도리",
            "자연어로 날짜, 시간, 위치를 입력하세요.",
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
            clean_user_text = user_text.strip()
            st.session_state.messages.append({"role": "user", "content": clean_user_text})

            parsed = parse_user_query(
                text=clean_user_text,
                pred=pred,
                area_info=area_info,
                fallback_date=st.session_state.selected_date,
                fallback_time=st.session_state.selected_time,
                fallback_zone_id=st.session_state.selected_zone_id,
            )

            if st.session_state.has_query:
                st.session_state.previous_focus_zone_id = st.session_state.selected_zone_id
            else:
                st.session_state.previous_focus_zone_id = None

            st.session_state.selected_date = parsed["date"]
            st.session_state.selected_time = parsed["time"]
            st.session_state.selected_zone_id = parsed["zone_id"]
            st.session_state.has_query = True
            st.session_state.animate_zoom = True
            st.session_state.pending_user_query = clean_user_text

            st.rerun()
