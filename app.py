# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import plotly.graph_objects as go
import streamlit as st


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

MAP_HEIGHT = 590
TOTAL_CHART_HEIGHT = 270
DETAIL_CHART_HEIGHT = 270


# =========================================================
# Streamlit 기본 설정
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
        padding-top: 1.4rem;
        padding-bottom: 2.0rem;
        max-width: 1660px;
    }

    header[data-testid="stHeader"] {
        background: rgba(238, 243, 248, 0.0);
        height: 0rem;
    }

    div[data-testid="stToolbar"] {
        visibility: hidden;
        height: 0%;
        position: fixed;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    section[data-testid="stSidebar"] {
        display: none;
    }

    div[data-testid="stVerticalBlock"] {
        gap: 0.72rem;
    }

    div[data-testid="column"] {
        gap: 0.4rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF;
        border-radius: 22px;
        box-shadow: 0 8px 28px rgba(35, 55, 80, 0.065);
        border: 1px solid rgba(220, 228, 238, 0.9);
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid rgba(220, 228, 238, 0.9);
        padding: 16px 18px;
        border-radius: 18px;
        box-shadow: 0 8px 24px rgba(35, 55, 80, 0.045);
        min-height: 92px;
    }

    div[data-testid="stMetricLabel"] {
        color: #8A93A3;
        font-size: 13px;
        font-weight: 800;
    }

    div[data-testid="stMetricValue"] {
        color: #232633;
        font-size: 25px;
        font-weight: 900;
        letter-spacing: -0.04em;
    }

    div[data-testid="stMetricDelta"] {
        font-weight: 800;
    }

    div[data-testid="stSelectbox"] label {
        font-weight: 800;
        color: #5A6270;
        font-size: 13px;
        margin-bottom: 0.25rem;
    }

    div[data-baseweb="select"] > div {
        background: #F8FAFD;
        border-radius: 14px;
        border-color: #E6ECF3;
        min-height: 46px;
        box-shadow: 0 4px 16px rgba(35, 55, 80, 0.035);
    }

    div[data-testid="stToggle"] {
        background: #F8FAFD;
        border: 1px solid #E6ECF3;
        border-radius: 14px;
        padding: 8px 14px 6px 14px;
        min-height: 46px;
        box-shadow: 0 4px 16px rgba(35, 55, 80, 0.035);
        display: flex;
        align-items: center;
    }

    .app-logo-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 0.9rem;
    }

    .app-logo-box {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: #FFFFFF;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 900;
        color: #2E6BEA;
        box-shadow: 0 6px 18px rgba(35, 55, 80, 0.08);
        border: 1px solid #DDE6F2;
    }

    .app-title {
        font-size: 26px;
        font-weight: 900;
        letter-spacing: -0.05em;
        color: #202532;
    }

    .app-subtitle {
        color: #7C8594;
        font-size: 13px;
        font-weight: 700;
        margin-top: -4px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 900;
        color: #222633;
        letter-spacing: -0.04em;
        margin: 0 0 4px 0;
    }

    .section-subtitle {
        color: #7C8594;
        font-size: 13px;
        font-weight: 650;
        margin: 0 0 12px 0;
    }

    .card-title {
        font-size: 19px;
        font-weight: 900;
        color: #222633;
        letter-spacing: -0.035em;
        margin-bottom: 0.45rem;
    }

    .card-subtitle {
        color: #7C8594;
        font-size: 13px;
        font-weight: 650;
        margin-bottom: 0.7rem;
    }

    .chat-guide {
        background: #F8FAFD;
        border: 1px solid #E3EAF3;
        border-radius: 16px;
        padding: 14px 15px;
        color: #5E6878;
        font-size: 13px;
        line-height: 1.55;
        font-weight: 650;
        margin-bottom: 10px;
    }

    .chat-example {
        background: #FFFFFF;
        border: 1px solid #E6EDF6;
        border-radius: 13px;
        padding: 10px 12px;
        margin-top: 8px;
        color: #2F3747;
        font-size: 12px;
        font-weight: 800;
    }

    .status-pill {
        display: inline-flex;
        border-radius: 999px;
        padding: 5px 10px;
        background: #EAF2FF;
        color: #1F6FE5;
        font-size: 12px;
        font-weight: 900;
        margin-bottom: 8px;
    }

    .focus-card {
        background: #F8FAFD;
        border-radius: 18px;
        padding: 16px 16px;
        border: 1px solid #E8EEF5;
        margin-top: 8px;
    }

    .focus-title {
        color: #222633;
        font-size: 18px;
        font-weight: 900;
        line-height: 1.35;
        margin-bottom: 4px;
    }

    .focus-id {
        color: #2E8B55;
        font-size: 12px;
        font-weight: 900;
        margin-bottom: 12px;
    }

    .focus-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
    }

    .focus-item {
        background: #FFFFFF;
        border-radius: 15px;
        padding: 12px 13px;
        border: 1px solid #E8EEF5;
    }

    .focus-label {
        color: #8A93A3;
        font-size: 12px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .focus-value {
        color: #222633;
        font-size: 22px;
        font-weight: 900;
        letter-spacing: -0.04em;
    }

    .legend-wrap {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 8px;
        margin-bottom: 0px;
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

    .rank-row-custom {
        display: grid;
        grid-template-columns: 22px minmax(120px, 1fr) 120px 52px;
        gap: 9px;
        align-items: center;
        margin: 7px 0;
    }

    .rank-num {
        color: #202532;
        font-weight: 900;
        font-size: 12px;
    }

    .rank-label {
        font-weight: 850;
        color: #303644;
        font-size: 12px;
        line-height: 1.25;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .rank-track {
        height: 8px;
        background: #E4ECF6;
        border-radius: 999px;
        overflow: hidden;
    }

    .rank-fill {
        height: 100%;
        background: linear-gradient(90deg, #67A8FF, #1F6FE5);
        border-radius: 999px;
    }

    .rank-value {
        font-weight: 900;
        color: #303644;
        font-size: 12px;
        text-align: right;
        line-height: 1.25;
    }

    .alert-card {
        background: #F8FAFD;
        border: 1px solid #E3EAF3;
        border-radius: 16px;
        padding: 12px 14px;
        margin-bottom: 9px;
    }

    .alert-top {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 12px;
    }

    .alert-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        padding: 4px 9px;
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

    .alert-title {
        color: #222633;
        font-size: 13px;
        font-weight: 900;
        line-height: 1.35;
        margin-bottom: 5px;
    }

    .alert-meta {
        color: #7C8594;
        font-size: 12px;
        font-weight: 700;
        line-height: 1.45;
    }

    .alert-value {
        color: #222633;
        font-size: 12px;
        font-weight: 900;
        margin-top: 6px;
    }

    .stPlotlyChart {
        border-radius: 18px;
        overflow: hidden;
        background: white;
    }

    iframe {
        border-radius: 18px;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
    }

    hr {
        margin-top: 0.55rem;
        margin-bottom: 0.55rem;
    }

    .mapboxgl-control-container {
        opacity: 0.92;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 화면 보조 함수
# =========================================================
def section_title(title: str, subtitle: str | None = None) -> None:
    if subtitle:
        st.markdown(
            f"""
            <div class="section-title">{title}</div>
            <div class="section-subtitle">{subtitle}</div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="section-title">{title}</div>
            """,
            unsafe_allow_html=True,
        )


def card_title(title: str, subtitle: str | None = None) -> None:
    if subtitle:
        st.markdown(
            f"""
            <div class="card-title">{title}</div>
            <div class="card-subtitle">{subtitle}</div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="card-title">{title}</div>
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


# =========================================================
# 유틸 함수
# =========================================================
def normalize_id(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def clean_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value).strip().lower())


def get_prediction_column(df: pd.DataFrame) -> str:
    candidates = ["y_pred_kwh", "predicted_kwh", "pred_kwh", "prediction_kwh"]
    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "예측값 컬럼을 찾을 수 없습니다. CSV에 y_pred_kwh 또는 predicted_kwh 컬럼이 필요합니다."
    )


def get_true_column(df: pd.DataFrame) -> str | None:
    candidates = ["y_true_kwh", "true_kwh", "actual_kwh"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def clean_label(label: str) -> str:
    if not isinstance(label, str):
        return ""
    return (
        label.replace("지역생활권(", "")
        .replace("권역생활권(", "")
        .replace(")", "")
        .replace("_", " · ")
    )


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

    pred["date"] = pred["datetime"].dt.date
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
        f"meta.json을 다음 인코딩으로 모두 읽지 못했습니다: {encodings}. 마지막 오류: {last_error}",
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

    area["search_text"] = (
        area["생활권역ID"].astype(str)
        + " "
        + area["생활권역표시명"].astype(str)
        + " "
        + area.get("생활권역라벨", "").astype(str)
        + " "
        + area.get("행정동명목록", "").astype(str)
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
    text = text.strip()

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
    text = text.strip()

    m = re.search(r"(\d{1,2})\s*:\s*(\d{1,2})", text)
    if m:
        h, mi = map(int, m.groups())
        mi = 30 if mi >= 15 and mi < 45 else 0
        if mi >= 45:
            h += 1
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
        half = m.group(2)
        mi = 30 if half else 0

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

    # 생활권ID 직접 입력 대응
    id_candidates = re.findall(r"생활권경계[_\s-]*\d{1,3}", text)
    for cand in id_candidates:
        cand_norm = cand.replace(" ", "").replace("-", "_")
        if "_" not in cand_norm:
            cand_norm = cand_norm.replace("생활권경계", "생활권경계_")
        match = area_info[area_info["생활권역ID"] == cand_norm]
        if not match.empty:
            return match["생활권역ID"].iloc[0]

    # 행정동/구/생활권명 기반 검색
    best_zone = None
    best_score = 0

    for _, row in area_info.iterrows():
        zone_id = row["생활권역ID"]
        label = str(row.get("생활권역표시명", ""))
        dongs = str(row.get("행정동명목록", ""))
        search_text = str(row.get("search_text_clean", ""))

        score = 0

        # 전체 search_text가 질의에 포함되거나, 질의가 search_text에 포함
        if search_text and any(token in search_text for token in re.split(r"[,\s·/]+", clean_text(text)) if len(token) >= 2):
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

        # 구 단위 키워드 대응
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
# 지도/차트 생성
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
                "datetime",
                "date_str",
                "time_str",
                "daily_slot",
                "predicted_kwh",
            ]
        ],
        on="생활권역ID",
        how="left",
    )

    if "생활권역라벨" not in gdf.columns:
        gdf["생활권역라벨"] = gdf["ID"]

    if "생활권역표시명" not in gdf.columns:
        gdf["생활권역표시명"] = gdf["생활권역라벨"].apply(clean_label)

    if "행정동명목록" not in gdf.columns:
        gdf["행정동명목록"] = ""

    gdf["is_focus"] = gdf["ID"] == focus_zone_id

    if "datetime" in gdf.columns:
        gdf["datetime"] = gdf["datetime"].astype(str)

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
) -> pdk.Deck:
    layers = [make_polygon_layer(map_gdf)]

    if use_3d_column:
        layers.append(make_column_layer(map_gdf))

    if focus_zone_id:
        focus = map_gdf[map_gdf["ID"] == focus_zone_id]
    else:
        focus = pd.DataFrame()

    if not focus.empty:
        latitude = float(focus["lat"].iloc[0])
        longitude = float(focus["lon"].iloc[0])
        zoom = 12.0
    else:
        latitude = 37.5665
        longitude = 126.9780
        zoom = 10.05

    view_state = pdk.ViewState(
        latitude=latitude,
        longitude=longitude,
        zoom=zoom,
        pitch=42 if use_3d_column else 0,
        bearing=0,
    )

    tooltip = {
        "html": """
        <div style="font-family: Inter, sans-serif;">
            <b>{생활권역표시명}</b><br/>
            <span style="color:#B7C1D3;">생활권ID:</span> {ID}<br/>
            <span style="color:#B7C1D3;">zone_idx:</span> {zone_idx}<br/>
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


def make_total_demand_chart(pred: pd.DataFrame, selected_date: str, selected_time: str):
    day_df = pred[pred["date_str"] == selected_date].copy()

    if day_df.empty:
        return None

    agg = (
        day_df.groupby("time_str", as_index=False)
        .agg(total_kwh=("predicted_kwh", "sum"))
        .sort_values("time_str")
    )

    colors = ["#2E5BEA" if t != selected_time else "#E84B5F" for t in agg["time_str"]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=agg["time_str"],
            y=agg["total_kwh"],
            mode="lines+markers",
            line=dict(color="#2E5BEA", width=3),
            marker=dict(size=6, color=colors),
            name="총 예측 충전량",
        )
    )

    fig.update_layout(
        height=TOTAL_CHART_HEIGHT,
        margin=dict(l=4, r=4, t=10, b=8),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#606775", size=12),
        xaxis=dict(
            title=None,
            showgrid=True,
            gridcolor="#EEF1F5",
            tickmode="array",
            tickvals=agg["time_str"][::4],
        ),
        yaxis=dict(
            title=None,
            showgrid=True,
            gridcolor="#EEF1F5",
        ),
        showlegend=False,
    )

    return fig


def make_daily_pattern_chart(pred: pd.DataFrame, selected_date: str, selected_zone_id: str):
    day_df = pred[
        (pred["date_str"] == selected_date)
        & (pred["생활권역ID"] == selected_zone_id)
    ].copy()

    if day_df.empty:
        return None

    day_df = day_df.sort_values("datetime")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=day_df["time_str"],
            y=day_df["predicted_kwh"],
            mode="lines+markers",
            line=dict(color="#2E5BEA", width=3),
            marker=dict(size=5, color="#2E5BEA"),
            name="예측 충전량",
        )
    )

    fig.update_layout(
        height=DETAIL_CHART_HEIGHT,
        margin=dict(l=4, r=4, t=10, b=8),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#606775", size=12),
        xaxis=dict(title=None, showgrid=True, gridcolor="#EEF1F5"),
        yaxis=dict(title=None, showgrid=True, gridcolor="#EEF1F5"),
        showlegend=False,
    )

    return fig


# =========================================================
# 렌더링 함수
# =========================================================
def draw_ranking(top_df: pd.DataFrame):
    if top_df.empty:
        st.info("표시할 랭킹 데이터가 없습니다.")
        return

    vmax = float(top_df["predicted_kwh"].max())
    vmin = float(top_df["predicted_kwh"].min())
    denom = max(vmax - vmin, 1e-9)

    rows_html = ""

    for i, row in enumerate(top_df.itertuples(), start=1):
        label = getattr(row, "생활권역표시명", getattr(row, "생활권역ID"))
        value = float(getattr(row, "predicted_kwh"))

        normalized = (value - vmin) / denom
        pct = 18 + normalized * 82
        pct = float(np.clip(pct, 8, 100))

        rows_html += f"""
        <div class="rank-row-custom">
            <div class="rank-num">{i}</div>
            <div class="rank-label">{label}</div>
            <div class="rank-track">
                <div class="rank-fill" style="width:{pct:.1f}%;"></div>
            </div>
            <div class="rank-value">{value:,.1f}</div>
        </div>
        """

    st.markdown(rows_html, unsafe_allow_html=True)


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


def draw_focus_detail(
    selected_label: str,
    selected_zone_id: str,
    selected_dongs: str,
    zone_pred_kwh: float,
    zone_rank: int,
    n_zones: int,
    total_day_kwh: float,
    peak_time: str,
    peak_kwh: float,
):
    st.markdown(
        f"""
        <div class="focus-card">
            <div class="focus-title">{selected_label}</div>
            <div class="focus-id">{selected_zone_id}</div>
            <div class="focus-label">포함 행정동</div>
            <div style="color:#626C7C;font-size:13px;font-weight:700;line-height:1.5;margin-bottom:14px;">
                {selected_dongs if selected_dongs else "행정동 정보 없음"}
            </div>
            <div class="focus-grid">
                <div class="focus-item">
                    <div class="focus-label">현재 예측</div>
                    <div class="focus-value">{zone_pred_kwh:.1f} kWh</div>
                </div>
                <div class="focus-item">
                    <div class="focus-label">수요 순위</div>
                    <div class="focus-value">{int(zone_rank)} / {n_zones}</div>
                </div>
                <div class="focus-item">
                    <div class="focus-label">일일 총량</div>
                    <div class="focus-value">{total_day_kwh:.0f} kWh</div>
                </div>
                <div class="focus-item">
                    <div class="focus-label">피크 시간</div>
                    <div class="focus-value">{peak_time}</div>
                    <div style="color:#248A3D;font-weight:900;font-size:13px;">↑ {peak_kwh:.1f} kWh</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
        f"{selected_date} {selected_time} 기준, 요청하신 위치는 "
        f"**{selected_label}** 생활권에 포함됩니다.\n\n"
        f"- 생활권ID: `{selected_zone_id}`\n"
        f"- 선택 시각 예측 충전 수요: **{zone_pred_kwh:.1f} kWh**\n"
        f"- 전체 {n_zones}개 생활권 중 수요 순위: **{int(zone_rank)}위**\n"
        f"- 선택 날짜의 해당 생활권 총 예측 충전량: **{total_day_kwh:.0f} kWh**\n"
        f"- 선택 날짜 피크 시간: **{peak_time}**, 피크 예측값 **{peak_kwh:.1f} kWh**\n\n"
        f"오른쪽 지도에서 해당 생활권을 확대하고 강조 표시했습니다."
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
# 세션 상태 초기화
# =========================================================
available_dates_all = sorted(pred["date_str"].unique())

if "selected_date" not in st.session_state:
    st.session_state.selected_date = DEFAULT_DATE if DEFAULT_DATE in available_dates_all else available_dates_all[0]

available_times_default = sorted(
    pred[pred["date_str"] == st.session_state.selected_date]["time_str"].unique()
)

if "selected_time" not in st.session_state:
    st.session_state.selected_time = DEFAULT_TIME if DEFAULT_TIME in available_times_default else available_times_default[0]

zone_label_map_init = area_info[area_info["생활권역ID"].isin(meta["zone_ids"])].copy()
zone_label_map_init = zone_label_map_init.sort_values("생활권역ID")

if "selected_zone_id" not in st.session_state:
    st.session_state.selected_zone_id = zone_label_map_init["생활권역ID"].iloc[0]

if "use_3d_column" not in st.session_state:
    st.session_state.use_3d_column = False

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "안녕하세요. 저는 E-Vlog 충전수요 분석 어시스턴트입니다.\n\n"
                "보고 싶은 **날짜, 시간, 위치**를 자연어로 입력하면 오른쪽 지도에서 해당 생활권을 확대하고 "
                "예측 충전수요를 보여드릴게요.\n\n"
                "예: `2025년 11월 25일 오후 6시에 청운효자동 충전 수요 보여줘`"
            ),
        }
    ]


# =========================================================
# 상단 로고
# =========================================================
st.markdown(
    """
    <div class="app-logo-row">
        <div class="app-logo-box">EV</div>
        <div>
            <div class="app-title">E-Vlog</div>
            <div class="app-subtitle">LLM 기반 생활권별 전기차 충전 수요 예측 서비스</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 메인 레이아웃: 좌측 챗봇 / 우측 지도
# =========================================================
chat_col, service_col = st.columns([0.72, 1.65], gap="small")


# =========================================================
# 좌측: 챗봇 LLM 패널
# =========================================================
with chat_col:
    with st.container(border=True):
        section_title(
            "챗봇형 수요 분석 LLM",
            "날짜, 시간, 위치를 자연어로 입력하면 예측 결과를 지도와 함께 확인합니다.",
        )

        st.markdown(
            """
            <div class="chat-guide">
                <div class="status-pill">자연어 질의 지원</div><br/>
                예측값은 GRU/ST-LLM 결과 파일에서 조회하고, LLM 인터페이스는 사용자의 질문을 
                날짜·시간·생활권 조건으로 변환하는 역할을 합니다.
                <div class="chat-example">예: 2025년 11월 25일 오후 6시에 청운효자동 수요 보여줘</div>
                <div class="chat-example">예: 11월 25일 18시에 마포구 성산생활권 알려줘</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_text = st.chat_input("보고 싶은 날짜, 시간, 위치를 입력하세요")

        if user_text:
            st.session_state.messages.append({"role": "user", "content": user_text})

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

            st.rerun()


# =========================================================
# 우측: 선택 상태 계산
# =========================================================
selected_date = st.session_state.selected_date
selected_time = st.session_state.selected_time
selected_zone_id = st.session_state.selected_zone_id
use_3d_column = st.session_state.use_3d_column

available_dates = sorted(pred["date_str"].unique())
available_times = sorted(pred[pred["date_str"] == selected_date]["time_str"].unique())

pred_filtered = pred[
    (pred["date_str"] == selected_date)
    & (pred["time_str"] == selected_time)
].copy()

if pred_filtered.empty:
    st.warning("선택한 날짜/시간에 해당하는 예측 데이터가 없습니다.")
    st.stop()

map_gdf = prepare_map_gdf(
    boundary_gdf=boundary_gdf,
    area_info=area_info,
    pred_filtered=pred_filtered,
    focus_zone_id=selected_zone_id,
)

selected_dt = pred_filtered["datetime"].iloc[0]
daily_slot = int(pred_filtered["daily_slot"].iloc[0])
global_time_idx = int(pred_filtered["global_time_idx"].iloc[0])
n_zones = pred["zone_idx"].nunique()

total_kwh = pred_filtered["predicted_kwh"].sum()
mean_kwh = pred_filtered["predicted_kwh"].mean()
q75 = pred_filtered["predicted_kwh"].quantile(0.75)
high_count = int((pred_filtered["predicted_kwh"] >= q75).sum())

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

# 챗봇 마지막 답변 자동 추가
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
# 우측: 서비스 패널
# =========================================================
with service_col:
    with st.container(border=True):
        section_title(
            "예측 조건 설정",
            "챗봇 질의 결과가 자동 반영됩니다. 직접 선택하여 탐색할 수도 있습니다.",
        )

        zone_label_map = area_info.copy()
        zone_label_map = zone_label_map[
            zone_label_map["생활권역ID"].isin(meta["zone_ids"])
        ].copy()
        zone_label_map["display"] = (
            zone_label_map["생활권역ID"] + " · " + zone_label_map["생활권역표시명"]
        )
        zone_label_map = zone_label_map.sort_values("생활권역ID")

        date_index = available_dates.index(selected_date) if selected_date in available_dates else 0
        time_index = available_times.index(selected_time) if selected_time in available_times else 0

        selected_zone_display = zone_label_map.loc[
            zone_label_map["생활권역ID"] == selected_zone_id,
            "display",
        ]

        if selected_zone_display.empty:
            zone_index = 0
        else:
            zone_index = zone_label_map["display"].tolist().index(selected_zone_display.iloc[0])

        c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.0, 3.3])

        with c1:
            new_date = st.selectbox(
                "날짜 선택",
                available_dates,
                index=date_index,
                key="ui_date",
            )

        new_available_times = sorted(pred[pred["date_str"] == new_date]["time_str"].unique())

        if selected_time in new_available_times:
            new_time_index = new_available_times.index(selected_time)
        else:
            new_time_index = 0

        with c2:
            new_time = st.selectbox(
                "시간 선택",
                new_available_times,
                index=new_time_index,
                key="ui_time",
            )

        with c3:
            st.markdown('<div class="toggle-label">3D 막대 표시</div>', unsafe_allow_html=True)
            new_3d = st.toggle(
                "3D 막대 표시",
                value=use_3d_column,
                label_visibility="collapsed",
                key="ui_3d",
            )

        with c4:
            new_zone_display = st.selectbox(
                "상세 조회 생활권",
                zone_label_map["display"].tolist(),
                index=zone_index,
                key="ui_zone",
            )

        new_zone_id = zone_label_map.loc[
            zone_label_map["display"] == new_zone_display,
            "생활권역ID",
        ].iloc[0]

        if (
            new_date != st.session_state.selected_date
            or new_time != st.session_state.selected_time
            or new_zone_id != st.session_state.selected_zone_id
            or new_3d != st.session_state.use_3d_column
        ):
            st.session_state.selected_date = new_date
            st.session_state.selected_time = new_time
            st.session_state.selected_zone_id = new_zone_id
            st.session_state.use_3d_column = new_3d
            st.rerun()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("예측 대상 권역", f"{n_zones} 개")
    kpi2.metric("선택 시각 총 예측 충전량", f"{total_kwh:,.0f} kWh")
    kpi3.metric("고수요 예상 권역", f"{high_count} 개")
    kpi4.metric("선택 생활권 예측 수요", f"{zone_pred_kwh:.1f} kWh")

    map_col, side_col = st.columns([1.55, 0.9], gap="small")

    with map_col:
        with st.container(border=True):
            section_title(
                "생활권별 충전 수요 지도",
                f"{selected_dt:%Y-%m-%d %H:%M} · Daily Slot {daily_slot} / 47 · 전체 Time Index {global_time_idx}",
            )

            deck = make_deck(
                map_gdf=map_gdf,
                use_3d_column=st.session_state.use_3d_column,
                focus_zone_id=selected_zone_id,
            )

            st.pydeck_chart(deck, use_container_width=True, height=MAP_HEIGHT)
            render_legend()

    with side_col:
        with st.container(border=True):
            card_title("선택 생활권 상세", "챗봇 질의 또는 직접 선택한 생활권입니다.")

            draw_focus_detail(
                selected_label=selected_label,
                selected_zone_id=selected_zone_id,
                selected_dongs=selected_dongs,
                zone_pred_kwh=zone_pred_kwh,
                zone_rank=int(zone_rank),
                n_zones=n_zones,
                total_day_kwh=total_day_kwh,
                peak_time=peak_time,
                peak_kwh=peak_kwh,
            )

        with st.container(border=True):
            card_title("수요 급증 알림", "선택 시각 기준 주의가 필요한 권역입니다.")
            draw_alerts(top10, selected_time)

    lower_left, lower_right = st.columns([1.1, 1.0], gap="small")

    with lower_left:
        with st.container(border=True):
            section_title("시간대별 전체 수요 추이", "선택 날짜의 전체 생활권 총 예측 충전량 변화입니다.")

            total_fig = make_total_demand_chart(
                pred=pred,
                selected_date=selected_date,
                selected_time=selected_time,
            )

            if total_fig is not None:
                st.plotly_chart(total_fig, use_container_width=True)
            else:
                st.info("선택 날짜의 시간대별 수요 데이터가 없습니다.")

    with lower_right:
        with st.container(border=True):
            section_title("선택 생활권 30분 단위 예측 패턴", "선택 생활권의 일중 충전 수요 변화를 확인합니다.")

            detail_fig = make_daily_pattern_chart(
                pred=pred,
                selected_date=selected_date,
                selected_zone_id=selected_zone_id,
            )

            if detail_fig is not None:
                st.plotly_chart(detail_fig, use_container_width=True)
            else:
                st.info("선택 생활권의 일별 패턴 데이터가 없습니다.")

    with st.container(border=True):
        card_title("수요 상위 권역 랭킹", "선택 시각 기준 상위 10개 생활권입니다.")
        draw_ranking(top10.head(10))


# =========================================================
# 데이터 테이블 / 해석 안내
# =========================================================
with st.expander("선택 시각 전체 예측 데이터 보기"):
    table_df = pred_filtered.merge(
        area_info[["생활권역ID", "생활권역라벨", "생활권역표시명", "행정동명목록"]],
        on="생활권역ID",
        how="left",
    ).sort_values("predicted_kwh", ascending=False)

    show_cols = [
        "datetime",
        "sample_idx",
        "global_time_idx",
        "daily_slot",
        "zone_idx",
        "생활권역ID",
        "생활권역표시명",
        "predicted_kwh",
    ]

    if "y_true_kwh" in table_df.columns:
        show_cols.append("y_true_kwh")

    display_df = table_df[show_cols].rename(
        columns={
            "datetime": "일시",
            "sample_idx": "test_sample_idx",
            "global_time_idx": "전체_time_idx",
            "daily_slot": "일중_slot",
            "zone_idx": "zone_idx",
            "생활권역ID": "생활권ID",
            "생활권역표시명": "생활권역",
            "predicted_kwh": "예측 충전량_kWh",
            "y_true_kwh": "실제 충전량_kWh",
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


with st.expander("데이터 해석 안내"):
    st.markdown(
        """
        ### 서비스 구조

        이 서비스는 다음과 같이 구성됩니다.

        ```text
        GRU / ST-LLM 예측 결과
        → 생활권별 30분 단위 충전수요 예측값
        → Streamlit 지도/대시보드 시각화
        → 챗봇형 LLM 인터페이스를 통한 자연어 질의
        ```

        현재 버전에서는 자연어 질의를 규칙 기반으로 해석합니다.
        예를 들어 `2025년 11월 25일 오후 6시에 청운효자동 수요 보여줘`라고 입력하면,
        시스템은 날짜, 시간, 위치를 추출하고 해당 위치가 포함된 생활권을 찾아 오른쪽 지도에 확대 표시합니다.

        ### 시간 인덱스 해석

        현재 GRU 예측 CSV의 `sample_idx`는 2025년 전체 기준 인덱스가 아니라,
        **test set 내부 인덱스**입니다.

        ```text
        global_time_idx = sample_idx + val_end + look_back
        datetime = 2025-01-01 00:00 + global_time_idx × 30분
        ```

        현재 `meta.json` 기준:

        ```text
        val_end = 15768
        look_back = 12
        test 시작 global_time_idx = 15780
        ```

        따라서 GRU 예측 CSV의 첫 번째 시각은 다음과 같습니다.

        ```text
        sample_idx 0 = 2025-11-25 18:00
        ```

        마지막 시각은 다음과 같습니다.

        ```text
        sample_idx 1739 = 2025-12-31 23:30
        ```

        ### 생활권역 매칭

        `area_info.xlsx`의 `생활권역ID`, `생활권역라벨`, `행정동명목록`을 이용해
        사용자가 입력한 위치명을 생활권역ID로 변환합니다.
        """
    )
