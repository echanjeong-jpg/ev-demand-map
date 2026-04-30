# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

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


# =========================================================
# Streamlit 기본 설정
# =========================================================
st.set_page_config(
    page_title="EV Demand Forecast Dashboard",
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
        padding-top: 2.6rem;
        padding-bottom: 2.2rem;
        max-width: 1540px;
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
        gap: 0.8rem;
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
        padding: 18px 20px;
        border-radius: 20px;
        box-shadow: 0 8px 24px rgba(35, 55, 80, 0.055);
        min-height: 106px;
    }

    div[data-testid="stMetricLabel"] {
        color: #8A93A3;
        font-size: 14px;
        font-weight: 800;
    }

    div[data-testid="stMetricValue"] {
        color: #232633;
        font-size: 29px;
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
    }

    div[data-baseweb="select"] > div {
        background: #F8FAFD;
        border-radius: 14px;
        border-color: #E6ECF3;
        min-height: 46px;
        box-shadow: 0 4px 16px rgba(35, 55, 80, 0.035);
    }

    .section-title {
        font-size: 23px;
        font-weight: 900;
        color: #222633;
        letter-spacing: -0.04em;
        margin: 0 0 4px 0;
    }

    .section-subtitle {
        color: #7C8594;
        font-size: 13px;
        font-weight: 600;
        margin: 0 0 12px 0;
    }

    .card-title {
        font-size: 20px;
        font-weight: 900;
        color: #222633;
        letter-spacing: -0.035em;
        margin-bottom: 0.45rem;
    }

    .card-subtitle {
        color: #7C8594;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 0.7rem;
    }

    .mini-label {
        color: #8A93A3;
        font-size: 12px;
        font-weight: 800;
        margin-bottom: 3px;
    }

    .mini-title {
        color: #222633;
        font-size: 15px;
        font-weight: 900;
        margin-bottom: 3px;
    }

    .mini-id {
        color: #2E8B55;
        font-size: 12px;
        font-weight: 900;
        margin-bottom: 6px;
    }

    .mini-value {
        color: #222633;
        font-size: 24px;
        font-weight: 900;
        letter-spacing: -0.04em;
    }

    .summary-mini-card {
        background: #F8FAFD;
        border-radius: 16px;
        padding: 14px 16px;
        border: 1px solid #E8EEF5;
        margin-top: 8px;
        margin-bottom: 10px;
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

    .rank-label {
        font-weight: 800;
        color: #303644;
        font-size: 13px;
    }

    .rank-value {
        font-weight: 900;
        color: #303644;
        font-size: 13px;
        text-align: right;
    }

    .muted {
        color: #8A93A3;
        font-size: 13px;
        font-weight: 700;
    }

    .green-id {
        color: #2E8B55;
        font-size: 13px;
        font-weight: 900;
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
        margin-top: 0.65rem;
        margin-bottom: 0.65rem;
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


def compact_summary_card(
    label: str,
    area_name: str,
    area_id: str,
    value_label: str,
    value: float,
) -> None:
    st.markdown(
        f"""
        <div class="summary-mini-card">
            <div class="mini-label">{label}</div>
            <div class="mini-title">{area_name}</div>
            <div class="mini-id">{area_id}</div>
            <div class="mini-label">{value_label}</div>
            <div class="mini-value">{value:.2f} kWh</div>
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


@st.cache_data(show_spinner="GRU 예측 결과 로딩 중...")
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

    zone_map = pd.DataFrame({
        "zone_idx": list(range(len(zone_ids))),
        "생활권역ID": zone_ids,
    })

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


def prepare_map_gdf(
    boundary_gdf: gpd.GeoDataFrame,
    area_info: pd.DataFrame,
    pred_filtered: pd.DataFrame,
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

    if "datetime" in gdf.columns:
        gdf["datetime"] = gdf["datetime"].astype(str)

    for col in ["zone_idx", "sample_idx", "global_time_idx", "daily_slot"]:
        if col in gdf.columns:
            gdf[col] = gdf[col].astype("Int64").astype(str)

    if "predicted_kwh" in gdf.columns:
        gdf["predicted_kwh"] = gdf["predicted_kwh"].astype(float).round(3)

    return gdf


# =========================================================
# 지도 / 차트 생성
# =========================================================
def make_polygon_layer(map_gdf: gpd.GeoDataFrame) -> pdk.Layer:
    vmin = float(map_gdf["predicted_kwh"].quantile(0.05))
    vmax = float(map_gdf["predicted_kwh"].quantile(0.95))

    map_gdf = map_gdf.copy()
    map_gdf["fill_color"] = map_gdf["predicted_kwh"].apply(
        lambda x: kwh_to_color(x, vmin, vmax)
    )

    return pdk.Layer(
        "GeoJsonLayer",
        data=json.loads(map_gdf.to_json()),
        pickable=True,
        stroked=True,
        filled=True,
        extruded=False,
        get_fill_color="properties.fill_color",
        get_line_color=[255, 255, 255, 185],
        get_line_width=20,
        line_width_min_pixels=1.2,
        auto_highlight=True,
    )


def make_column_layer(map_gdf: gpd.GeoDataFrame) -> pdk.Layer:
    df = pd.DataFrame(map_gdf.drop(columns="geometry")).copy()

    vmin = float(df["predicted_kwh"].quantile(0.05))
    vmax = float(df["predicted_kwh"].quantile(0.95))

    df["fill_color"] = df["predicted_kwh"].apply(lambda x: kwh_to_color(x, vmin, vmax))
    df["elevation"] = df["predicted_kwh"].fillna(0) * 70

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


def make_deck(map_gdf: gpd.GeoDataFrame, use_3d_column: bool) -> pdk.Deck:
    layers = [make_polygon_layer(map_gdf)]

    if use_3d_column:
        layers.append(make_column_layer(map_gdf))

    view_state = pdk.ViewState(
        latitude=37.5665,
        longitude=126.9780,
        zoom=10.05,
        pitch=38 if use_3d_column else 0,
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
        height=310,
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
        height=310,
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
# 화면 렌더링 함수
# =========================================================
def draw_ranking(top_df: pd.DataFrame):
    if top_df.empty:
        st.info("표시할 랭킹 데이터가 없습니다.")
        return

    max_v = float(top_df["predicted_kwh"].max())

    for i, row in enumerate(top_df.itertuples(), start=1):
        label = getattr(row, "생활권역표시명", getattr(row, "생활권역ID"))
        value = float(getattr(row, "predicted_kwh"))
        pct = 0.0 if max_v == 0 else value / max_v

        r1, r2, r3, r4 = st.columns([0.1, 0.42, 0.30, 0.18])
        r1.markdown(f"**{i}**")
        r2.markdown(f"<span class='rank-label'>{label}</span>", unsafe_allow_html=True)
        r3.progress(float(pct))
        r4.markdown(f"<div class='rank-value'>{value:,.1f} kWh</div>", unsafe_allow_html=True)


def draw_alerts(top_df: pd.DataFrame, selected_time: str):
    if top_df.empty:
        st.info("수요 알림을 생성할 수 없습니다.")
        return

    alert_rows = top_df.head(3).copy()

    for i, row in enumerate(alert_rows.itertuples(), start=1):
        label = getattr(row, "생활권역표시명", getattr(row, "생활권역ID"))
        value = float(getattr(row, "predicted_kwh"))

        if i == 1:
            message = f"{label} — 선택 시각 최고 수요 예상"
        elif i == 2:
            message = f"{label} — 충전 수요 집중 권역"
        else:
            message = f"{label} — 운영 여유 확인 필요"

        col_msg, col_time = st.columns([0.78, 0.22])
        with col_msg:
            st.markdown(f"**{message}**")
            st.caption(f"예측 {value:,.1f} kWh")
        with col_time:
            st.markdown(f"**{selected_time}**")
        if i < len(alert_rows):
            st.divider()


def draw_area_detail(
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
    st.markdown(f"#### {selected_label}")
    st.caption(selected_zone_id)

    if selected_dongs:
        st.markdown("**포함 행정동**")
        st.caption(selected_dongs)

    d1, d2 = st.columns(2)
    d3, d4 = st.columns(2)

    d1.metric("현재 예측", f"{zone_pred_kwh:.1f} kWh")
    d2.metric("수요 순위", f"{int(zone_rank)} / {n_zones}")
    d3.metric("일일 총량", f"{total_day_kwh:.0f} kWh")
    d4.metric("피크 시간", f"{peak_time}", f"{peak_kwh:.1f} kWh")


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
# 상단 여백: Framer iframe 상단 잘림 방지
# =========================================================
st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)


# =========================================================
# 컨트롤 영역
# =========================================================
with st.container(border=True):
    section_title("예측 조건 설정", "날짜, 시간, 지도 표시 방식, 상세 생활권을 선택합니다.")

    available_dates = sorted(pred["date_str"].unique())

    c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.15, 3.1])

    with c1:
        selected_date = st.selectbox(
            "날짜 선택",
            available_dates,
            index=0,
        )

    available_times = sorted(pred[pred["date_str"] == selected_date]["time_str"].unique())

    default_time_index = 0
    if "18:00" in available_times:
        default_time_index = available_times.index("18:00")

    with c2:
        selected_time = st.selectbox(
            "시간 선택",
            available_times,
            index=default_time_index,
        )

    with c3:
        view_mode = st.selectbox(
            "지도 표시 방식",
            ["2D 히트맵", "3D 막대"],
            index=0,
        )

    use_3d_column = view_mode == "3D 막대"

    zone_label_map = area_info.copy()
    zone_label_map = zone_label_map[
        zone_label_map["생활권역ID"].isin(meta["zone_ids"])
    ].copy()

    zone_label_map["display"] = (
        zone_label_map["생활권역ID"] + " · " + zone_label_map["생활권역표시명"]
    )
    zone_label_map = zone_label_map.sort_values("생활권역ID")

    with c4:
        selected_zone_display = st.selectbox(
            "상세 조회 생활권",
            zone_label_map["display"].tolist(),
            index=0,
        )

    selected_zone_id = zone_label_map.loc[
        zone_label_map["display"] == selected_zone_display,
        "생활권역ID",
    ].iloc[0]


# =========================================================
# 선택 시각 데이터 생성
# =========================================================
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
)

selected_dt = pred_filtered["datetime"].iloc[0]
daily_slot = int(pred_filtered["daily_slot"].iloc[0])
global_time_idx = int(pred_filtered["global_time_idx"].iloc[0])

n_zones = pred["zone_idx"].nunique()

total_kwh = pred_filtered["predicted_kwh"].sum()
mean_kwh = pred_filtered["predicted_kwh"].mean()

q75 = pred_filtered["predicted_kwh"].quantile(0.75)
high_count = int((pred_filtered["predicted_kwh"] >= q75).sum())

max_row = pred_filtered.loc[pred_filtered["predicted_kwh"].idxmax()]
min_row = pred_filtered.loc[pred_filtered["predicted_kwh"].idxmin()]


# =========================================================
# KPI 영역
# =========================================================
with st.container(border=True):
    section_title("선택 시각 핵심 지표", f"{selected_dt:%Y-%m-%d %H:%M} 기준 생활권별 예측 결과입니다.")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    kpi1.metric("예측 대상 권역", f"{n_zones} 개")
    kpi2.metric("선택 시각 총 예측 충전량", f"{total_kwh:,.0f} kWh")
    kpi3.metric("고수요 예상 권역", f"{high_count} 개")
    kpi4.metric("생활권 평균 예측 수요", f"{mean_kwh:.1f} kWh")


# =========================================================
# 지도 + 요약/랭킹 영역
# =========================================================
main_left, main_right = st.columns([1.72, 0.98], gap="large")

with main_left:
    with st.container(border=True):
        section_title(
            "생활권별 충전 수요 지도",
            f"{selected_dt:%Y-%m-%d %H:%M} · Daily Slot {daily_slot} / 47 · 전체 Time Index {global_time_idx}",
        )

        deck = make_deck(map_gdf, use_3d_column=use_3d_column)
        st.pydeck_chart(deck, use_container_width=True, height=650)
        render_legend()

with main_right:
    max_zone_id = max_row["생활권역ID"]
    min_zone_id = min_row["생활권역ID"]

    max_area = area_info[area_info["생활권역ID"] == max_zone_id]
    min_area = area_info[area_info["생활권역ID"] == min_zone_id]

    max_label = (
        max_area["생활권역표시명"].iloc[0]
        if not max_area.empty
        else max_zone_id
    )
    min_label = (
        min_area["생활권역표시명"].iloc[0]
        if not min_area.empty
        else min_zone_id
    )

    with st.container(border=True):
        card_title("선택 시각 요약")

        compact_summary_card(
            label="최대 수요 생활권",
            area_name=max_label,
            area_id=max_zone_id,
            value_label="최대 예측 수요",
            value=float(max_row["predicted_kwh"]),
        )

        compact_summary_card(
            label="최소 수요 생활권",
            area_name=min_label,
            area_id=min_zone_id,
            value_label="최소 예측 수요",
            value=float(min_row["predicted_kwh"]),
        )

    top10 = pred_filtered.sort_values("predicted_kwh", ascending=False).head(10)
    top10 = top10.merge(
        area_info[["생활권역ID", "생활권역라벨", "생활권역표시명"]],
        on="생활권역ID",
        how="left",
    )

    with st.container(border=True):
        card_title("수요 상위 권역 랭킹")
        draw_ranking(top10.head(5))


# =========================================================
# 시간대별 분석 영역
# =========================================================
trend_col, alert_col = st.columns([1.72, 0.98], gap="large")

with trend_col:
    with st.container(border=True):
        section_title("시간대별 수요 추이", "선택한 날짜의 전체 생활권 총 예측 충전량 변화입니다.")

        total_fig = make_total_demand_chart(
            pred=pred,
            selected_date=selected_date,
            selected_time=selected_time,
        )

        if total_fig is not None:
            st.plotly_chart(total_fig, use_container_width=True)
        else:
            st.info("선택 날짜의 시간대별 수요 데이터가 없습니다.")

with alert_col:
    with st.container(border=True):
        card_title("수요 급증 알림", "선택 시각 기준 주의가 필요한 권역입니다.")
        draw_alerts(top10, selected_time)


# =========================================================
# 상세 생활권 분석 영역
# =========================================================
detail_col, pattern_col = st.columns([0.98, 1.72], gap="large")

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

zone_now = pred_filtered[pred_filtered["생활권역ID"] == selected_zone_id].copy()

if not zone_now.empty:
    zone_pred_kwh = float(zone_now["predicted_kwh"].iloc[0])
    zone_rank = (
        pred_filtered["predicted_kwh"]
        .rank(method="min", ascending=False)
        .loc[zone_now.index[0]]
    )

    day_zone = pred[
        (pred["date_str"] == selected_date)
        & (pred["생활권역ID"] == selected_zone_id)
    ].copy()

    if not day_zone.empty:
        peak_row = day_zone.loc[day_zone["predicted_kwh"].idxmax()]
        total_day_kwh = day_zone["predicted_kwh"].sum()
        peak_time = peak_row["time_str"]
        peak_kwh = float(peak_row["predicted_kwh"])
    else:
        total_day_kwh = 0
        peak_time = "-"
        peak_kwh = 0
else:
    zone_pred_kwh = 0
    zone_rank = 0
    total_day_kwh = 0
    peak_time = "-"
    peak_kwh = 0

with detail_col:
    with st.container(border=True):
        card_title("선택 생활권 상세")
        draw_area_detail(
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

with pattern_col:
    with st.container(border=True):
        section_title("선택 생활권 30분 단위 예측 패턴", "선택한 생활권의 일중 충전 수요 변화를 확인합니다.")

        fig = make_daily_pattern_chart(
            pred=pred,
            selected_date=selected_date,
            selected_zone_id=selected_zone_id,
        )

        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("선택 생활권의 일별 패턴 데이터가 없습니다.")


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

    st.dataframe(
        table_df[show_cols].rename(columns={
            "datetime": "일시",
            "sample_idx": "test_sample_idx",
            "global_time_idx": "전체_time_idx",
            "daily_slot": "일중_slot",
            "zone_idx": "zone_idx",
            "생활권역ID": "생활권ID",
            "생활권역표시명": "생활권역",
            "predicted_kwh": "예측 충전량_kWh",
            "y_true_kwh": "실제 충전량_kWh",
        }),
        use_container_width=True,
        hide_index=True,
    )


with st.expander("데이터 해석 안내"):
    st.markdown(
        """
        ### 시간 인덱스 해석

        현재 GRU 예측 CSV의 `sample_idx`는 2025년 전체 기준 인덱스가 아니라,
        **test set 내부 인덱스**입니다.

        따라서 본 서비스에서는 다음 방식으로 실제 시간을 복원합니다.

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

        `meta.json`의 `zone_ids`를 사용해 GRU 결과의 `zone_idx`를 실제 생활권역ID와 연결합니다.

        전체 116개 생활권역 중 `생활권경계_063`은 텐서 생성 과정에서 제외되어,
        본 지도 서비스에서는 115개 생활권역만 표시합니다.
        """
    )
