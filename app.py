# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import plotly.express as px
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
# Streamlit 기본 UI
# =========================================================
st.set_page_config(
    page_title="서울시 생활권별 전기차 충전 수요 예측 지도",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 유틸 함수
# =========================================================
def normalize_id(value) -> str:
    """
    생활권역ID 문자열 정규화.
    예: 생활권경계_006 형태 유지.
    """
    if pd.isna(value):
        return ""
    return str(value).strip()


def get_prediction_column(df: pd.DataFrame) -> str:
    """
    예측 충전량 컬럼명 자동 탐색.
    GRU 결과는 y_pred_kwh를 사용.
    추후 ST-LLM 결과가 predicted_kwh 컬럼을 쓰더라도 대응 가능.
    """
    candidates = ["y_pred_kwh", "predicted_kwh", "pred_kwh", "prediction_kwh"]
    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "예측값 컬럼을 찾을 수 없습니다. "
        "CSV에 y_pred_kwh 또는 predicted_kwh 컬럼이 필요합니다."
    )


def get_true_column(df: pd.DataFrame) -> str | None:
    """
    실제값 컬럼명 탐색.
    서비스 화면에는 필수는 아니지만, 있으면 참고용으로 사용.
    """
    candidates = ["y_true_kwh", "true_kwh", "actual_kwh"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def kwh_to_color(value: float, vmin: float, vmax: float) -> list[int]:
    """
    predicted_kwh 값을 PyDeck 색상으로 변환.
    낮은 수요: 연한 파랑
    높은 수요: 진한 파랑/보라 계열
    """
    if pd.isna(value):
        return [210, 210, 210, 80]

    if vmax <= vmin:
        ratio = 0.5
    else:
        ratio = (value - vmin) / (vmax - vmin)
        ratio = float(np.clip(ratio, 0.0, 1.0))

    # 색상: light cyan → blue/purple
    r = int(90 + 80 * ratio)
    g = int(210 - 150 * ratio)
    b = int(245 - 40 * ratio)
    a = 180

    return [r, g, b, a]


def add_datetime_to_predictions(pred: pd.DataFrame, meta: Dict) -> pd.DataFrame:
    """
    GRU test prediction CSV의 sample_idx를 실제 datetime으로 변환.

    현재 GRU 결과 CSV는 전체 2025년 기준 sample_idx가 아니라,
    test set 내부 sample_idx임.

    meta 기준:
    train_end = 14016
    val_end = 15768
    look_back = 12

    test prediction sample_idx 0의 target 시각:
    global_time_idx = val_end + look_back = 15780
    datetime = 2025-01-01 00:00 + 15780 * 30분
             = 2025-11-25 18:00
    """
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


@st.cache_data(show_spinner="meta.json 로딩 중...")
def load_meta(meta_path: Path) -> Dict:
    """
    meta.json 로딩 함수.
    사용 환경에 따라 utf-8, utf-8-sig, cp949, euc-kr 인코딩이 섞일 수 있으므로
    여러 인코딩을 순서대로 시도한다.
    """
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

    # 음수 예측값은 시각화 서비스에서는 0으로 보정하는 것이 자연스러움
    # 원본 분석값을 유지하고 싶으면 아래 라인을 주석 처리
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


@st.cache_data(show_spinner="생활권역 엑셀 정보 로딩 중...")
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

    return area[keep_cols].copy()


@st.cache_data(show_spinner="생활권역 SHP 경계 로딩 중...")
def load_living_area_gdf(shp_path: Path, meta_zone_ids: list[str]) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(shp_path, encoding="cp949")

    # 컬럼명 확인 및 ID 컬럼 정리
    if "ID" not in gdf.columns:
        raise ValueError(
            f"SHP에 ID 컬럼이 없습니다. 현재 컬럼: {list(gdf.columns)}"
        )

    gdf["ID"] = gdf["ID"].apply(normalize_id)

    # 지역생활권만 사용
    if "ENT_NAME" in gdf.columns:
        gdf = gdf[gdf["ENT_NAME"] == "지역생활권"].copy()

    # GRU 텐서에 포함된 115개 생활권만 사용
    gdf = gdf[gdf["ID"].isin(meta_zone_ids)].copy()

    # 좌표계 설정
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=SOURCE_EPSG)

    gdf = gdf.to_crs(epsg=TARGET_EPSG)

    # geometry 단순화: 웹 지도 성능 개선
    gdf["geometry"] = gdf["geometry"].simplify(0.00015, preserve_topology=True)

    # 중심점 계산
    centroid = gdf.geometry.centroid
    gdf["lon"] = centroid.x
    gdf["lat"] = centroid.y

    return gdf


def prepare_map_gdf(
    boundary_gdf: gpd.GeoDataFrame,
    area_info: pd.DataFrame,
    pred_filtered: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """
    선택된 시각의 예측값과 생활권 경계 결합.
    """
    gdf = boundary_gdf.copy()

    gdf = gdf.merge(
        area_info,
        left_on="ID",
        right_on="생활권역ID",
        how="left",
    )

    gdf = gdf.merge(
        pred_filtered[[
            "생활권역ID",
            "zone_idx",
            "sample_idx",
            "global_time_idx",
            "datetime",
            "date_str",
            "time_str",
            "daily_slot",
            "predicted_kwh",
        ]],
        on="생활권역ID",
        how="left",
    )

    # 표시용 텍스트
    if "생활권역라벨" not in gdf.columns:
        gdf["생활권역라벨"] = gdf["ID"]

    if "행정동명목록" not in gdf.columns:
        gdf["행정동명목록"] = ""

        # GeoDataFrame.to_json()에서 Timestamp 타입이 직렬화되지 않는 문제 방지
    if "datetime" in gdf.columns:
        gdf["datetime"] = gdf["datetime"].astype(str)

    # numpy 타입도 JSON 직렬화 문제를 일으킬 수 있으므로 기본 Python 타입으로 정리
    for col in ["zone_idx", "sample_idx", "global_time_idx", "daily_slot"]:
        if col in gdf.columns:
            gdf[col] = gdf[col].astype("Int64").astype(str)

    if "predicted_kwh" in gdf.columns:
        gdf["predicted_kwh"] = gdf["predicted_kwh"].astype(float).round(3)

    return gdf


def make_polygon_layer(map_gdf: gpd.GeoDataFrame) -> pdk.Layer:
    """
    생활권역 polygon 색상 지도.
    """
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
        get_line_color=[255, 255, 255, 160],
        get_line_width=20,
        line_width_min_pixels=1,
        auto_highlight=True,
    )


def make_column_layer(map_gdf: gpd.GeoDataFrame) -> pdk.Layer:
    """
    생활권역 중심점 기준 3D 막대 지도.
    """
    df = pd.DataFrame(map_gdf.drop(columns="geometry")).copy()

    vmin = float(df["predicted_kwh"].quantile(0.05))
    vmax = float(df["predicted_kwh"].quantile(0.95))

    df["fill_color"] = df["predicted_kwh"].apply(
        lambda x: kwh_to_color(x, vmin, vmax)
    )

    # 막대 높이 스케일
    df["elevation"] = df["predicted_kwh"].fillna(0) * 80

    return pdk.Layer(
        "ColumnLayer",
        data=df,
        get_position=["lon", "lat"],
        get_elevation="elevation",
        elevation_scale=1,
        radius=450,
        get_fill_color="fill_color",
        pickable=True,
        auto_highlight=True,
    )


def make_deck(map_gdf: gpd.GeoDataFrame, use_3d_column: bool) -> pdk.Deck:
    """
    PyDeck 지도 객체 생성.
    """
    layers = [make_polygon_layer(map_gdf)]

    if use_3d_column:
        layers.append(make_column_layer(map_gdf))

    view_state = pdk.ViewState(
        latitude=37.5665,
        longitude=126.9780,
        zoom=10.1,
        pitch=45 if use_3d_column else 0,
        bearing=0,
    )

    tooltip = {
        "html": """
        <b>{생활권역라벨}</b><br/>
        생활권ID: {ID}<br/>
        zone_idx: {zone_idx}<br/>
        예측 충전량: <b>{predicted_kwh}</b> kWh<br/>
        행정동: {행정동명목록}
        """,
        "style": {
            "backgroundColor": "rgba(20, 30, 45, 0.88)",
            "color": "white",
            "fontSize": "12px",
        },
    }

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=None,
        tooltip=tooltip,
    )


def make_daily_pattern_chart(pred: pd.DataFrame, selected_date: str, selected_zone_id: str):
    """
    선택 생활권의 선택 날짜 48개 슬롯 예측 패턴.
    """
    day_df = pred[
        (pred["date_str"] == selected_date) &
        (pred["생활권역ID"] == selected_zone_id)
    ].copy()

    if day_df.empty:
        return None

    day_df = day_df.sort_values("datetime")

    fig = px.line(
        day_df,
        x="time_str",
        y="predicted_kwh",
        markers=True,
        title="선택 생활권의 30분 단위 예측 충전량",
        labels={
            "time_str": "시간",
            "predicted_kwh": "예측 충전량, kWh",
        },
    )

    fig.update_layout(
        height=330,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(tickangle=-45),
    )

    return fig


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
# 사이드바
# =========================================================
st.sidebar.title("설정")

st.sidebar.markdown("### 데이터 상태")

n_samples = pred["sample_idx"].nunique()
n_zones = pred["zone_idx"].nunique()
min_dt = pred["datetime"].min()
max_dt = pred["datetime"].max()

st.sidebar.write(f"예측 sample 수: **{n_samples:,}**")
st.sidebar.write(f"생활권 수: **{n_zones:,}**")
st.sidebar.write(f"기간: **{min_dt:%Y-%m-%d %H:%M} ~ {max_dt:%Y-%m-%d %H:%M}**")

st.sidebar.markdown("---")

available_dates = sorted(pred["date_str"].unique())

selected_date = st.sidebar.selectbox(
    "날짜 선택",
    available_dates,
    index=0,
)

available_times = sorted(
    pred[pred["date_str"] == selected_date]["time_str"].unique()
)

default_time_index = 0
if "18:00" in available_times:
    default_time_index = available_times.index("18:00")

selected_time = st.sidebar.selectbox(
    "시간 선택, 30분 단위",
    available_times,
    index=default_time_index,
)

use_3d_column = st.sidebar.checkbox(
    "3D 막대 표시",
    value=True,
)

st.sidebar.markdown("---")

# 선택 생활권
zone_label_map = area_info.copy()
zone_label_map = zone_label_map[
    zone_label_map["생활권역ID"].isin(meta["zone_ids"])
].copy()

if "생활권역라벨" in zone_label_map.columns:
    zone_label_map["display"] = (
        zone_label_map["생활권역ID"] + " · " + zone_label_map["생활권역라벨"]
    )
else:
    zone_label_map["display"] = zone_label_map["생활권역ID"]

zone_label_map = zone_label_map.sort_values("생활권역ID")

selected_zone_display = st.sidebar.selectbox(
    "상세 조회 생활권",
    zone_label_map["display"].tolist(),
    index=0,
)

selected_zone_id = zone_label_map.loc[
    zone_label_map["display"] == selected_zone_display,
    "생활권역ID"
].iloc[0]


# =========================================================
# 선택 시각 데이터 생성
# =========================================================
pred_filtered = pred[
    (pred["date_str"] == selected_date) &
    (pred["time_str"] == selected_time)
].copy()

if pred_filtered.empty:
    st.warning("선택한 날짜/시간에 해당하는 예측 데이터가 없습니다.")
    st.stop()

map_gdf = prepare_map_gdf(
    boundary_gdf=boundary_gdf,
    area_info=area_info,
    pred_filtered=pred_filtered,
)


# =========================================================
# 메인 화면
# =========================================================
st.title("서울시 생활권별 전기차 충전 수요 예측 지도")
st.caption(
    "GRU test 구간 예측 결과를 기반으로 2025년 30분 단위 생활권별 예측 충전량을 시각화한 서비스 프로토타입입니다."
)

selected_dt = pred_filtered["datetime"].iloc[0]
daily_slot = int(pred_filtered["daily_slot"].iloc[0])
global_time_idx = int(pred_filtered["global_time_idx"].iloc[0])

col_a, col_b, col_c, col_d = st.columns(4)

mean_kwh = pred_filtered["predicted_kwh"].mean()
max_row = pred_filtered.loc[pred_filtered["predicted_kwh"].idxmax()]
min_row = pred_filtered.loc[pred_filtered["predicted_kwh"].idxmin()]

col_a.metric("선택 시각", f"{selected_dt:%Y-%m-%d %H:%M}")
col_b.metric("Daily Slot", f"{daily_slot} / 47")
col_c.metric("평균 예측 수요", f"{mean_kwh:.2f} kWh")
col_d.metric("전체 Time Index", f"{global_time_idx}")

col1, col2 = st.columns([3.2, 1.2])

with col1:
    st.subheader("생활권별 예측 충전량 지도")
    deck = make_deck(map_gdf, use_3d_column=use_3d_column)
    st.pydeck_chart(deck, use_container_width=True)

with col2:
    st.subheader("선택 시각 요약")

    max_zone_id = max_row["생활권역ID"]
    min_zone_id = min_row["생활권역ID"]

    max_label = area_info.loc[
        area_info["생활권역ID"] == max_zone_id,
        "생활권역라벨"
    ]
    min_label = area_info.loc[
        area_info["생활권역ID"] == min_zone_id,
        "생활권역라벨"
    ]

    max_label_text = max_label.iloc[0] if not max_label.empty else max_zone_id
    min_label_text = min_label.iloc[0] if not min_label.empty else min_zone_id

    st.markdown(
        f"""
        **최대 수요 생활권**  
        {max_label_text}  
        `{max_zone_id}`  
        **{max_row["predicted_kwh"]:.2f} kWh**

        **최소 수요 생활권**  
        {min_label_text}  
        `{min_zone_id}`  
        **{min_row["predicted_kwh"]:.2f} kWh**
        """
    )

    st.markdown("---")

    st.subheader("수요 상위 10개 생활권")

    top10 = pred_filtered.sort_values("predicted_kwh", ascending=False).head(10)
    top10 = top10.merge(
        area_info[["생활권역ID", "생활권역라벨"]],
        on="생활권역ID",
        how="left",
    )

    st.dataframe(
        top10[[
            "생활권역ID",
            "생활권역라벨",
            "zone_idx",
            "predicted_kwh",
        ]].rename(columns={
            "생활권역ID": "생활권ID",
            "생활권역라벨": "생활권역",
            "zone_idx": "zone_idx",
            "predicted_kwh": "예측 kWh",
        }),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# 상세 분석 영역
# =========================================================
st.markdown("---")
st.header("생활권 상세 예측 패턴")

selected_area = area_info[area_info["생활권역ID"] == selected_zone_id].copy()

if selected_area.empty:
    selected_label = selected_zone_id
    selected_dongs = ""
else:
    selected_label = selected_area["생활권역라벨"].iloc[0] if "생활권역라벨" in selected_area.columns else selected_zone_id
    selected_dongs = selected_area["행정동명목록"].iloc[0] if "행정동명목록" in selected_area.columns else ""

zone_now = pred_filtered[pred_filtered["생활권역ID"] == selected_zone_id].copy()

detail_col1, detail_col2 = st.columns([1, 2])

with detail_col1:
    st.subheader(selected_label)
    st.write(f"생활권ID: `{selected_zone_id}`")

    if selected_dongs:
        st.write("포함 행정동")
        st.caption(str(selected_dongs))

    if not zone_now.empty:
        zone_pred_kwh = float(zone_now["predicted_kwh"].iloc[0])
        zone_rank = (
            pred_filtered["predicted_kwh"]
            .rank(method="min", ascending=False)
            .loc[zone_now.index[0]]
        )

        st.metric("현재 시각 예측 충전량", f"{zone_pred_kwh:.2f} kWh")
        st.metric("선택 시각 수요 순위", f"{int(zone_rank)} / {n_zones}")

        # 선택 날짜 기준 피크
        day_zone = pred[
            (pred["date_str"] == selected_date) &
            (pred["생활권역ID"] == selected_zone_id)
        ].copy()

        if not day_zone.empty:
            peak_row = day_zone.loc[day_zone["predicted_kwh"].idxmax()]
            total_day_kwh = day_zone["predicted_kwh"].sum()

            st.metric("선택 날짜 총 예측 충전량", f"{total_day_kwh:.2f} kWh")
            st.metric(
                "선택 날짜 피크 시간",
                f"{peak_row['time_str']} · {peak_row['predicted_kwh']:.2f} kWh",
            )
    else:
        st.warning("선택 생활권의 현재 시각 예측값이 없습니다.")

with detail_col2:
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
# 전체 데이터 표
# =========================================================
st.markdown("---")
st.header("선택 시각 예측 데이터")

table_df = pred_filtered.merge(
    area_info[["생활권역ID", "생활권역라벨", "행정동명목록"]],
    on="생활권역ID",
    how="left",
)

table_df = table_df.sort_values("predicted_kwh", ascending=False)

show_cols = [
    "datetime",
    "sample_idx",
    "global_time_idx",
    "daily_slot",
    "zone_idx",
    "생활권역ID",
    "생활권역라벨",
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
        "생활권역라벨": "생활권역",
        "predicted_kwh": "예측 충전량_kWh",
        "y_true_kwh": "실제 충전량_kWh",
    }),
    use_container_width=True,
    hide_index=True,
)


# =========================================================
# 안내 문구
# =========================================================
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

        ```text
        zone_idx 0 = 생활권경계_006
        zone_idx 1 = 생활권경계_007
        ...
        ```

        전체 116개 생활권역 중 `생활권경계_063`은 텐서 생성 과정에서 제외되어,
        본 지도 서비스에서는 115개 생활권역만 표시합니다.
        """
    )
