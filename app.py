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

# 새 모델 예측 결과
PRED_FILE_PATH = BASE_DIR / "preds.npy"
TRUE_FILE_PATH = BASE_DIR / "trues.npy"

META_JSON_PATH = BASE_DIR / "meta.json"
AREA_EXCEL_PATH = BASE_DIR / "area_info.xlsx"
SHP_PATH = BASE_DIR / "UPIS_SHP_ZON100.shp"

MODEL_DISPLAY_NAME = "E-Vlog"

SOURCE_EPSG = 5174
TARGET_EPSG = 4326

START_DATETIME = pd.Timestamp("2025-01-01 00:00:00")
TIME_UNIT_MINUTES = 30

DEFAULT_DATE = "2025-11-25"
DEFAULT_TIME = "18:00"

FIXED_QUERY_EXAMPLE = "2025년 11월 25일 오후 6시에 청운효자동 수요 보여줘"

PANEL_HEIGHT = 625
MAP_HEIGHT = 485
CHAT_SCROLL_HEIGHT = 430

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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Holtwood+One+SC&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif;
    }

    .stApp {
        background: #FFFFFF !important;
        box-shadow: none !important;
        overflow: visible !important;
    }

    .block-container {
        padding-top: 0.85rem;
        padding-bottom: 1.25rem;
        padding-left: 1.35rem;
        padding-right: 1.35rem;
        max-width: 1440px;
        overflow: visible !important;
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
        overflow: visible !important;
    }

    div[data-testid="column"] {
        padding: 10px 8px !important;
        overflow: visible !important;
    }

    /*
    기본 Streamlit border wrapper는 일단 평평하게 둔다.
    실제 패널 그림자는 ev-panel-marker가 들어간 컨테이너에만 적용한다.
    */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        overflow: visible !important;
    }

    /*
    핵심:
    각 큰 패널 내부에 ev-panel-marker를 넣고,
    해당 marker를 포함한 border wrapper에만 about us 카드처럼 외부 그림자 적용.
    */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.ev-panel-marker) {
        background: #FFFFFF !important;
        border-radius: 22px !important;
        border: 1.45px solid rgba(20, 20, 20, 0.48) !important;
        box-shadow:
            0 24px 42px rgba(0, 0, 0, 0.20),
            0 8px 18px rgba(0, 0, 0, 0.11),
            0 1px 0 rgba(255, 255, 255, 0.96) inset !important;
        overflow: visible !important;
        position: relative !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.ev-panel-marker) > div {
        overflow: visible !important;
    }

    .ev-panel-marker {
        display: none;
    }

    .panel-kicker {
        color: #1F78B4;
        font-family: "Holtwood One SC", Georgia, serif;
        font-size: 12px;
        font-weight: 400;
        letter-spacing: -0.015em;
        line-height: 1;
        margin-bottom: 6px;
        text-transform: uppercase;
    }

    .panel-title {
        color: #111111;
        font-size: 22px;
        font-weight: 900;
        letter-spacing: -0.06em;
        margin-bottom: 3px;
        line-height: 1.12;
    }

    .panel-subtitle {
        color: #5F666F;
        font-size: 12px;
        font-weight: 500;
        margin-bottom: 10px;
        line-height: 1.35;
    }

    .legend-wrap {
        display: flex;
        align-items: center;
        gap: 9px;
        margin-top: 8px;
        color: #555C66;
        font-weight: 600;
        font-size: 12px;
    }

    .legend-bar {
        height: 8px;
        flex: 1;
        border-radius: 999px;
        background: linear-gradient(90deg, #E4EDF7, #86B3F7, #2E6BEA, #132D6B);
        box-shadow: none;
    }

    iframe {
        border-radius: 16px;
        box-shadow: none !important;
        border: none !important;
    }

    div[data-testid="stTextInput"] {
        margin-bottom: 0rem;
    }

    div[data-testid="stTextInput"] > div {
        background: #FFFFFF !important;
        border-radius: 14px !important;
        border: 1px solid rgba(20, 20, 20, 0.18) !important;
        box-shadow: none !important;
    }

    div[data-testid="stTextInput"] input {
        background: #FFFFFF !important;
        color: #111111 !important;
        border-radius: 14px !important;
        min-height: 40px;
        font-size: 12px;
        font-weight: 500;
        border: none !important;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #8A8F98 !important;
        opacity: 1 !important;
        font-weight: 500;
    }

    div[data-testid="stForm"] {
        border: 0;
        padding: 0;
        background: transparent;
    }

    div[data-testid="stFormSubmitButton"] button {
        border-radius: 13px;
        min-height: 40px;
        font-weight: 850;
        background: #1F78B4;
        color: white;
        border: 0;
        margin-top: -0.15rem;
        box-shadow: none;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background: #17669B;
        color: white;
        border: 0;
    }

    div[data-testid="stToggle"] label {
        color: #111111 !important;
        font-weight: 700 !important;
    }

    .llm-error-caption {
        color: #9AA6B5;
        font-size: 10px;
        font-weight: 500;
        margin-top: 4px;
        word-break: break-all;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 화면 보조 함수
# =========================================================
def mark_panel() -> None:
    st.markdown('<span class="ev-panel-marker"></span>', unsafe_allow_html=True)


def panel_title(title: str, subtitle: str | None = None, kicker: str | None = None) -> None:
    kicker_html = f'<div class="panel-kicker">{kicker}</div>' if kicker else ""

    if subtitle:
        st.markdown(
            f"""
            {kicker_html}
            <div class="panel-title">{title}</div>
            <div class="panel-subtitle">{subtitle}</div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            {kicker_html}
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


def get_recent_chat_context(messages: list[dict], limit: int = 6) -> list[dict]:
    recent = messages[-limit:] if messages else []
    return [
        {
            "role": m.get("role", "assistant"),
            "content": str(m.get("content", ""))[:700],
        }
        for m in recent
    ]


# =========================================================
# Gemini API 함수
# =========================================================
def get_gemini_client() -> Optional[Any]:
    if genai is None:
        st.session_state.last_llm_error = (
            "google-genai 패키지를 불러오지 못했습니다. requirements.txt에 google-genai를 추가하세요."
        )
        return None

    api_key = None

    try:
        api_key = st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        api_key = None

    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        st.session_state.last_llm_error = "GEMINI_API_KEY가 설정되어 있지 않습니다."
        return None

    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        st.session_state.last_llm_error = f"Gemini client 생성 실패: {e}"
        return None


def get_gemini_model() -> str:
    try:
        return st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash")
    except Exception:
        return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


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

        text = getattr(response, "text", None)

        if text and str(text).strip():
            st.session_state.last_llm_error = None
            return str(text).strip()

        st.session_state.last_llm_error = "Gemini 응답 text가 비어 있습니다."
        return None

    except Exception as e:
        st.session_state.last_llm_error = f"Gemini 호출 실패: {e}"
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
def load_predictions(
    pred_path: Path,
    meta: Dict,
    true_path: Path | None = None,
) -> Tuple[pd.DataFrame, str, str | None]:
    zone_ids = meta["zone_ids"]
    n_zones_meta = len(zone_ids)
    suffix = pred_path.suffix.lower()

    if suffix == ".npy":
        pred_arr = np.load(pred_path)

        if pred_arr.ndim == 4:
            pred_arr = pred_arr[:, 0, :, 0]
        elif pred_arr.ndim == 3:
            if pred_arr.shape[-1] == 1:
                pred_arr = pred_arr[:, :, 0]
            else:
                pred_arr = pred_arr[:, 0, :]
        elif pred_arr.ndim == 2:
            pass
        else:
            raise ValueError(
                f"지원하지 않는 preds.npy shape입니다: {pred_arr.shape}. "
                "예상 shape은 (N,H,Z,1), (N,Z,1), (N,Z) 중 하나입니다."
            )

        n_samples, n_zones = pred_arr.shape

        if n_zones != n_zones_meta:
            raise ValueError(
                f"예측 결과의 zone 수({n_zones})와 meta.json의 zone_ids 수({n_zones_meta})가 다릅니다."
            )

        pred_arr = np.nan_to_num(pred_arr, nan=0.0, posinf=0.0, neginf=0.0)
        pred_arr = np.clip(pred_arr, 0, None)

        pred = pd.DataFrame(
            {
                "sample_idx": np.repeat(np.arange(n_samples), n_zones),
                "zone_idx": np.tile(np.arange(n_zones), n_samples),
                "predicted_kwh": pred_arr.reshape(-1),
            }
        )

        true_col = None

        if true_path is not None and true_path.exists():
            true_arr = np.load(true_path)

            if true_arr.ndim == 4:
                true_arr = true_arr[:, 0, :, 0]
            elif true_arr.ndim == 3:
                if true_arr.shape[-1] == 1:
                    true_arr = true_arr[:, :, 0]
                else:
                    true_arr = true_arr[:, 0, :]
            elif true_arr.ndim == 2:
                pass
            else:
                raise ValueError(
                    f"지원하지 않는 trues.npy shape입니다: {true_arr.shape}"
                )

            if true_arr.shape != pred_arr.shape:
                raise ValueError(
                    f"preds.npy shape {pred_arr.shape}와 trues.npy shape {true_arr.shape}가 다릅니다."
                )

            true_arr = np.nan_to_num(true_arr, nan=0.0, posinf=0.0, neginf=0.0)
            true_arr = np.clip(true_arr, 0, None)

            pred["y_true_kwh"] = true_arr.reshape(-1)
            true_col = "y_true_kwh"

        zone_map = pd.DataFrame(
            {
                "zone_idx": list(range(n_zones_meta)),
                "생활권역ID": zone_ids,
            }
        )

        pred = pred.merge(zone_map, on="zone_idx", how="left")

        if pred["생활권역ID"].isna().any():
            bad = pred[pred["생활권역ID"].isna()]["zone_idx"].unique()
            raise ValueError(f"meta.json과 매칭되지 않는 zone_idx가 있습니다: {bad}")

        pred = add_datetime_to_predictions(pred, meta)

        return pred, "predicted_kwh", true_col

    if suffix == ".csv":
        pred = pd.read_csv(pred_path)
    elif suffix in [".parquet", ".pq"]:
        pred = pd.read_parquet(pred_path)
    else:
        raise ValueError(
            f"지원하지 않는 예측 파일 형식입니다: {pred_path.name}. "
            "csv, parquet, npy만 지원합니다."
        )

    required_cols = {"sample_idx", "zone_idx"}
    missing = required_cols - set(pred.columns)
    if missing:
        raise ValueError(f"예측 파일에 필수 컬럼이 없습니다: {missing}")

    pred_col = get_prediction_column(pred)
    true_col = get_true_column(pred)

    pred[pred_col] = pd.to_numeric(pred[pred_col], errors="coerce")
    pred["sample_idx"] = pd.to_numeric(pred["sample_idx"], errors="coerce").astype(int)
    pred["zone_idx"] = pd.to_numeric(pred["zone_idx"], errors="coerce").astype(int)
    pred[pred_col] = pred[pred_col].clip(lower=0)

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
        true_col = "y_true_kwh"

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
def extract_any_date_candidate(text: str) -> Optional[str]:
    m = re.search(r"(20\d{2})[-년./\s]*(\d{1,2})[-월./\s]*(\d{1,2})", text)
    if m:
        y, mo, d = map(int, m.groups())
        return f"{y:04d}-{mo:02d}-{d:02d}"

    m = re.search(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일", text)
    if m:
        mo, d = map(int, m.groups())
        return f"2025-{mo:02d}-{d:02d}"

    m = re.search(r"(\d{1,2})[-/.](\d{1,2})", text)
    if m:
        mo, d = map(int, m.groups())
        return f"2025-{mo:02d}-{d:02d}"

    return None


def parse_date_from_text(text: str, available_dates: list[str]) -> Optional[str]:
    candidate = extract_any_date_candidate(text)
    if candidate and candidate in available_dates:
        return candidate
    return None


def extract_any_time_candidate(text: str) -> Optional[str]:
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

        if 0 <= h <= 23:
            return f"{h:02d}:{mi:02d}"

    m = re.search(r"(오전|오후)\s*(\d{1,2})\s*시\s*(반)?", text)
    if m:
        ampm, h_str, half = m.groups()
        h = int(h_str)

        if ampm == "오후" and h < 12:
            h += 12
        if ampm == "오전" and h == 12:
            h = 0

        mi = 30 if half else 0

        if 0 <= h <= 23:
            return f"{h:02d}:{mi:02d}"

    m = re.search(r"(\d{1,2})\s*시\s*(반)?", text)
    if m:
        h = int(m.group(1))
        mi = 30 if m.group(2) else 0

        if 0 <= h <= 23:
            return f"{h:02d}:{mi:02d}"

        if 1 <= h <= 11:
            return f"{h + 12:02d}:{mi:02d}"

    return None


def parse_time_from_text(text: str, available_times: list[str]) -> Optional[str]:
    candidate = extract_any_time_candidate(text)
    if candidate and candidate in available_times:
        return candidate
    return None


def llm_extract_query(
    text: str,
    pred: pd.DataFrame,
    area_info: pd.DataFrame,
) -> dict:
    available_dates = sorted(pred["date_str"].unique())
    min_date = available_dates[0]
    max_date = available_dates[-1]

    sample_zones = area_info[
        ["생활권역ID", "생활권역표시명", "행정동명목록"]
    ].head(80).to_dict(orient="records")

    system_prompt = f"""
너는 전기차 충전수요 지도 서비스의 자연어 질의 해석기다.
사용자의 한국어 질의를 분석해 intent, 날짜, 시간, 위치를 추출한다.

반드시 JSON만 출력한다.
설명 문장은 출력하지 않는다.

출력 형식:
{{
  "date_text": "사용자가 말한 날짜 표현 또는 null",
  "time_text": "사용자가 말한 시간 표현 또는 null",
  "location_text": "사용자가 말한 위치/동/구/생활권 표현 또는 null",
  "intent": "demand_lookup | service_explanation | greeting | other",
  "needs_data_lookup": true 또는 false
}}

intent 규칙:
- 특정 날짜/시간/위치의 충전수요, 예측, 피크, 순위, 지도 조회를 원하면 demand_lookup.
- 서비스 작동 방식, 데이터, 모델, 지도, 예측 방식, 사용법을 물으면 service_explanation.
- 인사면 greeting.
- 욕설, 장난, 무관한 질문은 other.
- demand_lookup일 때만 needs_data_lookup=true.
- service_explanation, greeting, other는 needs_data_lookup=false.
- 임의로 예측값을 만들지 마라.

예시 규칙:
- 사용자에게 예시를 제안해야 할 경우 반드시 다음 예시만 사용한다:
  "{FIXED_QUERY_EXAMPLE}"
- "내일", "오늘", "오후 3시", "강남구", "서울역" 같은 임의 예시는 절대 만들지 않는다.
- 충전수요 조회 예시는 반드시 연도, 월, 일, 시간, 위치를 모두 포함해야 한다.
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

    if not parsed:
        if re.search(r"어떻게|작동|원리|방식|설명|사용법|뭐야|무엇|데이터|모델|지도", text):
            intent = "service_explanation"
            needs_data_lookup = False
        elif re.search(r"안녕|하이|hello|hi", text, re.IGNORECASE):
            intent = "greeting"
            needs_data_lookup = False
        elif re.search(r"수요|충전|예측|전기차|생활권|지도|보여|알려|조회|분석|혼잡|급증|피크|kwh|kw", text, re.IGNORECASE):
            intent = "demand_lookup"
            needs_data_lookup = True
        else:
            intent = "other"
            needs_data_lookup = False

        return {
            "date_text": None,
            "time_text": None,
            "location_text": None,
            "intent": intent,
            "needs_data_lookup": needs_data_lookup,
            "raw": llm_text,
        }

    intent = parsed.get("intent", "other")
    needs_data_lookup = bool(parsed.get("needs_data_lookup", intent == "demand_lookup"))

    return {
        "date_text": parsed.get("date_text"),
        "time_text": parsed.get("time_text"),
        "location_text": parsed.get("location_text"),
        "intent": intent,
        "needs_data_lookup": needs_data_lookup,
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
        tokens = [t for t in re.split(r"[,\\s·/]+", clean_text(text)) if len(t) >= 2]

        if search_text and any(token in search_text for token in tokens):
            score += 5

        for part in re.split(r"[,·/\\s]+", label):
            part_clean = clean_text(part)
            if len(part_clean) >= 2 and part_clean in q:
                score += 14

        for part in re.split(r"[,·/\\s()_]+", raw_label):
            part_clean = clean_text(part)
            if len(part_clean) >= 2 and part_clean in q:
                score += 12

        for dong in re.split(r"[,·/\\s]+", dongs):
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
    llm_result = llm_extract_query(text, pred, area_info)

    date_text = str(llm_result.get("date_text") or "").strip()
    time_text = str(llm_result.get("time_text") or "").strip()
    location_text = str(llm_result.get("location_text") or "").strip()
    intent = str(llm_result.get("intent") or "other").strip()
    needs_data_lookup = bool(llm_result.get("needs_data_lookup", False))

    has_demand_keyword = bool(
        re.search(
            r"수요|충전|예측|전기차|생활권|지도|보여|알려|조회|분석|혼잡|급증|피크|kwh|kw",
            text,
            re.IGNORECASE,
        )
    )

    has_any_condition = bool(
        extract_any_date_candidate(text)
        or extract_any_time_candidate(text)
        or find_zone_by_location(text, area_info)
        or location_text
    )

    if not needs_data_lookup and not has_demand_keyword:
        return {
            "ok": False,
            "reason": intent if intent else "conversational",
            "message": "사용자의 입력은 데이터 조회 요청이 아니라 일반 대화 또는 서비스 설명 요청입니다.",
            "date": fallback_date,
            "time": fallback_time,
            "zone_id": fallback_zone_id,
            "llm_extract": llm_result,
        }

    if has_demand_keyword and not has_any_condition:
        return {
            "ok": False,
            "reason": "missing_conditions",
            "message": (
                "충전수요 예측을 조회하려면 연도, 월, 일, 시간, 위치가 모두 필요합니다. "
                f"예: {FIXED_QUERY_EXAMPLE}"
            ),
            "date": fallback_date,
            "time": fallback_time,
            "zone_id": fallback_zone_id,
            "llm_extract": llm_result,
        }

    date_source = date_text or text
    any_date = extract_any_date_candidate(date_source) or extract_any_date_candidate(text)

    if any_date and any_date not in available_dates:
        return {
            "ok": False,
            "reason": "date_unavailable",
            "message": (
                f"{any_date} 날짜는 현재 예측 데이터셋에 존재하지 않습니다. "
                f"현재 조회 가능한 날짜 범위는 {available_dates[0]}부터 {available_dates[-1]}까지입니다. "
                f"예: {FIXED_QUERY_EXAMPLE}"
            ),
            "date": fallback_date,
            "time": fallback_time,
            "zone_id": fallback_zone_id,
            "llm_extract": llm_result,
        }

    parsed_date = parse_date_from_text(date_source, available_dates)
    if parsed_date is None:
        parsed_date = parse_date_from_text(text, available_dates)

    if parsed_date is None:
        return {
            "ok": False,
            "reason": "missing_date",
            "message": (
                "조회할 날짜를 찾지 못했습니다. 연도, 월, 일을 모두 포함해 입력해 주세요. "
                f"현재 데이터는 {available_dates[0]}부터 {available_dates[-1]}까지 조회할 수 있습니다. "
                f"예: {FIXED_QUERY_EXAMPLE}"
            ),
            "date": fallback_date,
            "time": fallback_time,
            "zone_id": fallback_zone_id,
            "llm_extract": llm_result,
        }

    available_times = sorted(pred[pred["date_str"] == parsed_date]["time_str"].unique())

    time_source = time_text or text
    any_time = extract_any_time_candidate(time_source) or extract_any_time_candidate(text)

    if any_time and any_time not in available_times:
        return {
            "ok": False,
            "reason": "time_unavailable",
            "message": (
                f"{parsed_date} {any_time} 시간대는 현재 예측 데이터셋에 존재하지 않습니다. "
                "현재 서비스는 예측 파일에 포함된 30분 단위 시간대만 조회할 수 있습니다. "
                f"예: {FIXED_QUERY_EXAMPLE}"
            ),
            "date": fallback_date,
            "time": fallback_time,
            "zone_id": fallback_zone_id,
            "llm_extract": llm_result,
        }

    parsed_time = parse_time_from_text(time_source, available_times)
    if parsed_time is None:
        parsed_time = parse_time_from_text(text, available_times)

    if parsed_time is None:
        return {
            "ok": False,
            "reason": "missing_time",
            "message": (
                "조회할 시간을 찾지 못했습니다. 연도, 월, 일, 시간, 위치를 모두 포함해 입력해 주세요. "
                f"예: {FIXED_QUERY_EXAMPLE}"
            ),
            "date": fallback_date,
            "time": fallback_time,
            "zone_id": fallback_zone_id,
            "llm_extract": llm_result,
        }

    location_source = " ".join([location_text, text]).strip()
    parsed_zone_id = find_zone_by_location(location_source, area_info)

    if parsed_zone_id is None:
        return {
            "ok": False,
            "reason": "location_unavailable",
            "message": (
                "입력한 위치는 현재 서울시 생활권 데이터셋에서 찾을 수 없습니다. "
                "서울시 행정동, 자치구, 생활권 이름을 기준으로 다시 입력해 주세요. "
                f"예: {FIXED_QUERY_EXAMPLE}"
            ),
            "date": fallback_date,
            "time": fallback_time,
            "zone_id": fallback_zone_id,
            "llm_extract": llm_result,
        }

    return {
        "ok": True,
        "reason": "ok",
        "message": "",
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
                border: none;
                box-shadow: none;
            }}

            .deck-tooltip {{
                font-size: 12px !important;
                border-radius: 13px !important;
                padding: 10px 12px !important;
                background: rgba(17, 17, 17, 0.92) !important;
                color: #FFFFFF !important;
                box-shadow: none !important;
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
def draw_alerts_stack(top_df: pd.DataFrame, selected_time: str):
    if top_df.empty:
        st.info("수요 알림을 생성할 수 없습니다.")
        return

    # 화면에는 4개가 보이고, 로테이션은 top 8개를 순환
    alert_rows = top_df.head(8).copy()
    cards_html = ""

    for i, row in enumerate(alert_rows.itertuples(), start=1):
        label = getattr(row, "생활권역표시명", getattr(row, "생활권역ID"))
        value = float(getattr(row, "predicted_kwh"))

        if i == 1:
            state = "PEAK"
            state_kr = "급증"
            state_class = "hot"
            title = "가장 높은 충전수요가 예상됩니다"
        elif i <= 3:
            state = "WATCH"
            state_kr = "주의"
            state_class = "watch"
            title = "충전기 운영 여유를 확인하세요"
        else:
            state = "MONITOR"
            state_kr = "관찰"
            state_class = "monitor"
            title = "수요 변화를 함께 모니터링하세요"

        cards_html += f"""
        <div class="ev-alert-card">
            <div class="alert-left">
                <div class="state-circle {state_class}">
                    <span>{state_kr}</span>
                </div>
            </div>

            <div class="alert-center">
                <div class="alert-brand {state_class}">{state}</div>
                <div class="alert-zone">{escape_html(label)}</div>
                <div class="alert-copy">{escape_html(title)}</div>
            </div>

            <div class="alert-right">
                <div class="right-label">예측수요</div>
                <div class="right-value">{value:,.1f} kWh</div>
                <div class="right-label time">시간</div>
                <div class="right-value">{selected_time}</div>
            </div>
        </div>
        """

    loop_cards_html = cards_html + cards_html
    card_count = len(alert_rows)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8" />
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Holtwood+One+SC&display=swap');

        html, body {{
            margin: 0;
            padding: 0;
            background: transparent;
            font-family: Inter, -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Noto Sans KR", sans-serif;
            overflow: hidden;
        }}

        .alert-stack-panel {{
            position: relative;
            height: 508px;
            overflow: hidden;
            box-sizing: border-box;
            padding: 4px 2px 0 0;
            background: transparent;
        }}

        .alert-scroll-track {{
            will-change: transform;
            transform: translateY(0);
        }}

        .ev-alert-card {{
            height: 112px;
            display: grid;
            grid-template-columns: 62px minmax(0, 1fr) 106px;
            align-items: center;
            gap: 13px;
            background: #FFFFFF;
            border: 1.1px solid rgba(20, 20, 20, 0.20);
            border-radius: 18px;
            padding: 14px 14px;
            box-sizing: border-box;
            margin-bottom: 14px;
            box-shadow: none;
        }}

        .state-circle {{
            width: 55px;
            height: 55px;
            border-radius: 999px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #FFFFFF;
            font-size: 13px;
            font-weight: 800;
            letter-spacing: -0.035em;
            box-shadow: none;
        }}

        .state-circle.hot {{
            background: #FF3F4F;
        }}

        .state-circle.watch {{
            background: #F5A000;
        }}

        .state-circle.monitor {{
            background: #657386;
        }}

        .alert-center {{
            min-width: 0;
        }}

        .alert-brand {{
            font-family: "Holtwood One SC", Georgia, serif;
            font-size: 11px;
            font-weight: 400;
            letter-spacing: -0.02em;
            line-height: 1;
            margin-bottom: 8px;
        }}

        .alert-brand.hot {{
            color: #FF3F4F;
        }}

        .alert-brand.watch {{
            color: #F5A000;
        }}

        .alert-brand.monitor {{
            color: #657386;
        }}

        .alert-zone {{
            color: #111111;
            font-size: 15px;
            font-weight: 850;
            letter-spacing: -0.045em;
            line-height: 1.25;
            word-break: keep-all;
            white-space: normal;
        }}

        .alert-copy {{
            margin-top: 6px;
            color: #555C66;
            font-size: 11.4px;
            font-weight: 500;
            line-height: 1.35;
            word-break: keep-all;
            white-space: normal;
        }}

        .alert-right {{
            display: grid;
            justify-items: end;
            align-items: center;
        }}

        .right-label {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 70px;
            height: 21px;
            border-radius: 999px;
            background: #9A9A9A;
            color: #FFFFFF;
            font-size: 10px;
            font-weight: 650;
            line-height: 1;
        }}

        .right-label.time {{
            margin-top: 8px;
        }}

        .right-value {{
            color: #111111;
            font-size: 13px;
            font-weight: 750;
            line-height: 1.15;
            margin-top: 5px;
            white-space: nowrap;
            letter-spacing: -0.02em;
        }}
    </style>
    </head>
    <body>
        <div class="alert-stack-panel" id="alertPanel">
            <div class="alert-scroll-track" id="alertTrack">
                {loop_cards_html}
            </div>
        </div>

        <script>
            const panel = document.getElementById("alertPanel");
            const track = document.getElementById("alertTrack");
            const originalCount = {card_count};

            let index = 0;
            let isPausedByHover = false;

            const MOVE_DURATION = 650;
            const HOLD_DURATION = 1900;

            function getCards() {{
                return Array.from(track.querySelectorAll(".ev-alert-card"));
            }}

            function getOffsetForIndex(targetIndex) {{
                const cards = getCards();
                if (!cards[targetIndex]) return 0;
                return cards[targetIndex].offsetTop;
            }}

            function moveTo(targetIndex, withTransition = true) {{
                const offset = getOffsetForIndex(targetIndex);

                if (withTransition) {{
                    track.style.transition = `transform ${{MOVE_DURATION}}ms cubic-bezier(0.22, 1, 0.36, 1)`;
                }} else {{
                    track.style.transition = "none";
                }}

                track.style.transform = `translateY(-${{offset}}px)`;
            }}

            function resetIfNeeded() {{
                if (index >= originalCount) {{
                    index = 0;
                    moveTo(0, false);
                    void track.offsetHeight;
                }}
            }}

            function tick() {{
                if (isPausedByHover) {{
                    setTimeout(tick, HOLD_DURATION);
                    return;
                }}

                index += 1;
                moveTo(index, true);

                setTimeout(() => {{
                    resetIfNeeded();
                    setTimeout(tick, HOLD_DURATION);
                }}, MOVE_DURATION);
            }}

            panel.addEventListener("mouseenter", () => {{
                isPausedByHover = true;
            }});

            panel.addEventListener("mouseleave", () => {{
                isPausedByHover = false;
            }});

            setTimeout(tick, HOLD_DURATION);
        </script>
    </body>
    </html>
    """

    components.html(html, height=508, scrolling=False)


def build_selected_detail_html(
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
) -> str:
    return f"""
    <div class="selected-detail-card">
        <div class="detail-card-header">
            <div>
                <div class="detail-card-kicker">SELECTED AREA</div>
                <div class="detail-card-title">{escape_html(selected_label)}</div>
                <div class="detail-card-id">{escape_html(selected_zone_id)}</div>
            </div>
        </div>

        <div class="detail-card-meta">
            <div class="meta-block">
                <div class="meta-label">조회 시각</div>
                <div class="meta-text">{escape_html(selected_date)} {escape_html(selected_time)}</div>
            </div>
            <div class="meta-block">
                <div class="meta-label">포함 행정동</div>
                <div class="meta-text">{escape_html(selected_dongs if selected_dongs else "행정동 정보 없음")}</div>
            </div>
        </div>

        <div class="detail-metric-grid">
            <div class="detail-metric">
                <div class="metric-label">현재 예측</div>
                <div class="metric-value">{zone_pred_kwh:.1f} kWh</div>
            </div>
            <div class="detail-metric">
                <div class="metric-label">수요 순위</div>
                <div class="metric-value">{int(zone_rank)} / {n_zones}</div>
            </div>
            <div class="detail-metric">
                <div class="metric-label">일일 총량</div>
                <div class="metric-value">{total_day_kwh:.0f} kWh</div>
            </div>
            <div class="detail-metric">
                <div class="metric-label">피크 시간</div>
                <div class="metric-value">{escape_html(peak_time)}</div>
                <div class="metric-delta">↑ {peak_kwh:.1f} kWh</div>
            </div>
        </div>
    </div>
    """


def render_chat_panel(
    messages: list[dict],
    selected_detail_html: str | None = None,
) -> None:
    items_html = ""

    visible_messages = messages

    for msg in visible_messages:
        role = msg.get("role", "assistant")
        content = escape_html(msg.get("content", ""))
        role_class = "user" if role == "user" else "assistant"

        items_html += f"""
        <div class="chat-bubble-row {role_class}">
            <div class="chat-bubble {role_class}">{content}</div>
        </div>
        """

    if selected_detail_html:
        items_html += selected_detail_html

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8" />
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Holtwood+One+SC&display=swap');

        html, body {{
            margin: 0;
            padding: 0;
            background: transparent;
            font-family: Inter, -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Noto Sans KR", sans-serif;
            overflow: hidden;
        }}

        .chat-scroll-box {{
            height: {CHAT_SCROLL_HEIGHT}px;
            overflow-y: auto;
            overflow-x: hidden;
            padding: 8px 4px 8px 2px;
            box-sizing: border-box;
            border-top: 1px solid rgba(20, 20, 20, 0.08);
            border-bottom: 1px solid rgba(20, 20, 20, 0.08);
            background: transparent;
            scroll-behavior: smooth;
        }}

        .chat-scroll-box::-webkit-scrollbar {{
            width: 7px;
        }}

        .chat-scroll-box::-webkit-scrollbar-thumb {{
            background: rgba(120, 130, 145, 0.45);
            border-radius: 999px;
        }}

        .chat-scroll-box::-webkit-scrollbar-track {{
            background: transparent;
        }}

        .chat-bubble-row {{
            display: flex;
            margin-bottom: 11px;
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
            font-size: 12px;
            font-weight: 560;
            line-height: 1.58;
            word-break: keep-all;
            overflow-wrap: anywhere;
            box-sizing: border-box;
        }}

        .chat-bubble.user {{
            background: #1F78B4;
            color: #FFFFFF;
            border-bottom-right-radius: 5px;
            box-shadow: none;
        }}

        .chat-bubble.assistant {{
            background: #FFFFFF;
            color: #111111;
            border: 1px solid rgba(20, 20, 20, 0.20);
            border-bottom-left-radius: 5px;
            box-shadow: none;
        }}

        .selected-detail-card {{
            margin: 13px 2px 8px 2px;
            padding: 14px;
            border-radius: 18px;
            border: 1.1px solid rgba(20, 20, 20, 0.22);
            background: #FFFFFF;
            box-shadow: none;
            box-sizing: border-box;
        }}

        .detail-card-kicker {{
            color: #1F78B4;
            font-family: "Holtwood One SC", Georgia, serif;
            font-size: 11px;
            font-weight: 400;
            letter-spacing: -0.02em;
            margin-bottom: 5px;
        }}

        .detail-card-title {{
            color: #111111;
            font-size: 18px;
            font-weight: 850;
            letter-spacing: -0.045em;
            line-height: 1.25;
        }}

        .detail-card-id {{
            color: #5F666F;
            font-size: 11px;
            font-weight: 600;
            margin-top: 4px;
        }}

        .detail-card-meta {{
            margin-top: 11px;
            display: flex;
            flex-direction: column;
            gap: 7px;
        }}

        .meta-label {{
            color: #777D86;
            font-size: 10.5px;
            font-weight: 650;
            margin-bottom: 2px;
        }}

        .meta-text {{
            color: #333333;
            font-size: 11.5px;
            font-weight: 500;
            line-height: 1.42;
            word-break: keep-all;
            overflow-wrap: anywhere;
        }}

        .detail-metric-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-top: 12px;
        }}

        .detail-metric {{
            background: #FAFAFA;
            border: 1px solid rgba(20, 20, 20, 0.12);
            border-radius: 13px;
            padding: 9px 10px;
            min-height: 58px;
            box-sizing: border-box;
            box-shadow: none;
        }}

        .metric-label {{
            color: #777D86;
            font-size: 10.5px;
            font-weight: 600;
            margin-bottom: 4px;
        }}

        .metric-value {{
            color: #111111;
            font-size: 17px;
            font-weight: 820;
            letter-spacing: -0.04em;
        }}

        .metric-delta {{
            display: inline-flex;
            margin-top: 4px;
            padding: 2px 6px;
            border-radius: 999px;
            background: #EAF4FA;
            color: #1F78B4;
            font-size: 10.5px;
            font-weight: 700;
        }}
    </style>
    </head>
    <body>
        <div class="chat-scroll-box" id="chatbox">
            {items_html}
        </div>

        <script>
            const box = document.getElementById("chatbox");
            if (box) {{
                box.scrollTop = box.scrollHeight;
            }}
        </script>
    </body>
    </html>
    """

    components.html(html, height=CHAT_SCROLL_HEIGHT + 8, scrolling=False)


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
        f"선택 시각 예측 충전수요는 {zone_pred_kwh:.1f} kWh이며, "
        f"전체 {n_zones}개 생활권 중 {int(zone_rank)}위입니다.\n"
        f"해당 날짜의 총 예측 충전량은 {total_day_kwh:.0f} kWh이고, "
        f"피크 시간은 {peak_time}로 예상됩니다.\n\n"
        f"지도에서는 해당 생활권을 강조하고, 상세 정보는 이 채팅 패널에 표시했습니다."
    )


def build_conversational_answer(
    user_text: str,
    reason_message: str,
    reason: str,
    llm_extract: dict | None,
    messages: list[dict],
    pred: pd.DataFrame,
) -> str:
    available_dates = sorted(pred["date_str"].unique())
    chat_context = get_recent_chat_context(messages)

    system_prompt = f"""
너는 서울시 생활권별 전기차 충전수요 예측 서비스 E-Vlog의 대화형 LLM '모도리'다.

너는 단순 안내 챗봇이 아니라, 사용자의 질문 의도를 이해해서 자연스럽게 답해야 한다.
다만 예측 수치, 순위, 피크 시간 등 데이터 기반 정보는 제공된 데이터에서 조회된 경우에만 말할 수 있다.

답변 규칙:
- 사용자가 서비스 작동 방식, 데이터, 모델, 지도, 예측 방식에 대해 물으면 구체적으로 설명한다.
- 사용자가 인사하면 자연스럽게 인사한다.
- 사용자가 장난, 욕설, 무관한 말을 하면 짧게 받아주고 서비스 맥락으로 부드럽게 돌린다.
- 사용자가 날짜, 시간, 위치 없이 수요를 물으면 부족한 항목만 되묻는다.
- 매번 같은 안내문을 반복하지 않는다.
- 한국어로 답변한다.
- 3~7문장 정도로 답변한다.
- 예측값은 절대 임의로 만들지 않는다.

예시 규칙:
- 사용자에게 예시를 제안해야 할 경우 반드시 다음 예시만 사용한다:
  "{FIXED_QUERY_EXAMPLE}"
- "내일", "오늘", "오후 3시", "강남구", "서울역" 같은 임의 예시는 절대 만들지 않는다.
- 충전수요 조회에는 연도, 월, 일, 시간, 위치가 모두 필요하다고 안내한다.
"""

    user_prompt = f"""
사용자 입력:
{user_text}

최근 대화:
{json.dumps(chat_context, ensure_ascii=False, indent=2)}

LLM 해석 결과:
{json.dumps(llm_extract or {}, ensure_ascii=False, indent=2)}

데이터 검증 결과:
- reason: {reason}
- message: {reason_message}

서비스 정보:
- 이 서비스는 서울시 생활권별 전기차 충전수요를 예측하는 서비스다.
- 현재 지도는 {MODEL_DISPLAY_NAME}의 예측 결과 파일을 사용한다.
- 사용자는 날짜, 시간, 위치를 자연어로 입력할 수 있다.
- 정확한 조회를 위해서는 연도, 월, 일, 시간, 위치가 모두 필요하다.
- 시스템은 사용자의 문장에서 날짜, 시간, 위치를 해석한다.
- 이후 앱에 포함된 예측 파일에서 해당 조건의 predicted_kwh 값을 조회한다.
- 지도는 해당 생활권을 확대하고, 3D 막대로 수요 크기를 시각화한다.
- 예측 데이터 날짜 범위는 {available_dates[0]}부터 {available_dates[-1]}까지다.
- 30분 단위 시간대만 조회할 수 있다.
- 서울시 생활권 경계 데이터에 포함된 위치만 조회할 수 있다.
- 사용 가능한 예시는 반드시 "{FIXED_QUERY_EXAMPLE}" 하나만 사용한다.

사용자의 질문에 맞게 자연스럽게 답변하라.
"""

    llm_answer = call_gemini_text(system_prompt, user_prompt, temperature=0.65)

    if llm_answer and llm_answer.strip():
        return llm_answer.strip()

    if reason == "service_explanation" or re.search(r"어떻게|작동|원리|방식|설명|사용법|데이터|모델|지도", user_text):
        return (
            "이 서비스는 사용자가 입력한 연도, 월, 일, 시간, 위치를 먼저 해석한 뒤, "
            f"{MODEL_DISPLAY_NAME}의 예측 결과 파일에서 해당 조건의 충전수요 값을 찾아 보여주는 방식으로 작동합니다.\n\n"
            f"예를 들어 `{FIXED_QUERY_EXAMPLE}`라고 입력하면, "
            "날짜는 2025-11-25, 시간은 18:00, 위치는 청운효자동이 포함된 생활권으로 변환됩니다. "
            "그 다음 해당 생활권의 예측 kWh, 수요 순위, 피크 시간 등을 지도와 함께 보여줍니다."
        )

    if reason == "greeting" or re.search(r"안녕|하이|hello|hi", user_text, re.IGNORECASE):
        return (
            "안녕하세요. 저는 모도리입니다.\n\n"
            "서울시 생활권별 전기차 충전수요를 예측 데이터에서 찾아 설명해 드릴 수 있습니다. "
            "정확한 조회를 위해 연도, 월, 일, 시간, 위치를 함께 입력해 주세요.\n\n"
            f"예: {FIXED_QUERY_EXAMPLE}"
        )

    if reason in {"missing_conditions", "missing_date", "missing_time", "location_unavailable", "date_unavailable", "time_unavailable"}:
        return (
            f"{reason_message}\n\n"
            f"예: {FIXED_QUERY_EXAMPLE}"
        )

    return (
        "저는 전기차 충전수요 예측을 도와주는 모도리입니다.\n\n"
        "잡담도 이해할 수 있지만, 가장 잘할 수 있는 일은 서울시 생활권별 충전수요를 날짜와 시간 기준으로 찾아 설명하는 것입니다. "
        "정확한 조회를 위해 연도, 월, 일, 시간, 위치를 함께 입력해 주세요.\n\n"
        f"예: {FIXED_QUERY_EXAMPLE}"
    )


def build_invalid_answer(
    user_text: str,
    reason_message: str,
    reason: str,
    llm_extract: dict | None,
    messages: list[dict],
    pred: pd.DataFrame,
) -> str:
    return build_conversational_answer(
        user_text=user_text,
        reason_message=reason_message,
        reason=reason,
        llm_extract=llm_extract,
        messages=messages,
        pred=pred,
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
    messages: list[dict],
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
        "model": MODEL_DISPLAY_NAME,
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

    chat_context = get_recent_chat_context(messages)

    system_prompt = f"""
너는 E-Vlog 서비스의 전기차 충전수요 분석 LLM '모도리'다.
너는 실제 LLM처럼 자연스럽게 대화하되, 수요 예측 수치와 관련된 내용은 반드시 제공된 facts만 근거로 답변한다.

중요 규칙:
- 현재 예측 결과는 {MODEL_DISPLAY_NAME} 기반이다.
- facts에 없는 예측값, 순위, 원인, 외부 사건은 절대 만들지 않는다.
- 사용자의 질문에 직접 답한다.
- 사용자가 짧게 물어봐도 자연스럽게 맥락을 보완해 설명한다.
- 사용자가 후속 질문을 하면 최근 대화 맥락을 참고한다.
- 지도에서 해당 생활권이 강조된다는 점을 필요할 때만 언급한다.
- 한국어로 답변한다.
- 4~8문장 정도로 답변한다.
- 수치는 facts 값 그대로 사용한다.

예시 규칙:
- 예시를 제안해야 할 경우 반드시 다음 예시만 사용한다:
  "{FIXED_QUERY_EXAMPLE}"
- "내일", "오늘", "오후 3시", "강남구", "서울역" 같은 임의 예시는 절대 만들지 않는다.
"""

    user_prompt = f"""
최근 대화:
{json.dumps(chat_context, ensure_ascii=False, indent=2)}

사용자 질문:
{user_text}

facts:
{json.dumps(facts, ensure_ascii=False, indent=2)}

위 facts만 사용해서 자연어로 답변하라.
"""

    llm_answer = call_gemini_text(system_prompt, user_prompt, temperature=0.45)

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
    pred, pred_col, true_col = load_predictions(
        PRED_FILE_PATH,
        meta,
        TRUE_FILE_PATH,
    )
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

if "show_selected_detail" not in st.session_state:
    st.session_state.show_selected_detail = False

if "animate_zoom" not in st.session_state:
    st.session_state.animate_zoom = False

if "pending_user_query" not in st.session_state:
    st.session_state.pending_user_query = None

if "pending_invalid_query" not in st.session_state:
    st.session_state.pending_invalid_query = None

if "last_llm_error" not in st.session_state:
    st.session_state.last_llm_error = None

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "안녕하세요. 저는 모도리입니다.\n\n"
                "보고 싶은 연도, 월, 일, 시간, 위치를 자연어로 입력하면 "
                f"{MODEL_DISPLAY_NAME}의 예측 결과 파일을 조회해 해당 생활권의 충전수요를 알려드립니다.\n\n"
                f"예: {FIXED_QUERY_EXAMPLE}"
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
if st.session_state.pending_invalid_query:
    pending = st.session_state.pending_invalid_query
    answer = build_invalid_answer(
        user_text=pending["user_text"],
        reason_message=pending["reason_message"],
        reason=pending.get("reason", "invalid"),
        llm_extract=pending.get("llm_extract"),
        messages=st.session_state.messages,
        pred=pred,
    )
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.pending_invalid_query = None
    st.session_state.show_selected_detail = False

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
        messages=st.session_state.messages,
    )
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.pending_user_query = None
    st.session_state.show_selected_detail = True


# =========================================================
# 채팅 패널에 표시할 선택 생활권 상세 HTML
# =========================================================
selected_detail_html = None

if st.session_state.has_query and st.session_state.show_selected_detail:
    selected_detail_html = build_selected_detail_html(
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


# =========================================================
# 메인 3분할 레이아웃
# =========================================================
alert_col, map_col, chat_col = st.columns([0.86, 1.42, 0.78], gap="small")


# =========================================================
# 1. 왼쪽: 수요 급증 알림
# =========================================================
with alert_col:
    with st.container(border=True, height=PANEL_HEIGHT):
        mark_panel()

        panel_title(
            "수요 급증알림",
            "선택 시각 기준 충전수요가 높은 생활권을 보여줍니다.",
            kicker="DEMAND WATCH",
        )

        draw_alerts_stack(top10, selected_time)


# =========================================================
# 2. 가운데: 지도
# =========================================================
with map_col:
    with st.container(border=True, height=PANEL_HEIGHT):
        mark_panel()

        header_left, header_right = st.columns([0.72, 0.28], gap="small")

        with header_left:
            panel_title(
                "충전수요지도",
                (
                    f"{selected_dt:%Y-%m-%d %H:%M} · {MODEL_DISPLAY_NAME} 기반 서울시 생활권별 예측 충전수요"
                ),
                kicker="E-VLOG MAP",
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
        mark_panel()

        panel_title(
            "MODORI",
            "연도, 월, 일, 시간, 위치를 자연어로 입력하세요.",
            kicker="AI ASSISTANT",
        )

        render_chat_panel(
            messages=st.session_state.messages,
            selected_detail_html=selected_detail_html,
        )

        with st.form("chat_form", clear_on_submit=True):
            user_text = st.text_input(
                "질문 입력",
                placeholder=f"예: {FIXED_QUERY_EXAMPLE}",
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

            if not parsed["ok"]:
                st.session_state.pending_invalid_query = {
                    "user_text": clean_user_text,
                    "reason_message": parsed["message"],
                    "reason": parsed["reason"],
                    "llm_extract": parsed.get("llm_extract"),
                }

                st.session_state.show_selected_detail = False
                st.session_state.animate_zoom = False

                st.rerun()

            if st.session_state.has_query:
                st.session_state.previous_focus_zone_id = st.session_state.selected_zone_id
            else:
                st.session_state.previous_focus_zone_id = None

            st.session_state.selected_date = parsed["date"]
            st.session_state.selected_time = parsed["time"]
            st.session_state.selected_zone_id = parsed["zone_id"]
            st.session_state.has_query = True
            st.session_state.show_selected_detail = True
            st.session_state.animate_zoom = True
            st.session_state.pending_user_query = clean_user_text

            st.rerun()

        if st.session_state.get("last_llm_error"):
            st.markdown(
                f"""
                <div class="llm-error-caption">
                LLM error: {escape_html(st.session_state.last_llm_error)}
                </div>
                """,
                unsafe_allow_html=True,
            )
