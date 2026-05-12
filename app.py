# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
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

# GitHub에는 아래 두 파일명을 권장합니다.
# 다른 이름으로 올린 경우, 각 후보명 중 실제 존재하는 파일을 자동 선택합니다.
PRED_FILE_CANDIDATES = {
    "overall": [
        BASE_DIR / "predictions_v5-overall.npz",
        BASE_DIR / "predictions_overall.npz",
        BASE_DIR / "overall_predictions.npz",
        BASE_DIR / "preds_overall.npy",
        BASE_DIR / "preds.npy",
    ],
    "peak": [
        BASE_DIR / "predictions_v5-peak.npz",
        BASE_DIR / "predictions_peak.npz",
        BASE_DIR / "peak_predictions.npz",
        BASE_DIR / "preds_peak.npy",
    ],
}

MODEL_LABELS = {
    "overall": "일반 모델",
    "peak": "피크 모델",
}
MODEL_DISPLAY_NAME = "E-Vlog"

META_JSON_PATH = BASE_DIR / "meta.json"
AREA_EXCEL_PATH = BASE_DIR / "area_info.xlsx"
SHP_PATH = BASE_DIR / "UPIS_SHP_ZON100.shp"

SOURCE_EPSG = 5174
TARGET_EPSG = 4326
START_DATETIME = pd.Timestamp("2025-01-01 00:00:00")
TIME_UNIT_MINUTES = 30

DEFAULT_DATE = "2025-11-25"
DEFAULT_TIME = "18:00"
FIXED_QUERY_EXAMPLE = "2025년 11월 25일 오후 6시에 청운효자동 수요 보여줘"

PANEL_HEIGHT = 625
MAP_HEIGHT = 485
CHAT_SCROLL_HEIGHT = 365
ALERT_HEIGHT = 508

# 입력 시각 기준 30분 후 예측을 지도/알림에 표시
MAP_HORIZON = 1
PREDICT_HORIZON_30M = 1
PEAK_WINDOW_HORIZONS = 12  # 6시간 = 30분 * 12

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
        background: rgba(255,255,255,0) !important;
        height: 0rem;
    }

    div[data-testid="stToolbar"], div[data-testid="stStatusWidget"], div[data-testid="stDecoration"] {
        visibility: hidden !important;
        display: none !important;
        height: 0 !important;
    }

    #MainMenu, footer, section[data-testid="stSidebar"] {
        display: none !important;
        visibility: hidden !important;
    }

    div[data-testid="stVerticalBlock"] {
        gap: 0.42rem;
        overflow: visible !important;
    }

    div[data-testid="column"] {
        padding: 10px 8px !important;
        overflow: visible !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        overflow: visible !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.ev-panel-marker) {
        background: #FFFFFF !important;
        border-radius: 22px !important;
        border: 1.6px solid rgba(20, 20, 20, 0.56) !important;
        box-shadow:
            0 26px 44px rgba(0, 0, 0, 0.22),
            0 9px 18px rgba(0, 0, 0, 0.13),
            0 1px 0 rgba(255, 255, 255, 0.96) inset !important;
        overflow: visible !important;
        position: relative !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.ev-panel-marker) > div {
        overflow: visible !important;
    }

    .ev-panel-marker { display: none; }

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

    div[data-testid="stTextInput"] { margin-bottom: 0rem; }
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

    div[data-testid="stRadio"] label {
        color: #111111 !important;
        font-weight: 760 !important;
        font-size: 11px !important;
    }

    div[data-testid="stToggle"] {
        margin-top: 0.1rem;
        min-width: max-content !important;
    }

    div[data-testid="stToggle"] label {
        color: #111111 !important;
        font-weight: 700 !important;
        white-space: nowrap !important;
        word-break: keep-all !important;
        overflow-wrap: normal !important;
        min-width: max-content !important;
    }

    div[data-testid="stToggle"] label p {
        white-space: nowrap !important;
        word-break: keep-all !important;
        overflow-wrap: normal !important;
        min-width: max-content !important;
    }

    .chat-model-toggle-wrap {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        min-width: 90px;
        white-space: nowrap;
    }

    .chat-form-wrap {
        width: 100%;
        box-sizing: border-box;
        overflow: hidden;
        padding-right: 2px;
    }

    div[data-testid="stToggle"] p {
        font-size: 12px !important;
        font-weight: 700 !important;
        color: #111111 !important;
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
    subtitle_html = f'<div class="panel-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        {kicker_html}
        <div class="panel-title">{title}</div>
        {subtitle_html}
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


def escape_html(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )


# =========================================================
# 기본 유틸
# =========================================================
def normalize_id(value: Any) -> str:
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


def kwh_to_color(value: float, vmin: float, vmax: float) -> list[int]:
    if pd.isna(value):
        return [220, 225, 232, 90]
    if vmax <= vmin:
        ratio = 0.5
    else:
        ratio = float(np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0))
    r = int(205 - 155 * ratio)
    g = int(222 - 135 * ratio)
    b = int(247 - 25 * ratio)
    return [r, g, b, 190]


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
        {"role": m.get("role", "assistant"), "content": str(m.get("content", ""))[:700]}
        for m in recent
    ]


# =========================================================
# Gemini API 함수
# =========================================================
def get_gemini_client() -> Optional[Any]:
    if genai is None:
        st.session_state.last_llm_error = "google-genai 패키지를 불러오지 못했습니다. requirements.txt에 google-genai를 추가하세요."
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


def call_gemini_text(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> Optional[str]:
    client = get_gemini_client()
    if client is None:
        return None

    prompt = f"""
[System]
{system_prompt}

[User]
{user_prompt}
"""

    try:
        response = client.models.generate_content(
            model=get_gemini_model(),
            contents=prompt,
            config={"temperature": temperature},
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


# =========================================================
# 예측 파일 저장소: 메모리 폭발 방지용
# =========================================================
@dataclass
class PredictionStore:
    path: str
    pred: np.ndarray  # shape: (N, H, Z)
    true: Optional[np.ndarray]  # shape: (N, H, Z) or None
    zone_ids: list[str]
    val_end: int
    look_back: int

    @property
    def n_samples(self) -> int:
        return int(self.pred.shape[0])

    @property
    def n_horizons(self) -> int:
        return int(self.pred.shape[1])

    @property
    def n_zones(self) -> int:
        return int(self.pred.shape[2])

    @property
    def offset(self) -> int:
        return self.val_end + self.look_back

    def sample_to_datetime(self, sample_idx: int) -> pd.Timestamp:
        return START_DATETIME + pd.to_timedelta((int(sample_idx) + self.offset) * TIME_UNIT_MINUTES, unit="m")

    def target_datetime(self, sample_idx: int, horizon: int) -> pd.Timestamp:
        return START_DATETIME + pd.to_timedelta((int(sample_idx) + self.offset + int(horizon)) * TIME_UNIT_MINUTES, unit="m")

    def date_range(self) -> tuple[str, str]:
        first = self.sample_to_datetime(0).strftime("%Y-%m-%d")
        last = self.sample_to_datetime(self.n_samples - 1).strftime("%Y-%m-%d")
        return first, last

    def available_dates(self) -> list[str]:
        first_dt = self.sample_to_datetime(0).normalize()
        last_dt = self.sample_to_datetime(self.n_samples - 1).normalize()
        return pd.date_range(first_dt, last_dt, freq="D").strftime("%Y-%m-%d").tolist()

    def date_time_to_sample_idx(self, date_str: str, time_str: str) -> Optional[int]:
        dt = pd.Timestamp(f"{date_str} {time_str}")
        minutes = int((dt - START_DATETIME).total_seconds() // 60)
        if minutes % TIME_UNIT_MINUTES != 0:
            return None
        global_idx = minutes // TIME_UNIT_MINUTES
        sample_idx = int(global_idx - self.offset)
        if 0 <= sample_idx < self.n_samples:
            return sample_idx
        return None

    def available_times_for_date(self, date_str: str) -> list[str]:
        # 대부분 30분 단위 전체가 가능하지만, 시작/끝 날짜는 실제 sample 범위로 제한
        out = []
        for slot in range(48):
            dt = pd.Timestamp(date_str) + pd.to_timedelta(slot * TIME_UNIT_MINUTES, unit="m")
            sample_idx = self.date_time_to_sample_idx(dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M"))
            if sample_idx is not None:
                out.append(dt.strftime("%H:%M"))
        return out

    def horizon_safe(self, horizon: int) -> int:
        return int(min(max(horizon, 0), self.n_horizons - 1))

    def zone_index(self, zone_id: str) -> int:
        return self.zone_ids.index(zone_id)

    def pred_value(self, sample_idx: int, horizon: int, zone_idx: int) -> float:
        h = self.horizon_safe(horizon)
        return float(self.pred[int(sample_idx), h, int(zone_idx)])

    def true_value(self, sample_idx: int, horizon: int, zone_idx: int) -> Optional[float]:
        if self.true is None:
            return None
        h = self.horizon_safe(horizon)
        value = self.true[int(sample_idx), h, int(zone_idx)]
        if pd.isna(value):
            return None
        return float(max(value, 0.0))

    def slice_df(self, sample_idx: int, horizon: int) -> pd.DataFrame:
        h = self.horizon_safe(horizon)
        values = self.pred[int(sample_idx), h, :]
        base_dt = self.sample_to_datetime(sample_idx)
        target_dt = self.target_datetime(sample_idx, h)
        return pd.DataFrame(
            {
                "생활권역ID": self.zone_ids,
                "zone_idx": np.arange(self.n_zones),
                "sample_idx": int(sample_idx),
                "horizon": h,
                "global_time_idx": int(sample_idx + self.offset),
                "target_global_time_idx": int(sample_idx + self.offset + h),
                "date_str": base_dt.strftime("%Y-%m-%d"),
                "time_str": base_dt.strftime("%H:%M"),
                "target_time_str": target_dt.strftime("%H:%M"),
                "daily_slot": base_dt.hour * 2 + base_dt.minute // 30,
                "predicted_kwh": values,
            }
        )


def first_existing_path(candidates: list[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    names = ", ".join(p.name for p in candidates)
    raise FileNotFoundError(f"예측 파일을 찾을 수 없습니다. 다음 중 하나가 필요합니다: {names}")


def pick_npz_array(npz: np.lib.npyio.NpzFile, keys: list[str]) -> Optional[np.ndarray]:
    for key in keys:
        if key in npz.files:
            return npz[key]
    return None


def normalize_prediction_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 4:
        # (N,H,Z,1)
        if arr.shape[-1] != 1:
            raise ValueError(f"4차원 예측 배열의 마지막 차원은 1이어야 합니다. 현재 shape={arr.shape}")
        arr = arr[:, :, :, 0]
    elif arr.ndim == 3:
        # (N,H,Z) 또는 (N,Z,1)
        if arr.shape[-1] == 1:
            arr = arr[:, None, :, 0]
    elif arr.ndim == 2:
        # (N,Z)
        arr = arr[:, None, :]
    else:
        raise ValueError(f"지원하지 않는 예측 배열 shape입니다: {arr.shape}")

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0, None)
    return arr.astype(np.float32)


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
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"meta.json을 읽지 못했습니다. 마지막 오류: {last_error}")


@st.cache_resource(show_spinner="예측 결과 로딩 중...")
def load_prediction_store(mode: str, meta: Dict) -> PredictionStore:
    pred_path = first_existing_path(PRED_FILE_CANDIDATES[mode])
    zone_ids = list(meta["zone_ids"])
    n_zones_meta = len(zone_ids)

    if pred_path.suffix.lower() == ".npz":
        npz = np.load(pred_path)
        pred_arr = pick_npz_array(npz, ["preds", "pred", "predictions", "y_pred", "y_pred_kwh", "predicted_kwh"])
        true_arr = pick_npz_array(npz, ["trues", "true", "actual", "actuals", "y_true", "y_true_kwh", "true_kwh"])
        if pred_arr is None:
            raise ValueError(f"{pred_path.name} 내부에서 예측 배열을 찾지 못했습니다. 현재 keys={npz.files}")
    elif pred_path.suffix.lower() == ".npy":
        pred_arr = np.load(pred_path)
        true_arr = None
    else:
        raise ValueError(f"현재 안정 버전은 .npz 또는 .npy 예측 파일만 지원합니다: {pred_path.name}")

    pred = normalize_prediction_array(pred_arr)
    true = normalize_prediction_array(true_arr) if true_arr is not None else None

    if pred.shape[2] != n_zones_meta:
        raise ValueError(f"예측 결과 zone 수({pred.shape[2]})와 meta.json zone_ids 수({n_zones_meta})가 다릅니다.")
    if true is not None and true.shape != pred.shape:
        # 실제값이 horizon 없는 (N,Z)이고 pred가 (N,H,Z)인 경우 horizon0만 확장
        if true.shape[0] == pred.shape[0] and true.shape[2] == pred.shape[2] and true.shape[1] == 1:
            tmp = np.full_like(pred, np.nan, dtype=np.float32)
            tmp[:, 0:1, :] = true
            true = tmp
        else:
            true = None

    return PredictionStore(
        path=str(pred_path),
        pred=pred,
        true=true,
        zone_ids=zone_ids,
        val_end=int(meta.get("val_end", 0)),
        look_back=int(meta.get("look_back", 0)),
    )


@st.cache_data(show_spinner="생활권역 정보 로딩 중...")
def load_area_info(excel_path: Path) -> pd.DataFrame:
    area = pd.read_excel(excel_path)
    if "생활권역ID" not in area.columns:
        raise ValueError("엑셀 파일에 생활권역ID 컬럼이 없습니다.")

    area["생활권역ID"] = area["생활권역ID"].apply(normalize_id)

    keep_cols = []
    for col in ["생활권역ID", "생활권역OBJECTID", "생활권역코드", "생활권역구분", "생활권역라벨", "행정동수", "행정동명목록"]:
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
    return candidate if candidate and candidate in available_dates else None


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
    return candidate if candidate and candidate in available_times else None


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

    return best_zone if best_score > 0 else None


def llm_extract_query(text: str, store: PredictionStore, area_info: pd.DataFrame) -> dict:
    min_date, max_date = store.date_range()
    sample_zones = area_info[["생활권역ID", "생활권역표시명", "행정동명목록"]].head(80).to_dict(orient="records")

    system_prompt = f"""
너는 전기차 충전수요 지도 서비스의 자연어 질의 해석기다.
사용자의 한국어 질의를 분석해 intent, 날짜, 시간, 위치를 추출한다.
반드시 JSON만 출력한다.

출력 형식:
{{
  "date_text": "사용자가 말한 날짜 표현 또는 null",
  "time_text": "사용자가 말한 시간 표현 또는 null",
  "location_text": "사용자가 말한 위치/동/구/생활권 표현 또는 null",
  "intent": "demand_lookup | service_explanation | greeting | other",
  "needs_data_lookup": true 또는 false
}}

- 특정 날짜/시간/위치의 충전수요, 예측, 피크, 지도 조회를 원하면 demand_lookup.
- 서비스 작동 방식, 데이터, 모델, 지도, 예측 방식, 사용법을 물으면 service_explanation.
- 인사면 greeting.
- 예시는 반드시 다음 예시만 사용한다: "{FIXED_QUERY_EXAMPLE}"
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
    if parsed:
        intent = parsed.get("intent", "other")
        return {
            "date_text": parsed.get("date_text"),
            "time_text": parsed.get("time_text"),
            "location_text": parsed.get("location_text"),
            "intent": intent,
            "needs_data_lookup": bool(parsed.get("needs_data_lookup", intent == "demand_lookup")),
            "raw": llm_text,
        }

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


def parse_user_query(text: str, store: PredictionStore, area_info: pd.DataFrame, fallback_date: str, fallback_time: str, fallback_zone_id: str) -> Dict:
    available_dates = store.available_dates()
    llm_result = llm_extract_query(text, store, area_info)

    date_text = str(llm_result.get("date_text") or "").strip()
    time_text = str(llm_result.get("time_text") or "").strip()
    location_text = str(llm_result.get("location_text") or "").strip()
    intent = str(llm_result.get("intent") or "other").strip()
    needs_data_lookup = bool(llm_result.get("needs_data_lookup", False))

    has_demand_keyword = bool(re.search(r"수요|충전|예측|전기차|생활권|지도|보여|알려|조회|분석|혼잡|급증|피크|kwh|kw", text, re.IGNORECASE))
    has_any_condition = bool(extract_any_date_candidate(text) or extract_any_time_candidate(text) or find_zone_by_location(text, area_info) or location_text)

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
            "message": f"충전수요 예측을 조회하려면 연도, 월, 일, 시간, 위치가 모두 필요합니다. 예: {FIXED_QUERY_EXAMPLE}",
            "date": fallback_date,
            "time": fallback_time,
            "zone_id": fallback_zone_id,
            "llm_extract": llm_result,
        }

    date_source = date_text or text
    any_date = extract_any_date_candidate(date_source) or extract_any_date_candidate(text)
    min_date, max_date = store.date_range()
    if any_date and any_date not in available_dates:
        return {
            "ok": False,
            "reason": "date_unavailable",
            "message": f"{any_date} 날짜는 현재 예측 데이터셋에 존재하지 않습니다. 현재 조회 가능한 날짜 범위는 {min_date}부터 {max_date}까지입니다. 예: {FIXED_QUERY_EXAMPLE}",
            "date": fallback_date,
            "time": fallback_time,
            "zone_id": fallback_zone_id,
            "llm_extract": llm_result,
        }

    parsed_date = parse_date_from_text(date_source, available_dates) or parse_date_from_text(text, available_dates)
    if parsed_date is None:
        return {
            "ok": False,
            "reason": "missing_date",
            "message": f"조회할 날짜를 찾지 못했습니다. 연도, 월, 일을 모두 포함해 입력해 주세요. 현재 데이터는 {min_date}부터 {max_date}까지 조회할 수 있습니다. 예: {FIXED_QUERY_EXAMPLE}",
            "date": fallback_date,
            "time": fallback_time,
            "zone_id": fallback_zone_id,
            "llm_extract": llm_result,
        }

    available_times = store.available_times_for_date(parsed_date)
    time_source = time_text or text
    any_time = extract_any_time_candidate(time_source) or extract_any_time_candidate(text)
    if any_time and any_time not in available_times:
        return {
            "ok": False,
            "reason": "time_unavailable",
            "message": f"{parsed_date} {any_time} 시간대는 현재 예측 데이터셋에 존재하지 않습니다. 현재 서비스는 예측 파일에 포함된 30분 단위 시간대만 조회할 수 있습니다. 예: {FIXED_QUERY_EXAMPLE}",
            "date": fallback_date,
            "time": fallback_time,
            "zone_id": fallback_zone_id,
            "llm_extract": llm_result,
        }

    parsed_time = parse_time_from_text(time_source, available_times) or parse_time_from_text(text, available_times)
    if parsed_time is None:
        return {
            "ok": False,
            "reason": "missing_time",
            "message": f"조회할 시간을 찾지 못했습니다. 연도, 월, 일, 시간, 위치를 모두 포함해 입력해 주세요. 예: {FIXED_QUERY_EXAMPLE}",
            "date": fallback_date,
            "time": fallback_time,
            "zone_id": fallback_zone_id,
            "llm_extract": llm_result,
        }

    parsed_zone_id = find_zone_by_location(" ".join([location_text, text]).strip(), area_info)
    if parsed_zone_id is None:
        return {
            "ok": False,
            "reason": "location_unavailable",
            "message": f"입력한 위치는 현재 서울시 생활권 데이터셋에서 찾을 수 없습니다. 서울시 행정동, 자치구, 생활권 이름을 기준으로 다시 입력해 주세요. 예: {FIXED_QUERY_EXAMPLE}",
            "date": fallback_date,
            "time": fallback_time,
            "zone_id": fallback_zone_id,
            "llm_extract": llm_result,
        }

    return {"ok": True, "reason": "ok", "message": "", "date": parsed_date, "time": parsed_time, "zone_id": parsed_zone_id, "llm_extract": llm_result}


# =========================================================
# 데이터 계산 함수
# =========================================================
def calc_zone_metrics(store: PredictionStore, sample_idx: int, zone_id: str) -> Dict[str, Any]:
    zone_idx = store.zone_index(zone_id)
    h0 = store.horizon_safe(0)
    h1 = store.horizon_safe(PREDICT_HORIZON_30M)

    true_current = store.true_value(sample_idx, h0, zone_idx)
    current_kwh = true_current if true_current is not None else store.pred_value(sample_idx, h0, zone_idx)
    pred_30m_kwh = store.pred_value(sample_idx, h1, zone_idx)

    rank_values = store.pred[sample_idx, h1, :]
    rank = int((rank_values > pred_30m_kwh).sum() + 1)

    max_h = min(PEAK_WINDOW_HORIZONS, store.n_horizons)
    window = store.pred[sample_idx, :max_h, zone_idx]
    peak_h = int(np.argmax(window))
    low_h = int(np.argmin(window))

    return {
        "current_kwh": float(current_kwh),
        "pred_30m_kwh": float(pred_30m_kwh),
        "rank": rank,
        "n_zones": store.n_zones,
        "peak_time": store.target_datetime(sample_idx, peak_h).strftime("%H:%M"),
        "peak_kwh": float(window[peak_h]),
        "low_time": store.target_datetime(sample_idx, low_h).strftime("%H:%M"),
        "low_kwh": float(window[low_h]),
    }


def get_selected_area(area_info: pd.DataFrame, zone_id: str) -> tuple[str, str]:
    selected_area = area_info[area_info["생활권역ID"] == zone_id].copy()
    if selected_area.empty:
        return zone_id, ""
    selected_label = selected_area["생활권역표시명"].iloc[0]
    selected_dongs = selected_area["행정동명목록"].iloc[0] if "행정동명목록" in selected_area.columns else ""
    return selected_label, selected_dongs


# =========================================================
# 지도 데이터 생성
# =========================================================
def prepare_map_gdf(boundary_gdf: gpd.GeoDataFrame, area_info: pd.DataFrame, pred_filtered: pd.DataFrame, focus_zone_id: Optional[str] = None) -> gpd.GeoDataFrame:
    gdf = boundary_gdf.copy()
    gdf = gdf.merge(area_info, left_on="ID", right_on="생활권역ID", how="left")
    gdf = gdf.merge(
        pred_filtered[[
            "생활권역ID", "zone_idx", "sample_idx", "horizon", "global_time_idx",
            "target_global_time_idx", "date_str", "time_str", "target_time_str",
            "daily_slot", "predicted_kwh",
        ]],
        on="생활권역ID",
        how="left",
    )

    if "생활권역표시명" not in gdf.columns:
        gdf["생활권역표시명"] = gdf["ID"]
    if "행정동명목록" not in gdf.columns:
        gdf["행정동명목록"] = ""

    gdf["is_focus"] = gdf["ID"] == focus_zone_id
    gdf["predicted_kwh"] = pd.to_numeric(gdf["predicted_kwh"], errors="coerce").fillna(0).astype(float).round(3)
    return gdf


def get_zone_view(gdf: gpd.GeoDataFrame, zone_id: Optional[str], use_3d_column: bool) -> dict:
    focus = gdf[gdf["ID"] == zone_id] if zone_id else pd.DataFrame()
    if not focus.empty:
        return {
            "latitude": float(focus["lat"].iloc[0]),
            "longitude": float(focus["lon"].iloc[0]),
            "zoom": 12.0,
            "pitch": 42 if use_3d_column else 0,
            "bearing": 0,
        }
    return {"latitude": OVERVIEW_LATITUDE, "longitude": OVERVIEW_LONGITUDE, "zoom": OVERVIEW_ZOOM, "pitch": 40 if use_3d_column else 0, "bearing": 0}


def prepare_map_payload(map_gdf: gpd.GeoDataFrame, use_3d_column: bool, focus_zone_id: Optional[str], previous_focus_zone_id: Optional[str]) -> dict:
    gdf = map_gdf.copy()
    vmin = float(gdf["predicted_kwh"].quantile(0.05))
    vmax = float(gdf["predicted_kwh"].quantile(0.95))
    max_kwh = max(float(gdf["predicted_kwh"].max()) if len(gdf) else 1.0, 1.0)

    gdf["fill_color"] = gdf.apply(lambda row: [255, 130, 80, 220] if bool(row.get("is_focus", False)) else kwh_to_color(row["predicted_kwh"], vmin, vmax), axis=1)
    gdf["line_color"] = gdf["is_focus"].apply(lambda x: [255, 60, 60, 255] if bool(x) else [255, 255, 255, 185])
    gdf["line_width"] = gdf["is_focus"].apply(lambda x: 90 if bool(x) else 20)

    for col in ["zone_idx", "sample_idx", "horizon", "global_time_idx", "target_global_time_idx", "daily_slot"]:
        if col in gdf.columns:
            gdf[col] = gdf[col].astype("Int64").astype(str)

    geojson = json.loads(gdf.to_json())
    columns_df = pd.DataFrame(gdf.drop(columns="geometry")).copy()
    columns_df["fill_color"] = columns_df.apply(lambda row: [255, 90, 70, 235] if bool(row.get("is_focus", False)) else row["fill_color"], axis=1)

    def calc_elevation(row: pd.Series) -> float:
        value = float(row["predicted_kwh"]) if pd.notna(row["predicted_kwh"]) else 0.0
        ratio = np.sqrt(np.clip(value / max_kwh, 0.0, 1.0))
        height = 220.0 + 2450.0 * ratio
        if bool(row.get("is_focus", False)):
            height += 450.0
        return float(np.clip(height, 220.0, 3100.0))

    columns_df["elevation"] = columns_df.apply(calc_elevation, axis=1)
    columns = columns_df[["ID", "생활권역표시명", "행정동명목록", "predicted_kwh", "target_time_str", "lon", "lat", "is_focus", "fill_color", "elevation"]].to_dict(orient="records")

    overview_view = {"latitude": OVERVIEW_LATITUDE, "longitude": OVERVIEW_LONGITUDE, "zoom": OVERVIEW_ZOOM, "pitch": 40 if use_3d_column else 0, "bearing": 0}
    previous_view = get_zone_view(gdf, previous_focus_zone_id, use_3d_column)
    target_view = get_zone_view(gdf, focus_zone_id, use_3d_column)

    has_focus = bool(focus_zone_id and not gdf[gdf["ID"] == focus_zone_id].empty)
    has_previous_focus = bool(previous_focus_zone_id and previous_focus_zone_id != focus_zone_id and not gdf[gdf["ID"] == previous_focus_zone_id].empty)

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
            html, body {{ margin: 0; padding: 0; width: 100%; height: {height}px; overflow: hidden; background: #FFFFFF; font-family: Inter, -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Noto Sans KR", sans-serif; }}
            #map {{ position: relative; width: 100%; height: {height}px; border-radius: 16px; overflow: hidden; background: #FFFFFF; border: none; box-shadow: none; }}
            .deck-tooltip {{ font-size: 12px !important; border-radius: 13px !important; padding: 10px 12px !important; background: rgba(17,17,17,0.92) !important; color: #FFFFFF !important; box-shadow: none !important; }}
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

            function smootherstep(t) {{ return t * t * t * (t * (t * 6 - 15) + 10); }}
            function lerp(a, b, t) {{ return a + (b - a) * t; }}
            function makeView(start, target, t) {{
                const e = smootherstep(t);
                return {{ latitude: lerp(start.latitude, target.latitude, e), longitude: lerp(start.longitude, target.longitude, e), zoom: lerp(start.zoom, target.zoom, e), pitch: lerp(start.pitch || 0, target.pitch || 0, e), bearing: lerp(start.bearing || 0, target.bearing || 0, e) }};
            }}

            function polygonLayer() {{
                return new deck.GeoJsonLayer({{
                    id: "living-area-polygons",
                    data: geojsonData,
                    pickable: true,
                    stroked: true,
                    filled: true,
                    extruded: false,
                    getFillColor: f => f.properties.fill_color || [220,225,232,90],
                    getLineColor: f => f.properties.line_color || [255,255,255,180],
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
                    getFillColor: d => d.fill_color || [100,140,230,190],
                    getElevation: d => d.elevation || 0,
                    elevationScale: 1,
                    autoHighlight: true
                }});
            }}
            function makeLayers() {{ const base = [polygonLayer()]; if (use3d) base.push(columnLayer()); return base; }}

            let initialView;
            if (shouldAnimate && hasFocus && hasPreviousFocus) initialView = previousView;
            else if (shouldAnimate && hasFocus) initialView = overviewView;
            else if (hasFocus) initialView = targetView;
            else initialView = overviewView;
            let currentView = initialView;

            const deckgl = new deck.DeckGL({{
                container: "map",
                mapStyle: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                viewState: currentView,
                controller: true,
                layers: makeLayers(),
                onViewStateChange: e => {{ currentView = e.viewState; deckgl.setProps({{ viewState: currentView }}); }},
                getTooltip: info => {{
                    if (!info.object) return null;
                    const p = info.object.properties || info.object;
                    const label = p["생활권역표시명"] || p.ID || "";
                    const id = p.ID || "";
                    const kwh = p.predicted_kwh || "";
                    const targetTime = p.target_time_str || "";
                    const dongs = p["행정동명목록"] || "";
                    return {{ html: `<div style="font-family: Inter, sans-serif;"><b>${{label}}</b><br/><span style="color:#B7C1D3;">생활권ID:</span> ${{id}}<br/><span style="color:#B7C1D3;">예측 시각:</span> ${{targetTime}}<br/><span style="color:#B7C1D3;">예측 충전량:</span> <b>${{kwh}} kWh</b><br/><span style="color:#B7C1D3;">행정동:</span> ${{dongs}}</div>` }};
                }}
            }});

            function animateBetween(startView, endView, duration) {{
                return new Promise(resolve => {{
                    const startTime = performance.now();
                    function step(now) {{
                        const raw = (now - startTime) / duration;
                        const t = Math.min(Math.max(raw, 0), 1);
                        const nextView = makeView(startView, endView, t);
                        deckgl.setProps({{ viewState: nextView, layers: makeLayers() }});
                        if (t < 1) requestAnimationFrame(step);
                        else {{ deckgl.setProps({{ viewState: endView, layers: makeLayers() }}); resolve(); }}
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
# 알림 및 채팅 HTML
# =========================================================
def draw_alerts_stack(top_df: pd.DataFrame, selected_time: str) -> None:
    if top_df.empty:
        st.info("수요 알림을 생성할 수 없습니다.")
        return

    alert_rows = top_df.head(8).copy()
    cards_html = ""
    for i, row in enumerate(alert_rows.itertuples(), start=1):
        label = getattr(row, "생활권역표시명", getattr(row, "생활권역ID"))
        value = float(getattr(row, "predicted_kwh"))
        target_time = str(getattr(row, "target_time_str", selected_time))
        if i == 1:
            state, state_kr, state_class, title = "PEAK", "급증", "hot", "가장 높은 충전수요가 예상됩니다"
        elif i <= 3:
            state, state_kr, state_class, title = "WATCH", "주의", "watch", "충전기 운영 여유를 확인하세요"
        else:
            state, state_kr, state_class, title = "MONITOR", "관찰", "monitor", "수요 변화를 함께 모니터링하세요"
        cards_html += f"""
        <div class="ev-alert-card">
            <div class="alert-left"><div class="state-circle {state_class}"><span>{state_kr}</span></div></div>
            <div class="alert-center"><div class="alert-brand {state_class}">{state}</div><div class="alert-zone">{escape_html(label)}</div><div class="alert-copy">{escape_html(title)}</div></div>
            <div class="alert-right"><div class="right-label">예측수요</div><div class="right-value">{value:,.1f} kWh</div><div class="right-label time">시간</div><div class="right-value">{escape_html(target_time)}</div></div>
        </div>
        """

    loop_cards_html = cards_html + cards_html
    card_count = len(alert_rows)

    html = f"""
    <!DOCTYPE html><html><head><meta charset="utf-8" />
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Holtwood+One+SC&display=swap');
        html, body {{ margin: 0; padding: 0; background: transparent; font-family: Inter, -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Noto Sans KR", sans-serif; overflow: hidden; }}
        .alert-stack-panel {{ position: relative; height: {ALERT_HEIGHT}px; overflow: hidden; box-sizing: border-box; padding: 4px 2px 0 0; background: transparent; }}
        .alert-scroll-track {{ will-change: transform; transform: translateY(0); }}
        .ev-alert-card {{ height: 112px; display: grid; grid-template-columns: 62px minmax(0, 1fr) 106px; align-items: center; gap: 13px; background: #FFFFFF; border: 1.1px solid rgba(20,20,20,0.20); border-radius: 18px; padding: 14px 14px; box-sizing: border-box; margin-bottom: 14px; box-shadow: none; }}
        .state-circle {{ width: 55px; height: 55px; border-radius: 999px; display: flex; align-items: center; justify-content: center; color: #FFFFFF; font-size: 13px; font-weight: 800; letter-spacing: -0.035em; }}
        .state-circle.hot {{ background: #FF3F4F; }} .state-circle.watch {{ background: #F5A000; }} .state-circle.monitor {{ background: #657386; }}
        .alert-center {{ min-width: 0; }} .alert-brand {{ font-family: "Holtwood One SC", Georgia, serif; font-size: 11px; font-weight: 400; letter-spacing: -0.02em; line-height: 1; margin-bottom: 8px; }}
        .alert-brand.hot {{ color: #FF3F4F; }} .alert-brand.watch {{ color: #F5A000; }} .alert-brand.monitor {{ color: #657386; }}
        .alert-zone {{ color: #111111; font-size: 15px; font-weight: 850; letter-spacing: -0.045em; line-height: 1.25; word-break: keep-all; white-space: normal; }}
        .alert-copy {{ margin-top: 6px; color: #555C66; font-size: 11.4px; font-weight: 500; line-height: 1.35; word-break: keep-all; white-space: normal; }}
        .alert-right {{ display: grid; justify-items: end; align-items: center; }}
        .right-label {{ display: inline-flex; align-items: center; justify-content: center; min-width: 70px; height: 21px; border-radius: 999px; background: #9A9A9A; color: #FFFFFF; font-size: 10px; font-weight: 650; line-height: 1; }}
        .right-label.time {{ margin-top: 8px; }} .right-value {{ color: #111111; font-size: 13px; font-weight: 750; line-height: 1.15; margin-top: 5px; white-space: nowrap; letter-spacing: -0.02em; }}
    </style></head><body>
        <div class="alert-stack-panel" id="alertPanel"><div class="alert-scroll-track" id="alertTrack">{loop_cards_html}</div></div>
        <script>
            const panel = document.getElementById("alertPanel"); const track = document.getElementById("alertTrack"); const originalCount = {card_count};
            let index = 0; let isPausedByHover = false; const MOVE_DURATION = 650; const HOLD_DURATION = 1900;
            function getCards() {{ return Array.from(track.querySelectorAll(".ev-alert-card")); }}
            function getOffsetForIndex(targetIndex) {{ const cards = getCards(); if (!cards[targetIndex]) return 0; return cards[targetIndex].offsetTop; }}
            function moveTo(targetIndex, withTransition = true) {{ const offset = getOffsetForIndex(targetIndex); track.style.transition = withTransition ? `transform ${{MOVE_DURATION}}ms cubic-bezier(0.22, 1, 0.36, 1)` : "none"; track.style.transform = `translateY(-${{offset}}px)`; }}
            function resetIfNeeded() {{ if (index >= originalCount) {{ index = 0; moveTo(0, false); void track.offsetHeight; }} }}
            function tick() {{ if (isPausedByHover) {{ setTimeout(tick, HOLD_DURATION); return; }} index += 1; moveTo(index, true); setTimeout(() => {{ resetIfNeeded(); setTimeout(tick, HOLD_DURATION); }}, MOVE_DURATION); }}
            panel.addEventListener("mouseenter", () => {{ isPausedByHover = true; }}); panel.addEventListener("mouseleave", () => {{ isPausedByHover = false; }}); setTimeout(tick, HOLD_DURATION);
        </script>
    </body></html>
    """
    components.html(html, height=ALERT_HEIGHT, scrolling=False)


def build_selected_detail_html(
    selected_label: str,
    selected_zone_id: str,
    selected_dongs: str,
    current_kwh: float,
    pred_30m_kwh: float,
    peak_time: str,
    peak_kwh: float,
    low_time: str,
    low_kwh: float,
    selected_date: str,
    selected_time: str,
    model_label: str,
) -> str:
    return f"""
    <div class="selected-detail-card">
        <div class="detail-card-header"><div><div class="detail-card-kicker">SELECTED AREA · {escape_html(model_label)}</div><div class="detail-card-title">{escape_html(selected_label)}</div><div class="detail-card-id">{escape_html(selected_zone_id)}</div></div></div>
        <div class="detail-card-meta">
            <div class="meta-block"><div class="meta-label">조회 시각</div><div class="meta-text">{escape_html(selected_date)} {escape_html(selected_time)}</div></div>
            <div class="meta-block"><div class="meta-label">포함 행정동</div><div class="meta-text">{escape_html(selected_dongs if selected_dongs else "행정동 정보 없음")}</div></div>
        </div>
        <div class="detail-metric-grid">
            <div class="detail-metric"><div class="metric-label">현재 수요</div><div class="metric-value">{current_kwh:.1f} kWh</div></div>
            <div class="detail-metric"><div class="metric-label">예측 수요</div><div class="metric-value">{pred_30m_kwh:.1f} kWh</div><div class="metric-delta">30분 후</div></div>
            <div class="detail-metric"><div class="metric-label">피크 타임</div><div class="metric-value">{escape_html(peak_time)}</div><div class="metric-delta">{peak_kwh:.1f} kWh</div></div>
            <div class="detail-metric"><div class="metric-label">오프 피크</div><div class="metric-value">{escape_html(low_time)}</div><div class="metric-delta low">{low_kwh:.1f} kWh</div></div>
        </div>
    </div>
    """


def render_chat_panel(messages: list[dict], selected_detail_html: str | None = None, is_typing: bool = False) -> None:
    items_html = ""
    for msg in messages:
        role = msg.get("role", "assistant")
        content = escape_html(msg.get("content", ""))
        role_class = "user" if role == "user" else "assistant"
        items_html += f'<div class="chat-bubble-row {role_class}"><div class="chat-bubble {role_class}">{content}</div></div>'

    if is_typing:
        items_html += """
        <div class="chat-bubble-row assistant"><div class="chat-bubble assistant typing-bubble"><span class="typing-dot dot-1"></span><span class="typing-dot dot-2"></span><span class="typing-dot dot-3"></span></div></div>
        """
    if selected_detail_html and not is_typing:
        items_html += selected_detail_html

    html = f"""
    <!DOCTYPE html><html><head><meta charset="utf-8" />
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Holtwood+One+SC&display=swap');
        html, body {{ margin: 0; padding: 0; background: transparent; font-family: Inter, -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Noto Sans KR", sans-serif; overflow: hidden; }}
        .chat-scroll-box {{ height: {CHAT_SCROLL_HEIGHT}px; overflow-y: auto; overflow-x: hidden; padding: 8px 4px 8px 2px; box-sizing: border-box; border-top: 1px solid rgba(20,20,20,0.08); border-bottom: 1px solid rgba(20,20,20,0.08); background: transparent; scroll-behavior: smooth; }}
        .chat-scroll-box::-webkit-scrollbar {{ width: 7px; }} .chat-scroll-box::-webkit-scrollbar-thumb {{ background: rgba(120,130,145,0.45); border-radius: 999px; }} .chat-scroll-box::-webkit-scrollbar-track {{ background: transparent; }}
        .chat-bubble-row {{ display: flex; margin-bottom: 11px; }} .chat-bubble-row.user {{ justify-content: flex-end; }} .chat-bubble-row.assistant {{ justify-content: flex-start; }}
        .chat-bubble {{ max-width: 88%; border-radius: 16px; padding: 11px 13px; font-size: 12px; font-weight: 560; line-height: 1.58; word-break: keep-all; overflow-wrap: anywhere; box-sizing: border-box; }}
        .chat-bubble.user {{ background: #1F78B4; color: #FFFFFF; border-bottom-right-radius: 5px; box-shadow: none; }} .chat-bubble.assistant {{ background: #FFFFFF; color: #111111; border: 1px solid rgba(20,20,20,0.20); border-bottom-left-radius: 5px; box-shadow: none; }}
        .typing-bubble {{ display: inline-flex; align-items: center; gap: 5px; min-width: 54px; height: 34px; padding: 9px 13px; }}
        .typing-dot {{ width: 7px; height: 7px; border-radius: 999px; background: #1F78B4; display: inline-block; animation: typingBounce 1.05s infinite ease-in-out; }} .typing-dot.dot-1 {{ animation-delay: 0s; }} .typing-dot.dot-2 {{ animation-delay: 0.16s; }} .typing-dot.dot-3 {{ animation-delay: 0.32s; }}
        @keyframes typingBounce {{ 0%,80%,100% {{ transform: translateY(0); opacity: 0.42; }} 40% {{ transform: translateY(-7px); opacity: 1; }} }}
        .selected-detail-card {{ margin: 13px 2px 8px 2px; padding: 14px; border-radius: 18px; border: 1.1px solid rgba(20,20,20,0.22); background: #FFFFFF; box-shadow: none; box-sizing: border-box; }}
        .detail-card-kicker {{ color: #1F78B4; font-family: "Holtwood One SC", Georgia, serif; font-size: 10.5px; font-weight: 400; letter-spacing: -0.02em; margin-bottom: 5px; }}
        .detail-card-title {{ color: #111111; font-size: 18px; font-weight: 850; letter-spacing: -0.045em; line-height: 1.25; }} .detail-card-id {{ color: #5F666F; font-size: 11px; font-weight: 600; margin-top: 4px; }}
        .detail-card-meta {{ margin-top: 11px; display: flex; flex-direction: column; gap: 7px; }} .meta-label {{ color: #777D86; font-size: 10.5px; font-weight: 650; margin-bottom: 2px; }} .meta-text {{ color: #333333; font-size: 11.5px; font-weight: 500; line-height: 1.42; word-break: keep-all; overflow-wrap: anywhere; }}
        .detail-metric-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px; }} .detail-metric {{ background: #FAFAFA; border: 1px solid rgba(20,20,20,0.12); border-radius: 13px; padding: 9px 10px; min-height: 58px; box-sizing: border-box; box-shadow: none; }}
        .metric-label {{ color: #777D86; font-size: 10.5px; font-weight: 600; margin-bottom: 4px; }} .metric-value {{ color: #111111; font-size: 17px; font-weight: 820; letter-spacing: -0.04em; }} .metric-delta {{ display: inline-flex; margin-top: 4px; padding: 2px 6px; border-radius: 999px; background: #EAF4FA; color: #1F78B4; font-size: 10.5px; font-weight: 700; }} .metric-delta.low {{ background: #F1F3F5; color: #657386; }}
    </style></head><body><div class="chat-scroll-box" id="chatbox">{items_html}</div><script>const box = document.getElementById("chatbox"); if (box) {{ requestAnimationFrame(() => {{ box.scrollTop = box.scrollHeight; }}); }}</script></body></html>
    """
    components.html(html, height=CHAT_SCROLL_HEIGHT + 8, scrolling=False)


# =========================================================
# 답변 생성
# =========================================================
def build_fallback_answer(selected_date: str, selected_time: str, selected_label: str, metrics: Dict[str, Any], model_label: str) -> str:
    return (
        f"{selected_date} {selected_time} 기준으로 요청하신 위치는 {selected_label} 생활권에 포함됩니다.\n\n"
        f"현재 수요는 {metrics['current_kwh']:.1f} kWh이고, {model_label} 기준 30분 후 예측 수요는 {metrics['pred_30m_kwh']:.1f} kWh입니다.\n"
        f"입력 시각부터 6시간 범위에서 피크 타임은 {metrics['peak_time']}이며, 해당 시각의 예측 수요는 {metrics['peak_kwh']:.1f} kWh입니다.\n"
        f"같은 6시간 범위에서 오프 피크는 {metrics['low_time']}이며, 예측 수요는 {metrics['low_kwh']:.1f} kWh입니다.\n\n"
        f"지도와 알림은 현재 선택된 {model_label} 기준으로 표시됩니다."
    )


def build_conversational_answer(user_text: str, reason_message: str, reason: str, llm_extract: dict | None, messages: list[dict], store: PredictionStore, model_label: str) -> str:
    min_date, max_date = store.date_range()
    chat_context = get_recent_chat_context(messages)
    system_prompt = f"""
너는 서울시 생활권별 전기차 충전수요 예측 서비스 E-Vlog의 대화형 LLM '모도리'다.
예측 수치, 순위, 피크 시간 등 데이터 기반 정보는 제공된 데이터에서 조회된 경우에만 말할 수 있다.
한국어로 3~7문장으로 답변한다.
예시는 반드시 다음 예시만 사용한다: "{FIXED_QUERY_EXAMPLE}"
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
- 현재 선택된 모델은 {model_label}이다.
- 사용자는 모도리 패널 상단 토글 버튼으로 일반 모델과 피크 모델을 전환할 수 있다.
- 예측 데이터 날짜 범위는 {min_date}부터 {max_date}까지다.
"""
    llm_answer = call_gemini_text(system_prompt, user_prompt, temperature=0.65)
    if llm_answer and llm_answer.strip():
        return llm_answer.strip()

    if reason == "service_explanation" or re.search(r"어떻게|작동|원리|방식|설명|사용법|데이터|모델|지도", user_text):
        return (
            f"이 서비스는 사용자가 입력한 연도, 월, 일, 시간, 위치를 해석한 뒤, 현재 선택된 {model_label}의 예측 파일에서 충전수요를 조회합니다.\n\n"
            "모도리 패널 상단의 피크 모델 토글 버튼으로 일반 모델과 피크 모델을 전환할 수 있고, 전환된 모델 기준으로 지도와 모도리 답변이 함께 바뀝니다. "
            f"예: {FIXED_QUERY_EXAMPLE}"
        )
    if reason == "greeting" or re.search(r"안녕|하이|hello|hi", user_text, re.IGNORECASE):
        return f"안녕하세요. 저는 모도리입니다.\n\n현재는 {model_label} 기준으로 서울시 생활권별 전기차 충전수요를 조회할 수 있습니다. 연도, 월, 일, 시간, 위치를 함께 입력해 주세요.\n\n예: {FIXED_QUERY_EXAMPLE}"
    if reason in {"missing_conditions", "missing_date", "missing_time", "location_unavailable", "date_unavailable", "time_unavailable"}:
        return append_example_once(reason_message)
    return f"저는 전기차 충전수요 예측을 도와주는 모도리입니다. 현재는 {model_label} 기준으로 답변하고 있습니다.\n\n예: {FIXED_QUERY_EXAMPLE}"


def build_llm_answer(user_text: str, selected_date: str, selected_time: str, selected_label: str, selected_zone_id: str, selected_dongs: str, metrics: Dict[str, Any], top10: pd.DataFrame, messages: list[dict], model_label: str) -> str:
    top_items = []
    for i, row in enumerate(top10.head(5).itertuples(), start=1):
        top_items.append({"rank": i, "zone": getattr(row, "생활권역표시명", getattr(row, "생활권역ID")), "predicted_kwh": round(float(getattr(row, "predicted_kwh")), 1), "target_time": str(getattr(row, "target_time_str", ""))})

    facts = {
        "model": model_label,
        "user_query": user_text,
        "date": selected_date,
        "time": selected_time,
        "zone_label": selected_label,
        "zone_id": selected_zone_id,
        "included_dongs": selected_dongs,
        "current_kwh": round(metrics["current_kwh"], 1),
        "pred_30m_kwh": round(metrics["pred_30m_kwh"], 1),
        "peak_time_next_6h": metrics["peak_time"],
        "peak_kwh": round(metrics["peak_kwh"], 1),
        "off_peak_time_next_6h": metrics["low_time"],
        "low_kwh": round(metrics["low_kwh"], 1),
        "top5_at_30m_forecast": top_items,
    }
    chat_context = get_recent_chat_context(messages)

    system_prompt = f"""
너는 E-Vlog 서비스의 전기차 충전수요 분석 LLM '모도리'다.
현재 사용자가 선택한 모델은 {model_label}이다.
수요 예측 수치와 관련된 내용은 반드시 제공된 facts만 근거로 답변한다.
한국어로 4~8문장 정도로 답변한다.
예시를 만들지 않는다.
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
    return build_fallback_answer(selected_date, selected_time, selected_label, metrics, model_label)


# =========================================================
# 데이터 로딩
# =========================================================
try:
    meta = load_meta(META_JSON_PATH)
    area_info = load_area_info(AREA_EXCEL_PATH)
    boundary_gdf = load_living_area_gdf(SHP_PATH, meta["zone_ids"])
except Exception as e:
    st.error("기본 데이터 로딩 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()


# =========================================================
# 세션 상태
# =========================================================
if "model_mode" not in st.session_state:
    st.session_state.model_mode = "overall"

for key, default in [
    ("previous_focus_zone_id", None),
    ("use_3d_column", True),
    ("has_query", False),
    ("show_selected_detail", False),
    ("animate_zoom", False),
    ("last_llm_error", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "안녕하세요. 저는 모도리입니다.\n\n"
                "모도리 패널 상단에서 일반 모델과 피크 모델을 선택할 수 있습니다. "
                "보고 싶은 연도, 월, 일, 시간, 위치를 자연어로 입력하면 현재 선택된 모델 기준으로 충전수요를 알려드립니다.\n\n"
                f"예: {FIXED_QUERY_EXAMPLE}"
            ),
        }
    ]

try:
    store = load_prediction_store(st.session_state.model_mode, meta)
except Exception as e:
    st.error("예측 결과 로딩 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()

model_mode = st.session_state.model_mode
model_label = MODEL_LABELS[model_mode]

available_dates_all = store.available_dates()
if "selected_date" not in st.session_state:
    st.session_state.selected_date = DEFAULT_DATE if DEFAULT_DATE in available_dates_all else available_dates_all[0]
if st.session_state.selected_date not in available_dates_all:
    st.session_state.selected_date = available_dates_all[0]

available_times_selected = store.available_times_for_date(st.session_state.selected_date)
if "selected_time" not in st.session_state:
    st.session_state.selected_time = DEFAULT_TIME if DEFAULT_TIME in available_times_selected else available_times_selected[0]
if st.session_state.selected_time not in available_times_selected:
    st.session_state.selected_time = available_times_selected[0]

zone_candidates = area_info[area_info["생활권역ID"].isin(meta["zone_ids"])].sort_values("생활권역ID")
if "selected_zone_id" not in st.session_state:
    st.session_state.selected_zone_id = zone_candidates["생활권역ID"].iloc[0]
if st.session_state.selected_zone_id not in store.zone_ids:
    st.session_state.selected_zone_id = store.zone_ids[0]

selected_date = st.session_state.selected_date
selected_time = st.session_state.selected_time
selected_zone_id = st.session_state.selected_zone_id
previous_focus_zone_id = st.session_state.previous_focus_zone_id
sample_idx = store.date_time_to_sample_idx(selected_date, selected_time)
if sample_idx is None:
    st.warning("선택한 날짜/시간에 해당하는 예측 데이터가 없습니다.")
    st.stop()

map_horizon = store.horizon_safe(MAP_HORIZON)
pred_filtered = store.slice_df(sample_idx, map_horizon)
selected_dt = store.sample_to_datetime(sample_idx)
target_dt = store.target_datetime(sample_idx, map_horizon)

if selected_zone_id not in set(pred_filtered["생활권역ID"]):
    selected_zone_id = pred_filtered["생활권역ID"].iloc[0]
    st.session_state.selected_zone_id = selected_zone_id

metrics = calc_zone_metrics(store, sample_idx, selected_zone_id)
selected_label, selected_dongs = get_selected_area(area_info, selected_zone_id)

top10 = pred_filtered.sort_values("predicted_kwh", ascending=False).head(10)
top10 = top10.merge(area_info[["생활권역ID", "생활권역라벨", "생활권역표시명"]], on="생활권역ID", how="left")

focus_zone_id = selected_zone_id if st.session_state.has_query else None
map_gdf = prepare_map_gdf(boundary_gdf, area_info, pred_filtered, focus_zone_id)

selected_detail_html = None
if st.session_state.has_query and st.session_state.show_selected_detail:
    selected_detail_html = build_selected_detail_html(
        selected_label=selected_label,
        selected_zone_id=selected_zone_id,
        selected_dongs=selected_dongs,
        current_kwh=metrics["current_kwh"],
        pred_30m_kwh=metrics["pred_30m_kwh"],
        peak_time=metrics["peak_time"],
        peak_kwh=metrics["peak_kwh"],
        low_time=metrics["low_time"],
        low_kwh=metrics["low_kwh"],
        selected_date=selected_date,
        selected_time=selected_time,
        model_label=model_label,
    )


# =========================================================
# 메인 3분할 레이아웃
# =========================================================
alert_col, map_col, chat_col = st.columns([0.86, 1.42, 0.78], gap="small")

with alert_col:
    with st.container(border=True, height=PANEL_HEIGHT):
        mark_panel()
        panel_title("수요 급증알림", f"30분 후 예측수요가 높은 생활권을 보여줍니다.", kicker="DEMAND WATCH")
        draw_alerts_stack(top10, selected_time)

with map_col:
    with st.container(border=True, height=PANEL_HEIGHT):
        mark_panel()
        header_left, header_right = st.columns([0.72, 0.28], gap="small")
        with header_left:
            panel_title("충전수요지도", f"기준 {selected_dt:%Y-%m-%d %H:%M} · 예측 {target_dt:%H:%M} · {model_label} 기반 서울시 생활권별 예측 충전수요", kicker="E-VLOG MAP")
        with header_right:
            use_2d = st.toggle("2D 지도 보기", value=not st.session_state.use_3d_column, key="map_2d_toggle")
            next_use_3d = not use_2d
            if next_use_3d != st.session_state.use_3d_column:
                st.session_state.use_3d_column = next_use_3d
                st.rerun()

        map_payload = prepare_map_payload(map_gdf, st.session_state.use_3d_column, focus_zone_id, previous_focus_zone_id)
        should_animate = bool(st.session_state.animate_zoom and st.session_state.has_query)
        render_deck_map_html(map_payload, should_animate, MAP_HEIGHT)
        if st.session_state.animate_zoom:
            st.session_state.animate_zoom = False
        render_legend()

with chat_col:
    with st.container(border=True, height=PANEL_HEIGHT):
        mark_panel()
        title_col, model_col = st.columns([0.50, 0.32], gap="small")
        with title_col:
            panel_title("MODORI", "모도리에게 무엇이든 물어보세요.", kicker="AI ASSISTANT")
        with model_col:
            st.markdown('<div class="chat-model-toggle-wrap">', unsafe_allow_html=True)
            use_peak_model = st.toggle(
                "피크모델",
                value=(st.session_state.model_mode == "peak"),
                key="model_peak_toggle",
                help="끄면 일반 모델, 켜면 피크 모델 기준으로 지도와 모도리가 응답합니다.",
            )
            st.markdown('</div>', unsafe_allow_html=True)
            next_mode = "peak" if use_peak_model else "overall"
            if next_mode != st.session_state.model_mode:
                st.session_state.model_mode = next_mode
                st.session_state.animate_zoom = False
                st.rerun()

        chat_placeholder = st.empty()
        with chat_placeholder:
            render_chat_panel(st.session_state.messages, selected_detail_html=selected_detail_html, is_typing=False)

        st.markdown('<div class="chat-form-wrap">', unsafe_allow_html=True)
        with st.form("chat_form", clear_on_submit=True):
            user_text = st.text_input("질문 입력", placeholder=f"예: {FIXED_QUERY_EXAMPLE}", label_visibility="collapsed")
            submitted = st.form_submit_button("질문하기", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if submitted and user_text.strip():
            clean_user_text = user_text.strip()
            st.session_state.messages.append({"role": "user", "content": clean_user_text})
            with chat_placeholder:
                render_chat_panel(st.session_state.messages, selected_detail_html=None, is_typing=True)

            parsed = parse_user_query(clean_user_text, store, area_info, st.session_state.selected_date, st.session_state.selected_time, st.session_state.selected_zone_id)

            if not parsed["ok"]:
                answer = build_conversational_answer(clean_user_text, parsed["message"], parsed["reason"], parsed.get("llm_extract"), st.session_state.messages, store, model_label)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.show_selected_detail = False
                st.session_state.animate_zoom = False
                st.rerun()

            next_sample_idx = store.date_time_to_sample_idx(parsed["date"], parsed["time"])
            if next_sample_idx is None:
                answer = "선택한 날짜와 시간에 해당하는 예측 데이터가 없습니다."
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.rerun()

            if st.session_state.has_query:
                st.session_state.previous_focus_zone_id = st.session_state.selected_zone_id
            else:
                st.session_state.previous_focus_zone_id = None

            next_metrics = calc_zone_metrics(store, next_sample_idx, parsed["zone_id"])
            next_label, next_dongs = get_selected_area(area_info, parsed["zone_id"])
            next_top10 = store.slice_df(next_sample_idx, store.horizon_safe(PREDICT_HORIZON_30M)).sort_values("predicted_kwh", ascending=False).head(10)
            next_top10 = next_top10.merge(area_info[["생활권역ID", "생활권역라벨", "생활권역표시명"]], on="생활권역ID", how="left")

            answer = build_llm_answer(
                clean_user_text,
                parsed["date"],
                parsed["time"],
                next_label,
                parsed["zone_id"],
                next_dongs,
                next_metrics,
                next_top10,
                st.session_state.messages,
                model_label,
            )
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.selected_date = parsed["date"]
            st.session_state.selected_time = parsed["time"]
            st.session_state.selected_zone_id = parsed["zone_id"]
            st.session_state.has_query = True
            st.session_state.show_selected_detail = True
            st.session_state.animate_zoom = True
            st.rerun()

        if st.session_state.get("last_llm_error"):
            st.markdown(f'<div class="llm-error-caption">LLM error: {escape_html(st.session_state.last_llm_error)}</div>', unsafe_allow_html=True)
