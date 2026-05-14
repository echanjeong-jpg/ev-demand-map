# E-Vlog Streamlit 수정 패치: 12-step 충전수요 예측 그래프 추가

아래 코드는 사용자가 제공한 app.py에 적용할 수정 패치입니다.

## 1) 기본 설정 상수 교체

기존:

```python
PANEL_HEIGHT = 625
MAP_HEIGHT = 485
CHAT_SCROLL_HEIGHT = 410
ALERT_HEIGHT = 512
```

아래로 교체:

```python
PANEL_HEIGHT = 625
LEFT_MAP_TOP_HEIGHT = 492
GRAPH_PANEL_HEIGHT = 126
MAP_HEIGHT = 355
CHAT_SCROLL_HEIGHT = 410
ALERT_HEIGHT = 380

GRAPH_START_HORIZON = 1
GRAPH_STEP_COUNT = 12
```

## 2) `calc_zone_metrics()` 아래에 그래프용 함수 추가

```python
def build_forecast_graph_payload(
    store: PredictionStore,
    sample_idx: int,
    zone_id: Optional[str],
    area_info: pd.DataFrame,
    model_label: str,
    has_query: bool,
) -> Dict[str, Any]:
    start_h = GRAPH_START_HORIZON
    end_h = min(store.n_horizons, GRAPH_START_HORIZON + GRAPH_STEP_COUNT)

    if end_h <= start_h:
        start_h = 0
        end_h = min(store.n_horizons, GRAPH_STEP_COUNT)

    horizons = list(range(start_h, end_h))

    if has_query and zone_id is not None and zone_id in store.zone_ids:
        zone_idx = store.zone_index(zone_id)
        values = [float(store.pred_value(sample_idx, h, zone_idx)) for h in horizons]
        selected_label, _ = get_selected_area(area_info, zone_id)
        title = f"{selected_label} 6시간 충전수요 예측"
        subtitle = f"{model_label} · 선택 생활권 기준"
    else:
        values = [
            float(np.nansum(store.pred[int(sample_idx), store.horizon_safe(h), :]))
            for h in horizons
        ]
        title = "서울시 전체 6시간 충전수요 예측"
        subtitle = f"{model_label} · 전체 생활권 합산"

    labels = [store.target_datetime(sample_idx, h).strftime("%H:%M") for h in horizons]
    base_dt = store.sample_to_datetime(sample_idx)

    return {
        "title": title,
        "subtitle": subtitle,
        "base_time": base_dt.strftime("%Y-%m-%d %H:%M"),
        "labels": labels,
        "values": [round(v, 3) for v in values],
        "unit": "kWh",
    }


def render_forecast_graph_html(payload: Dict[str, Any], height: int) -> None:
    payload_json = json.dumps(payload, ensure_ascii=False)

    html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                width: 100%;
                height: {height}px;
                background: transparent;
                font-family: Inter, -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Noto Sans KR", sans-serif;
                overflow: hidden;
            }}
            .graph-wrap {{
                width: 100%;
                height: {height}px;
                box-sizing: border-box;
                padding: 11px 16px 9px 16px;
                background: #FFFFFF;
                overflow: hidden;
            }}
            .graph-head {{
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                gap: 12px;
                margin-bottom: 4px;
            }}
            .graph-kicker {{
                color: #1F78B4;
                font-family: "Holtwood One SC", Georgia, serif;
                font-size: 10px;
                line-height: 1;
                margin-bottom: 3px;
            }}
            .graph-title {{
                color: #111111;
                font-size: 17px;
                font-weight: 900;
                letter-spacing: -0.045em;
                line-height: 1.05;
                white-space: nowrap;
            }}
            .graph-subtitle {{
                color: #69717C;
                font-size: 11px;
                font-weight: 600;
                line-height: 1.2;
                margin-top: 2px;
                text-align: right;
                white-space: nowrap;
            }}
            #chart {{
                width: 100%;
                height: calc({height}px - 48px);
                display: block;
            }}
            .axis-label {{
                fill: #7B8491;
                font-size: 10px;
                font-weight: 650;
            }}
            .value-label {{
                fill: #111111;
                font-size: 10px;
                font-weight: 750;
            }}
            .grid-line {{
                stroke: rgba(20,20,20,0.08);
                stroke-width: 1;
            }}
            .line {{
                fill: none;
                stroke: #1F78B4;
                stroke-width: 3;
                stroke-linecap: round;
                stroke-linejoin: round;
            }}
            .area {{
                fill: url(#areaGrad);
            }}
            .dot {{
                fill: #1F78B4;
                stroke: #FFFFFF;
                stroke-width: 2;
            }}
        </style>
    </head>
    <body>
        <div class="graph-wrap">
            <div class="graph-head">
                <div>
                    <div class="graph-kicker">12-STEP FORECAST</div>
                    <div class="graph-title" id="title"></div>
                </div>
                <div class="graph-subtitle" id="subtitle"></div>
            </div>
            <svg id="chart"></svg>
        </div>

        <script>
            const payload = {payload_json};

            document.getElementById("title").textContent = payload.title || "";
            document.getElementById("subtitle").innerHTML =
                `${{payload.subtitle || ""}}<br/>기준 ${{payload.base_time || ""}}`;

            const labels = payload.labels || [];
            const values = payload.values || [];
            const unit = payload.unit || "kWh";
            const svg = document.getElementById("chart");

            function draw() {{
                while (svg.firstChild) svg.removeChild(svg.firstChild);

                const width = svg.clientWidth || 720;
                const height = svg.clientHeight || 72;
                const margin = {{ left: 42, right: 18, top: 12, bottom: 22 }};
                const innerW = Math.max(10, width - margin.left - margin.right);
                const innerH = Math.max(10, height - margin.top - margin.bottom);

                const maxV = Math.max(...values, 1);
                const minV = Math.min(...values, 0);
                const pad = Math.max((maxV - minV) * 0.12, maxV * 0.05, 1);
                const yMax = maxV + pad;
                const yMin = Math.max(0, minV - pad);

                const x = i => margin.left + (values.length <= 1 ? 0 : i * innerW / (values.length - 1));
                const y = v => margin.top + (yMax - v) * innerH / Math.max(yMax - yMin, 1e-9);

                const ns = "http://www.w3.org/2000/svg";

                const defs = document.createElementNS(ns, "defs");
                defs.innerHTML = `
                    <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stop-color="#1F78B4" stop-opacity="0.22"/>
                        <stop offset="100%" stop-color="#1F78B4" stop-opacity="0.02"/>
                    </linearGradient>
                `;
                svg.appendChild(defs);

                [0.25, 0.5, 0.75].forEach(t => {{
                    const yy = margin.top + innerH * t;
                    const line = document.createElementNS(ns, "line");
                    line.setAttribute("x1", margin.left);
                    line.setAttribute("x2", margin.left + innerW);
                    line.setAttribute("y1", yy);
                    line.setAttribute("y2", yy);
                    line.setAttribute("class", "grid-line");
                    svg.appendChild(line);
                }});

                const points = values.map((v, i) => [x(i), y(v)]);
                const lineD = points.map((p, i) => `${{i === 0 ? "M" : "L"}} ${{p[0]}} ${{p[1]}}`).join(" ");
                const baseY = margin.top + innerH;
                const areaD = lineD + ` L ${{x(values.length - 1)}} ${{baseY}} L ${{x(0)}} ${{baseY}} Z`;

                const area = document.createElementNS(ns, "path");
                area.setAttribute("d", areaD);
                area.setAttribute("class", "area");
                svg.appendChild(area);

                const path = document.createElementNS(ns, "path");
                path.setAttribute("d", lineD);
                path.setAttribute("class", "line");
                svg.appendChild(path);

                points.forEach((p, i) => {{
                    const c = document.createElementNS(ns, "circle");
                    c.setAttribute("cx", p[0]);
                    c.setAttribute("cy", p[1]);
                    c.setAttribute("r", i === 0 || i === points.length - 1 ? 4.2 : 3.3);
                    c.setAttribute("class", "dot");
                    svg.appendChild(c);
                }});

                const maxIdx = values.indexOf(maxV);
                const valueText = document.createElementNS(ns, "text");
                valueText.setAttribute("x", x(maxIdx));
                valueText.setAttribute("y", Math.max(10, y(maxV) - 7));
                valueText.setAttribute("text-anchor", "middle");
                valueText.setAttribute("class", "value-label");
                valueText.textContent = `${{maxV.toFixed(1)}} ${{unit}}`;
                svg.appendChild(valueText);

                labels.forEach((lab, i) => {{
                    if (i % 2 !== 0 && i !== labels.length - 1) return;
                    const txt = document.createElementNS(ns, "text");
                    txt.setAttribute("x", x(i));
                    txt.setAttribute("y", height - 5);
                    txt.setAttribute("text-anchor", "middle");
                    txt.setAttribute("class", "axis-label");
                    txt.textContent = lab;
                    svg.appendChild(txt);
                }});
            }}

            draw();
            window.addEventListener("resize", draw);
        </script>
    </body>
    </html>
    '''
    components.html(html, height=height, scrolling=False)
```

## 3) `selected_detail_html` 계산 아래에 그래프 payload 추가

```python
forecast_graph_payload = build_forecast_graph_payload(
    store=store,
    sample_idx=sample_idx,
    zone_id=selected_zone_id,
    area_info=area_info,
    model_label=model_label,
    has_query=bool(st.session_state.has_query),
)
```

## 4) 메인 레이아웃 구조 교체

기존 `alert_col, map_col, chat_col = st.columns(...)`부터 마지막까지의 메인 레이아웃을 아래 구조로 바꾸세요.

```python
outer_left_col, outer_chat_col = st.columns([2.28, 0.78], gap="small")

with outer_left_col:
    top_alert_col, top_map_col = st.columns([0.86, 1.42], gap="small")

    with top_alert_col:
        with st.container(border=True, height=LEFT_MAP_TOP_HEIGHT):
            mark_panel()
            panel_title(
                "수요 급증알림",
                "30분 후 예측수요가 높은 생활권을 보여줍니다.",
                kicker="DEMAND WATCH",
            )
            draw_alerts_stack(top10, selected_time)

    with top_map_col:
        with st.container(border=True, height=LEFT_MAP_TOP_HEIGHT):
            mark_panel()
            header_left, header_right = st.columns([0.72, 0.28], gap="small")

            with header_left:
                panel_title(
                    "충전수요지도",
                    f"기준 {selected_dt:%Y-%m-%d %H:%M} · 예측 {target_dt:%H:%M} · {model_label} 기반 서울시 생활권별 예측 충전수요",
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
                map_gdf,
                st.session_state.use_3d_column,
                focus_zone_id,
                previous_focus_zone_id,
            )
            should_animate = bool(st.session_state.animate_zoom and st.session_state.has_query)
            render_deck_map_html(map_payload, should_animate, MAP_HEIGHT)

            if st.session_state.animate_zoom:
                st.session_state.animate_zoom = False

            render_legend()

    with st.container(border=True, height=GRAPH_PANEL_HEIGHT):
        mark_panel()
        render_forecast_graph_html(forecast_graph_payload, GRAPH_PANEL_HEIGHT - 4)


with outer_chat_col:
    with st.container(border=True, height=PANEL_HEIGHT):
        mark_panel()

        title_col, model_col = st.columns([0.50, 0.50], gap="small")
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
            render_chat_panel(
                st.session_state.messages,
                selected_detail_html=selected_detail_html,
                is_typing=False,
            )

        st.markdown('<div class="chat-form-wrap">', unsafe_allow_html=True)
        with st.form("chat_form", clear_on_submit=True):
            user_text = st.text_input(
                "질문 입력",
                placeholder=f"예: {FIXED_QUERY_EXAMPLE}",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("질문하기", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if submitted and user_text.strip():
            clean_user_text = user_text.strip()
            st.session_state.messages.append({"role": "user", "content": clean_user_text})

            with chat_placeholder:
                render_chat_panel(
                    st.session_state.messages,
                    selected_detail_html=None,
                    is_typing=True,
                )

            parsed = parse_user_query(
                clean_user_text,
                store,
                area_info,
                st.session_state.selected_date,
                st.session_state.selected_time,
                st.session_state.selected_zone_id,
            )

            if not parsed["ok"]:
                answer = build_conversational_answer(
                    clean_user_text,
                    parsed["message"],
                    parsed["reason"],
                    parsed.get("llm_extract"),
                    st.session_state.messages,
                    store,
                    model_label,
                )
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.show_selected_detail = False
                st.session_state.animate_zoom = False
                st.rerun()

            next_sample_idx = store.date_time_to_sample_idx(parsed["date"], parsed["time"])
            if next_sample_idx is None:
                st.session_state.messages.append(
                    {"role": "assistant", "content": "선택한 날짜와 시간에 해당하는 예측 데이터가 없습니다."}
                )
                st.rerun()

            if st.session_state.has_query:
                st.session_state.previous_focus_zone_id = st.session_state.selected_zone_id
            else:
                st.session_state.previous_focus_zone_id = None

            next_metrics = calc_zone_metrics(store, next_sample_idx, parsed["zone_id"])
            next_label, next_dongs = get_selected_area(area_info, parsed["zone_id"])
            next_top10 = (
                store.slice_df(next_sample_idx, store.horizon_safe(PREDICT_HORIZON_30M))
                .sort_values("predicted_kwh", ascending=False)
                .head(10)
            )
            next_top10 = next_top10.merge(
                area_info[["생활권역ID", "생활권역라벨", "생활권역표시명"]],
                on="생활권역ID",
                how="left",
            )

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
            st.markdown(
                f'<div class="llm-error-caption">LLM error: {escape_html(st.session_state.last_llm_error)}</div>',
                unsafe_allow_html=True,
            )
```
