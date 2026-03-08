import time
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

from core.detector import PersonDetector
from core.optical_flow import OpticalFlowEngine
from core.physics import PhysicsEngine
from core.forecaster import Forecaster
from core.alert_engine import AlertEngine


st.set_page_config(
    page_title="FluidFlow",
    page_icon="wave",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.status-green  {
    background:#1a5e35;
    color:white;
    padding:12px 20px;
    border-radius:8px;
    font-size:18px;
    font-weight:bold;
    text-align:center;
}

.status-yellow {
    background:#7a5c00;
    color:white;
    padding:12px 20px;
    border-radius:8px;
    font-size:18px;
    font-weight:bold;
    text-align:center;
}

.status-red {
    background:#7a0000;
    color:white;
    padding:12px 20px;
    border-radius:8px;
    font-size:18px;
    font-weight:bold;
    text-align:center;
    animation:blink 0.8s infinite;
}

@keyframes blink {
    0%{opacity:1}
    50%{opacity:0.3}
    100%{opacity:1}
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_detector():
    return PersonDetector(conf=0.35)


@st.cache_resource
def load_forecaster():
    return Forecaster("models/lstm_forecaster.pt")


@st.cache_resource
def load_alert_engine():
    return AlertEngine("db/alerts.db")


@st.cache_data
def load_cache(video_name: str) -> dict:
    base = Path("cache") / video_name

    return {
        "density": np.load(base / "density.npy", mmap_mode="r"),
        "pressure": np.load(base / "pressure.npy", mmap_mode="r"),
        "ti": np.load(base / "ti_series.npy", mmap_mode="r"),
        "re": np.load(base / "re_series.npy", mmap_mode="r"),
        "shock": np.load(base / "shock_series.npy", mmap_mode="r"),
    }


def render_heatmap(pressure: np.ndarray) -> go.Figure:

    fig = go.Figure(
        go.Heatmap(
            z=pressure,
            colorscale=[
                [0, "#1a1a2e"],
                [0.4, "#16213e"],
                [0.7, "#e94560"],
                [1, "#ff0000"],
            ],
            zmin=0,
            zmax=1,
            showscale=True,
        )
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=24, b=0),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        title=dict(text="Pressure Field", font=dict(color="white", size=13)),
        height=300,
    )

    return fig


def render_gauge(TI: float, predicted_TI: float) -> go.Figure:

    color = "#00cc44" if TI < 0.15 else ("#ffaa00" if TI < 0.30 else "#ff3300")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=round(TI, 3),
            delta={"reference": predicted_TI},
            gauge={
                "axis": {"range": [0, 0.5]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 0.15], "color": "#0d1b2a"},
                    {"range": [0.15, 0.30], "color": "#1a1200"},
                    {"range": [0.30, 0.50], "color": "#1a0000"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.8,
                    "value": predicted_TI,
                },
            },
            title={"text": "Turbulence Intensity", "font": {"color": "white"}},
            number={"font": {"color": "white"}},
        )
    )

    fig.update_layout(
        paper_bgcolor="#0E1117",
        height=260,
        margin=dict(l=30, r=30, t=60, b=0),
    )

    return fig


def status_html(level: int) -> str:

    labels = {
        0: "<div class='status-green'>LAMINAR FLOW - Safe</div>",
        1: "<div class='status-yellow'>TRANSITIONAL - Monitor Closely</div>",
        2: "<div class='status-red'>TURBULENT - EVACUATE CORRIDOR</div>",
    }

    return labels[level]


def main():

    st.sidebar.title("FluidFlow")

    video_choice = st.sidebar.selectbox(
        "Select Scenario",
        ["normal_flow", "transitional", "stampede_precursor"],
    )

    fps = st.sidebar.slider("Playback FPS", 1, 20, 10)
    show_log = st.sidebar.checkbox("Show Alert Log", True)

    detector = load_detector()
    forecaster = load_forecaster()
    alert_engine = load_alert_engine()

    cache = load_cache(video_choice)
    total = len(cache["ti"])

    st.title("FluidFlow - Live Crowd Safety Monitor")

    col_vid, col_viz = st.columns([1, 1])

    with col_vid:
        st.subheader("Video Feed")

        status_ph = st.empty()
        video_ph = st.empty()
        metrics_ph = st.empty()

    with col_viz:
        st.subheader("Pressure + Turbulence")

        heatmap_ph = st.empty()
        gauge_ph = st.empty()

    log_ph = st.empty()

    cap = cv2.VideoCapture(f"data/videos/{video_choice}.mp4")

    delay = 1.0 / fps

    for fi in range(total):
        ret, frame = cap.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(
            cv2.resize(frame, (640, 360)),
            cv2.COLOR_BGR2RGB,
        )

        TI = float(cache["ti"][fi])
        Re = float(cache["re"][fi])
        pressure = cache["pressure"][fi]
        density = cache["density"][fi]
        shock = int(cache["shock"][fi])

        pred_TI = forecaster.update(
            TI,
            Re,
            float(density.max()),
            float(pressure.max()),
        )

        event = alert_engine.evaluate(
            fi,
            {
                "TI": TI,
                "Re": Re,
                "density_max": float(density.max()),
                "pressure_max": float(pressure.max()),
                "shockwave_flag": shock,
            },
            pred_TI,
        )

        status_ph.markdown(status_html(event.level), unsafe_allow_html=True)

        video_ph.image(frame_rgb, use_column_width=True)

        metrics_ph.markdown(
            f"TI: {TI:.3f} | Re: {Re:.0f} | Density: {density.max():.1f} p/m2"
            f" | Pred TI+30s: {pred_TI:.3f}"
        )

        heatmap_ph.plotly_chart(
            render_heatmap(pressure),
            use_container_width=True,
            key=f"hm_{fi}",
        )

        gauge_ph.plotly_chart(
            render_gauge(TI, pred_TI),
            use_container_width=True,
            key=f"g_{fi}",
        )

        if show_log:
            log = alert_engine.get_log()

            if log:
                log_ph.dataframe(
                    pd.DataFrame(log),
                    use_container_width=True,
                    height=180,
                )

        time.sleep(delay)

    cap.release()

    st.success("Playback complete.")


if __name__ == "__main__" or True:
    main()
