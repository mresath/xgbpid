"""
XGBPID | Live Global Dashboard

A public-facing view of the CERN T9 beamline experiment, built for students,
physicists, and anyone curious about what real particle data looks like.
It shows the same telemetry that feeds the internal control room, stripped
down to the parts that are actually interesting to an outside audience.

The acquisition loop at CERN uploads a rolling Parquet snapshot to Supabase
every 30 seconds; this dashboard fetches it and re-renders automatically.

Usage:
    streamlit run public/dashboard.py
"""

import io
import time

import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
import streamlit as st
from supabase import create_client, Client

_NS_PER_SAMPLE: float = 8.0  # 125 MHz at decimation=1 → 8 ns between ADC samples
_PARTICLE_COLORS = {"electron": "#4C9BE8", "pion": "#E8704C", "kaon": "#A855F7"}

_SUPABASE_URL = "https://lnpudktmbrjbrvpdkjdf.supabase.co"
_SUPABASE_ANON_KEY: str = st.secrets.get("SUPABASE_KEY", "")

_supabase: Client | None = (
    create_client(_SUPABASE_URL, _SUPABASE_ANON_KEY) if _SUPABASE_ANON_KEY else None
)

_CONF_THRESHOLD: float = 0.70
_LABEL_MAP: dict[int, str] = {0: "pion", 1: "electron", 2: "kaon"}


@st.cache_data(ttl=5, show_spinner=False)
def _load_telemetry() -> pl.DataFrame | None:
    """Fetch the latest telemetry snapshot from Supabase. Returns None on failure."""
    if _supabase is None:
        return None
    try:
        data = _supabase.storage.from_("data").download("live_telemetry.parquet")
        return pl.read_parquet(io.BytesIO(data))
    except Exception:
        return None


def _synthesize_waveform(row: dict) -> np.ndarray:
    """Reconstruct an illustrative pulse shape from extracted features.

    The asymmetric bi-exponential model captures the physics without raw ADC
    storage: tau_rise drives the fast Cherenkov/EM onset, while the extended
    tail controlled by ``tail_to_total`` reflects the hadronic cascade length.

    K⁻ pulses have a systematically longer decay than π⁻ because kaons carry
    more hadronic cascade energy — their nuclear interaction length in the
    scintillator is ~10% greater than pions at 5–10 GeV/c.
    """
    n = 256  # 256 samples × 8 ns = 2048 ns window, wide enough to capture kaon tails
    t = np.arange(n, dtype=np.float32)

    peak_amp = max(float(row.get("peak_amplitude", 0.5)), 0.01)
    rise_ns  = max(float(row.get("rise_time_ns", 8.0)), 1.0)
    fwhm_ns  = max(float(row.get("fwhm_ns", 40.0)), rise_ns * 1.5)

    tau_r = rise_ns / (2.3 * _NS_PER_SAMPLE)
    tau_d = fwhm_ns / (1.386 * _NS_PER_SAMPLE)

    onset = int(n * 0.20)
    ts = np.maximum(t - onset, 0.0)
    waveform = peak_amp * (1.0 - np.exp(-ts / tau_r)) * np.exp(-ts / tau_d)

    rng = np.random.default_rng(seed=int(row.get("event_id", 0)) & 0xFFFF)
    waveform += rng.normal(0, max(float(row.get("baseline_rms", 0.005)), 1e-4), n)
    return waveform


def _live_pulse_viewer(df: pl.DataFrame) -> None:
    st.subheader("Live Pulse Viewer")
    st.caption(
        "Reconstructed waveform of the most recently classified particle — "
        "synthesised from the pulse-shape features the XGBoost model was trained on."
    )
    latest = df.tail(1).to_dicts()[0]
    wave = _synthesize_waveform(latest)
    time_ns = np.arange(len(wave)) * _NS_PER_SAMPLE

    particle = latest.get("label", "unknown")
    color = _PARTICLE_COLORS.get(particle, "#888888")

    pulse_df = pd.DataFrame({"Time (ns)": time_ns, f"{particle} pulse (V)": wave})
    st.line_chart(pulse_df.set_index("Time (ns)"), color=color, height=220)

    conf = latest.get("confidence", 0.0)
    above = latest.get("above_threshold", False)
    evt   = latest.get("event_id", "—")
    st.caption(
        f"Event {evt} \u2192 **{particle}** | confidence {conf:.3f} "
        f"{'\u2713' if above else '\u26a0 low confidence'}"
    )


def _species_counter(df: pl.DataFrame) -> None:
    st.subheader("Global Species Counter")
    st.caption("Cumulative particle counts since the start of this run.")

    n_e = int((df["label"] == "electron").sum())
    n_pi = int((df["label"] == "pion").sum())
    n_k  = int((df["label"] == "kaon").sum())
    total = len(df)

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f"<div style='text-align:center; color:{_PARTICLE_COLORS['electron']}'>"
        f"<span style='font-size:3rem; font-weight:bold'>{n_e:,}</span><br>"
        f"<span style='font-size:1.1rem'>⚡ Electrons</span><br>"
        f"<span style='font-size:0.85rem; color:grey'>{100*n_e/total:.1f} % of beam</span></div>",
        unsafe_allow_html=True,
    )
    c2.markdown(
        f"<div style='text-align:center; color:{_PARTICLE_COLORS['pion']}'>"
        f"<span style='font-size:3rem; font-weight:bold'>{n_pi:,}</span><br>"
        f"<span style='font-size:1.1rem'>🔴 Pions</span><br>"
        f"<span style='font-size:0.85rem; color:grey'>{100*n_pi/total:.1f} % of beam</span></div>",
        unsafe_allow_html=True,
    )
    c3.markdown(
        f"<div style='text-align:center; color:{_PARTICLE_COLORS['kaon']}'>"
        f"<span style='font-size:3rem; font-weight:bold'>{n_k:,}</span><br>"
        f"<span style='font-size:1.1rem'>🟣 Kaons</span><br>"
        f"<span style='font-size:0.85rem; color:grey'>{100*n_k/total:.2f} % of beam</span></div>",
        unsafe_allow_html=True,
    )


def _confidence_heatmap(df: pl.DataFrame) -> None:
    st.subheader("AI Confidence Heatmap")
    st.caption(
        "How confident is the XGBoost classifier across the measured pulse amplitude range? "
        "Dense bright regions mean the model recognises the particle type clearly."
    )
    pdf = df.select(["confidence", "peak_amplitude", "label"]).to_pandas()
    fig = px.density_heatmap(
        pdf,
        x="peak_amplitude",
        y="confidence",
        facet_col="label",
        color_continuous_scale="Viridis",
        labels={
            "peak_amplitude": "Peak amplitude (V)",
            "confidence": "Model confidence",
        },
        nbinsx=25,
        nbinsy=25,
        title="Classifier Confidence vs. Peak Amplitude",
    )
    fig.add_hline(
        y=_CONF_THRESHOLD,
        line_dash="dash",
        line_color="white",
        annotation_text=f"threshold {_CONF_THRESHOLD:.2f}",
        annotation_font_color="white",
        annotation_position="bottom right",
    )
    fig.update_layout(
        margin=dict(t=50, b=20),
        coloraxis_colorbar=dict(title="Count"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _science_corner() -> None:
    with st.sidebar:
        st.title("XGBPID")
        st.caption("Live Global Dashboard")
        st.divider()
        st.subheader("🔬 Science Corner")

        with st.expander("What is the Teacher–Student model?", expanded=True):
            st.markdown(
                """
The **Teacher** is a slow but highly accurate algorithm — typically a
full likelihood fit using multiple Cherenkov counters — that labels
each beam particle with high purity.

The **Student** is our XGBoost classifier.  It learns to reproduce
the Teacher's labels from only the fast, single-channel pulse-shape
features (rise time, FWHM, tail-to-total ratio, AUC).  Once trained,
the student runs in **< 1 ms per event** — fast enough for real-time
beam tagging without the Teacher's hardware overhead.

This is the essence of *knowledge distillation*: compress the
Teacher's expertise into a lightweight, deployable model.
                """
            )

        with st.expander("Why hunt for Kaons?", expanded=False):
            st.markdown(
                r"""
At **5–10 GeV/c** in the CERN T9 secondary beam, $K^-$ mesons
comprise only **~1.5 %** of all beam particles.  Yet kaons are
scientifically coveted:

- They are the lightest strange-quark carriers, offering a clean
  probe of **QCD strangeness production**.
- Their hadronic interactions in a scintillator produce a
  characteristically **slower rise time** and **longer decay tail**
  compared to pions — exactly what the pulse-shape features capture.
- Contamination from misidentified pions is the dominant systematic
  — the XGBoost model is explicitly optimised to maximise the
  **Pion Rejection Factor (PRF)**.

Spotting a $K^-$ in a flood of $\pi^-$ is like identifying one
Turkish coffee bean in a sack of espresso.
                """
            )

        st.divider()
        st.caption(
            "Follow us from anywhere — this dashboard is designed "
            "to load on a mobile phone. No CERN VPN needed."
        )



def _beam_status_banner(df: pl.DataFrame | None) -> None:
    if df is not None and len(df) > 0:
        label, colour = "LIVE", "green"
    else:
        label, colour = "OFFLINE", "red"
    st.markdown(
        f"<div style='text-align:center; padding:6px 0; "
        f"background:rgba(0,0,0,0.08); border-radius:6px'>"
        f"<span style='font-size:1.3rem; font-weight:bold; color:{colour}'>"
        f"⬤ {label}</span></div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="XGBPID | Live Global",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _science_corner()

    st.title("XGBPID — Live Global")
    st.caption(
        "Real-time particle identification on the CERN T9 beamline, "
        "open to the world."
    )

    df = _load_telemetry()

    _beam_status_banner(df)
    st.divider()

    if df is None or len(df) == 0:
        st.info(
            "Waiting for the first telemetry flush…\n\n"
            "The run will populate this dashboard automatically "
            "once the acquisition loop starts.",
            icon="📡",
        )
        time.sleep(5)
        st.rerun()
        return

    _species_counter(df)
    st.divider()

    _live_pulse_viewer(df)
    st.divider()

    _confidence_heatmap(df)

    time.sleep(5)
    st.rerun()


if __name__ == "__main__":
    main()
