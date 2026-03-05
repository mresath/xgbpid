"""
Live Monitoring Dashboard

Runs as a separate OS process from main.py; all configuration is read from the
same experiment YAML used by the acquisition loop.  main.py atomically
overwrites ``data/live_telemetry.parquet`` every ``LIVE_FLUSH_INTERVAL`` events
via write-to-temp-then-rename, so this dashboard always reads a complete file
with no coordination primitives required.

Usage:
    streamlit run dashboard.py
    streamlit run dashboard.py -- --config configs/experiment_v1.yaml
"""

import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

_LATENCY_BUDGET_US = 1_000.0   # µs — not user-configurable
_PARTICLE_COLORS   = {"electron": "#4C9BE8", "pion": "#E8704C", "kaon": "#A855F7"}


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--config",
        default="configs/experiment_v1.yaml",
        help="Path to experiment YAML config (default: configs/experiment_v1.yaml)",
    )
    known, _ = p.parse_known_args(sys.argv[1:])
    return known


_cfg           = _load_config(_parse_args().config)
_TELEMETRY_PATH      = Path(_cfg["logging"]["output_dir"]).parent / "live_telemetry.parquet"
_KAON_TELEMETRY_PATH = Path(_cfg["logging"]["output_dir"]).parent / "kaon_telemetry.parquet"
_DATA_DIR       = Path(_cfg["logging"]["output_dir"]).parent
_MODEL_PATH     = Path(_cfg["model"]["path"])
_PAUSE_FILE     = _DATA_DIR / "retrain.pause"
_STATUS_FILE    = _DATA_DIR / "retrain_status.json"
_RETRAIN_INTERVAL_S = int(_cfg.get("retraining", {}).get("interval_seconds", 120))
_CONF_THRESHOLD = float(_cfg["model"]["confidence_threshold"])
_DEFAULT_REFRESH = int(_cfg["dashboard"]["refresh_seconds"])
_DEFAULT_WINDOW  = int(_cfg["dashboard"]["latency_window_events"])
_LABEL_MAP       = {int(k): v for k, v in _cfg["model"]["labels"].items()}
_VALIDATION_WINDOW = int(_cfg.get("validation", {}).get("window", 500))
_RELAY_STATUS_FILE = _DATA_DIR / "relay_status.json"
_RELAY_INTERVAL_S  = int(_cfg.get("relay", {}).get("upload_interval_seconds", 30))
_RELAY_ENABLED     = bool(_cfg.get("relay", {}).get("enabled", False))


@st.cache_data(ttl=2, show_spinner=False)
def _load(path_str: str) -> pl.DataFrame | None:
    """Read the rolling telemetry Parquet, returning None if not yet created."""
    p = Path(path_str)
    if not p.exists():
        return None
    try:
        return pl.read_parquet(p)
    except Exception:
        return None


def _load_retrain_status() -> dict | None:
    """Read the retrainer status JSON, returning None if not yet written."""
    if not _STATUS_FILE.exists():
        return None
    try:
        return json.loads(_STATUS_FILE.read_text())
    except Exception:
        return None


def _load_relay_status() -> dict | None:
    """Read the relay status JSON, returning None if not yet written."""
    if not _RELAY_STATUS_FILE.exists():
        return None
    try:
        return json.loads(_RELAY_STATUS_FILE.read_text())
    except Exception:
        return None


def _set_retrain_paused(paused: bool) -> None:
    """Write or remove the sentinel file that pauses the retrainer process."""
    if paused:
        _PAUSE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _PAUSE_FILE.touch()
    elif _PAUSE_FILE.exists():
        _PAUSE_FILE.unlink()


def _metric_row(df: pl.DataFrame) -> None:
    total      = len(df)
    n_electron = (df["label"] == "electron").sum()
    n_pion     = (df["label"] == "pion").sum()
    n_kaon     = (df["label"] == "kaon").sum()
    e_pct      = 100.0 * n_electron / total if total else 0.0
    pi_pct     = 100.0 * n_pion / total if total else 0.0
    k_pct      = 100.0 * n_kaon / total if total else 0.0
    mean_lat   = df["latency_us"].mean() or 0.0
    p99_lat    = df["latency_us"].quantile(0.99) or 0.0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total events",  f"{total:,}")
    c2.metric("Electrons",     f"{n_electron:,}",  f"{e_pct:.1f} %")
    c3.metric("Pions",         f"{n_pion:,}",      f"{pi_pct:.1f} %")
    c4.metric("Kaons",         f"{n_kaon:,}",      f"{k_pct:.2f} %")
    c5.metric("Mean latency",  f"{mean_lat:.1f} µs")

    p99_delta = f"budget {'OK' if p99_lat <= _LATENCY_BUDGET_US else 'EXCEEDED'}"
    c6.metric(
        "p99 latency",
        f"{p99_lat:.1f} µs",
        p99_delta,
        delta_color="normal" if p99_lat <= _LATENCY_BUDGET_US else "inverse",
    )


def _particle_distribution(df: pl.DataFrame) -> go.Figure:
    counts = (
        df.group_by("label")
        .agg(pl.len().alias("count"))
        .sort("label")
    )
    fig = px.bar(
        counts.to_pandas(),
        x="label",
        y="count",
        color="label",
        color_discrete_map=_PARTICLE_COLORS,
        labels={"label": "Particle", "count": "Events"},
        title="Particle Distribution",
        text_auto=True,
    )
    fig.update_layout(showlegend=False, margin=dict(t=40, b=10))
    return fig


def _confidence_distribution(df: pl.DataFrame) -> go.Figure:
    fig = px.histogram(
        df.to_pandas(),
        x="confidence",
        color="label",
        color_discrete_map=_PARTICLE_COLORS,
        nbins=40,
        barmode="overlay",
        opacity=0.75,
        labels={"confidence": "Classifier confidence", "count": "Events"},
        title="Confidence Score Distribution",
    )
    fig.add_vline(
        x=_CONF_THRESHOLD,
        line_dash="dash",
        line_color="grey",
        annotation_text=f"threshold {_CONF_THRESHOLD:.2f}",
        annotation_position="top right",
    )
    fig.update_layout(margin=dict(t=40, b=10))
    return fig


def _tail_to_total_distribution(df: pl.DataFrame) -> go.Figure:
    fig = px.histogram(
        df.to_pandas(),
        x="tail_to_total",
        color="label",
        color_discrete_map=_PARTICLE_COLORS,
        nbins=50,
        barmode="overlay",
        opacity=0.75,
        labels={"tail_to_total": "Tail-to-total ratio (PSD)", "count": "Events"},
        title="Pulse-Shape Discrimination — Tail-to-Total Ratio",
    )
    fig.update_layout(margin=dict(t=40, b=10))
    return fig


def _latency_histogram(df: pl.DataFrame) -> go.Figure:
    fig = px.histogram(
        df.to_pandas(),
        x="latency_us",
        nbins=60,
        color_discrete_sequence=["#6DB8B8"],
        labels={"latency_us": "Latency (µs)", "count": "Events"},
        title="End-to-End Latency Distribution",
    )
    fig.add_vline(
        x=_LATENCY_BUDGET_US,
        line_dash="dash",
        line_color="red",
        annotation_text="1 ms budget",
        annotation_position="top right",
    )
    fig.update_layout(margin=dict(t=40, b=10))
    return fig


def _latency_over_time(df: pl.DataFrame, window: int) -> go.Figure:
    tail = df.tail(window)
    fig = px.line(
        tail.to_pandas(),
        x="event_id",
        y="latency_us",
        color_discrete_sequence=["#6DB8B8"],
        labels={"event_id": "Event ID", "latency_us": "Latency (µs)"},
        title=f"Rolling Latency — last {window} events",
    )
    fig.add_hline(
        y=_LATENCY_BUDGET_US,
        line_dash="dash",
        line_color="red",
        annotation_text="1 ms",
        annotation_position="top right",
    )
    fig.update_layout(margin=dict(t=40, b=10))
    return fig


def _confusion_matrix(val_df: pl.DataFrame) -> go.Figure:
    """Heatmap of true (teacher) vs predicted (student) particle labels."""
    cm_df = (
        val_df
        .with_columns(
            pl.col("teacher_label")
            .cast(pl.Int32)
            .map_elements(lambda x: _LABEL_MAP.get(x, str(x)), return_dtype=pl.Utf8)
            .alias("true_label")
        )
        .group_by(["true_label", "label"])
        .agg(pl.len().alias("count"))
    )
    label_names = sorted(_LABEL_MAP.values())
    idx = {name: i for i, name in enumerate(label_names)}
    matrix = [[0] * len(label_names) for _ in label_names]
    for row in cm_df.to_dicts():
        i = idx.get(row["true_label"], -1)
        j = idx.get(row["label"], -1)
        if i >= 0 and j >= 0:
            matrix[i][j] = row["count"]
    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=[f"Pred: {n}" for n in label_names],
        y=[f"True: {n}" for n in label_names],
        colorscale="Blues",
        text=[[str(v) for v in row] for row in matrix],
        texttemplate="%{text}",
        showscale=False,
    ))
    fig.update_layout(title="Live Confusion Matrix", margin=dict(t=40, b=10))
    return fig


def _rolling_accuracy(val_df: pl.DataFrame, window: int) -> go.Figure:
    """Line chart of rolling classification accuracy against teacher labels."""
    acc_df = (
        val_df
        .filter(pl.col("is_correct").is_not_null())
        .with_columns(pl.col("is_correct").cast(pl.Float32))
        .with_columns(
            pl.col("is_correct")
            .rolling_mean(window_size=window, min_periods=1)
            .alias("rolling_accuracy")
        )
    )
    fig = px.line(
        acc_df.to_pandas(),
        x="event_id",
        y="rolling_accuracy",
        color_discrete_sequence=["#4C9BE8"],
        labels={"event_id": "Event ID", "rolling_accuracy": "Accuracy"},
        title=f"Rolling Accuracy — window {window}",
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(margin=dict(t=40, b=10))
    return fig


def _teacher_student_comparison(val_df: pl.DataFrame) -> go.Figure:
    """Grouped bar chart overlaying teacher vs student label distributions."""
    teacher_counts = (
        val_df
        .with_columns(
            pl.col("teacher_label")
            .cast(pl.Int32)
            .map_elements(lambda x: _LABEL_MAP.get(x, str(x)), return_dtype=pl.Utf8)
            .alias("label_name")
        )
        .group_by("label_name")
        .agg(pl.len().alias("count"))
        .with_columns(pl.lit("Teacher").alias("source"))
        .rename({"label_name": "label"})
    )
    student_counts = (
        val_df
        .group_by("label")
        .agg(pl.len().alias("count"))
        .with_columns(pl.lit("Student").alias("source"))
    )
    combined = pl.concat([teacher_counts, student_counts])
    fig = px.bar(
        combined.to_pandas(),
        x="label",
        y="count",
        color="source",
        barmode="group",
        color_discrete_map={"Teacher": "#888888", "Student": "#4C9BE8"},
        labels={"label": "Particle", "count": "Events", "source": "Source"},
        title="Teacher vs. Student Label Distribution",
    )
    fig.update_layout(margin=dict(t=40, b=10))
    return fig


def _latest_events_table(df: pl.DataFrame, n: int = 20) -> pl.DataFrame:
    return (
        df.tail(n)
        .select([
            "event_id", "label", "confidence", "above_threshold",
            "latency_us", "tail_to_total", "peak_amplitude",
            "rise_time_ns", "fwhm_ns", "teacher_label",
        ])
        .reverse()   # most-recent first
    )


def _relay_status_panel() -> None:
    """Show current cloud publishing state: enabled flag, upload count, last upload age, and any errors."""
    status = _load_relay_status()

    if not _RELAY_ENABLED:
        st.warning("Relay is **disabled** in config — telemetry is not being published.", icon="☁️")
        return

    if status is None:
        st.info("Relay is enabled but has not run yet. Start `python main.py` to begin publishing.", icon="☁️")
        return

    if not status.get("configured"):
        st.warning("Relay is enabled but Supabase credentials are missing — check your `.env` file.", icon="⚠️")
    elif status.get("last_error"):
        st.error(f"Last upload failed: `{status['last_error']}`", icon="❌")
    else:
        st.success("Publishing to Supabase is **active**.", icon="☁️")

    p1, p2, p3 = st.columns(3)
    p1.metric("Uploads this run", str(status.get("upload_count", 0)))

    last_ts = status.get("last_upload", 0)
    if last_ts:
        age_s = time.time() - last_ts
        age_str = f"{int(age_s)}s ago" if age_s < 120 else f"{int(age_s/60)}m ago"
    else:
        age_str = "never"
    p2.metric("Last upload", age_str)

    interval_s = status.get("interval_s", _RELAY_INTERVAL_S)
    if last_ts and status.get("configured") and not status.get("last_error"):
        next_s = max(0, int(interval_s - (time.time() - last_ts)))
        p3.metric("Next upload in", f"{next_s}s")
    else:
        p3.metric("Next upload in", f"{int(interval_s)}s")


def _retrain_status_panel(paused: bool) -> None:
    """Show current model file age, retraining cycle info, and feature importance."""
    status = _load_retrain_status()
    model_mtime = _MODEL_PATH.stat().st_mtime if _MODEL_PATH.exists() else None

    if paused:
        st.warning("Retraining is **paused** — the live model is frozen.", icon="⏸️")
    else:
        st.success("Retraining is **active**.", icon="⚡")

    r1, r2, r3, r4 = st.columns(4)
    if model_mtime is not None:
        age_s = time.time() - model_mtime
        age_str = f"{int(age_s)}s ago" if age_s < 120 else f"{int(age_s/60)}m ago"
        r1.metric("Model last updated", age_str)
    else:
        r1.metric("Model last updated", "unknown")

    if status:
        r2.metric("Retrain cycles", str(status["cycle"]))
        r3.metric("Mode", status["mode"].capitalize())
        next_s = max(0, int(status["interval_s"] - (time.time() - status["last_retrain"])))
        next_str = f"{next_s}s" if not paused else "Paused"
        r4.metric("Next cycle in", next_str)

        importance = status.get("feature_importance", {})
        if importance:
            imp_df = (
                pl.DataFrame({"feature": list(importance.keys()), "gain": list(importance.values())})
                .sort("gain", descending=True)
            )
            fig = px.bar(
                imp_df.to_pandas(),
                x="gain",
                y="feature",
                orientation="h",
                color_discrete_sequence=["#6DB8B8"],
                labels={"gain": "Importance (gain)", "feature": ""},
                title="Feature Importance — Latest Trained Model",
            )
            fig.update_layout(margin=dict(t=40, b=10), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
    else:
        r2.metric("Retrain cycles", "0")
        r3.metric("Mode", "—")
        r4.metric("Next cycle in", f"{_RETRAIN_INTERVAL_S}s" if not paused else "paused")


def _kaon_panel(kaon_df: pl.DataFrame) -> None:
    """Dedicated section for the kaon stream.

    Kaons are flushed on every occurrence into a separate file because they
    represent only ~1-2% of the beam.  This panel reads that dedicated stream
    so kaon events are never diluted by the rolling main-telemetry window.
    """
    st.subheader("Kaon Stream")
    total_k = len(kaon_df)
    mean_conf = kaon_df["confidence"].mean() or 0.0
    above_thresh = (kaon_df["above_threshold"] == True).sum()  # noqa: E712
    above_pct = 100.0 * above_thresh / total_k if total_k else 0.0

    k1, k2, k3 = st.columns(3)
    k1.metric("Kaon events captured", f"{total_k:,}")
    k2.metric("Mean confidence",      f"{mean_conf:.3f}")
    k3.metric("Above threshold",      f"{above_thresh:,}", f"{above_pct:.1f} %")

    col_k1, col_k2 = st.columns(2)
    with col_k1:
        fig_conf = px.histogram(
            kaon_df.to_pandas(),
            x="confidence",
            nbins=30,
            color_discrete_sequence=[_PARTICLE_COLORS["kaon"]],
            labels={"confidence": "Classifier confidence", "count": "Events"},
            title="Kaon Confidence Distribution",
        )
        fig_conf.add_vline(
            x=_CONF_THRESHOLD,
            line_dash="dash",
            line_color="grey",
            annotation_text=f"threshold {_CONF_THRESHOLD:.2f}",
            annotation_position="top right",
        )
        fig_conf.update_layout(margin=dict(t=40, b=10))
        st.plotly_chart(fig_conf, use_container_width=True)

    with col_k2:
        fig_psd = px.scatter(
            kaon_df.to_pandas(),
            x="tail_to_total",
            y="rise_time_ns",
            color_discrete_sequence=[_PARTICLE_COLORS["kaon"]],
            opacity=0.6,
            labels={
                "tail_to_total": "Tail-to-total ratio (PSD)",
                "rise_time_ns": "Rise time (ns)",
            },
            title="Kaon PSD Feature Space",
        )
        fig_psd.update_layout(margin=dict(t=40, b=10))
        st.plotly_chart(fig_psd, use_container_width=True)

    with st.expander("Latest kaon events (most recent first)", expanded=False):
        st.dataframe(
            kaon_df.tail(20)
            .select(["event_id", "label", "confidence", "above_threshold",
                     "tail_to_total", "rise_time_ns", "peak_amplitude", "teacher_label"])
            .reverse(),
            use_container_width=True,
            hide_index=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="XGBPID | Live",
        page_icon="assets/quarc-circular.png",
        layout="wide",
    )

    with st.sidebar:
        st.title("XGBPID - Team QuaRC")
        st.divider()

        st.subheader("Telemetry source")
        telemetry_path = st.text_input(
            "Parquet file",
            value=str(_TELEMETRY_PATH),
            help="Path to live_telemetry.parquet written by main.py",
        )
        kaon_telemetry_path = st.text_input(
            "Kaon stream file",
            value=str(_KAON_TELEMETRY_PATH),
            help="Path to kaon_telemetry.parquet (flushed on every kaon event)",
        )

        st.divider()
        st.subheader("Display settings")
        refresh_s = st.select_slider(
            "Auto-refresh interval",
            options=[1, 2, 5, 10, 30],
            value=_DEFAULT_REFRESH,
            format_func=lambda v: f"{v} s",
        )
        latency_window = st.slider(
            "Latency plot window (events)",
            min_value=50,
            max_value=2_000,
            value=_DEFAULT_WINDOW,
            step=50,
        )
        auto_refresh = st.checkbox("Auto-refresh", value=True)

        st.divider()
        st.subheader("Validation Mode")
        validation_mode = st.toggle("Enable Validation Mode", value=False)
        if validation_mode:
            st.caption("Retraining is automatically paused while validation is active.")

        st.divider()
        st.subheader("Retraining")
        manual_pause = st.toggle(
            "Pause retraining",
            value=_PAUSE_FILE.exists(),
            disabled=validation_mode,
            help="Pause the periodic retrainer. Automatically forced on by Validation Mode.",
        )
        retraining_paused = validation_mode or manual_pause
        _set_retrain_paused(retraining_paused)

    st.title("Live Monitor")
    st.caption(f"Telemetry file: `{telemetry_path}`")

    df = _load(telemetry_path)
    kaon_df = _load(kaon_telemetry_path)

    if df is None or len(df) == 0:
        st.info(
            "Waiting for telemetry data…\n\n"
            "Start `python main.py` and the dashboard will populate "
            f"after the first {50} events are flushed.",
            icon="📡",
        )
        if auto_refresh:
            time.sleep(refresh_s)
            st.rerun()
        return

    st.caption(f"Showing **{len(df):,}** events — refreshes every {refresh_s} s")

    _metric_row(df)
    st.divider()

    with st.expander("Model & Retraining", expanded=not retraining_paused):
        _retrain_status_panel(retraining_paused)
    st.divider()

    with st.expander("Cloud Publishing", expanded=True):
        _relay_status_panel()
    st.divider()

    if validation_mode:
        val_df = df.filter(pl.col("teacher_label").is_not_null()) if "teacher_label" in df.columns else None
        has_corrections = (
            val_df is not None
            and len(val_df) > 0
            and "is_correct" in df.columns
            and df["is_correct"].drop_nulls().__len__() > 0
        )
        if has_corrections:
            st.subheader("Validation")
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.plotly_chart(_confusion_matrix(val_df), use_container_width=True)
            with col_v2:
                st.plotly_chart(_teacher_student_comparison(val_df), use_container_width=True)
            st.plotly_chart(_rolling_accuracy(val_df, _VALIDATION_WINDOW), use_container_width=True)
            st.divider()
        else:
            st.info(
                "Validation data not yet available. "
                "Ensure `validation.enabled: true` in the config and teacher labels are present.",
                icon="⚠️",
            )
            st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(_particle_distribution(df), use_container_width=True)
    with col_b:
        st.plotly_chart(_confidence_distribution(df), use_container_width=True)

    st.plotly_chart(_tail_to_total_distribution(df), use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(_latency_histogram(df), use_container_width=True)
    with col_d:
        st.plotly_chart(_latency_over_time(df, latency_window), use_container_width=True)

    st.divider()
    if kaon_df is not None and len(kaon_df) > 0:
        _kaon_panel(kaon_df)
    else:
        st.info(
            "No kaon events captured yet. "
            "Kaons appear at ~1-2% beam fraction; the stream will populate automatically.",
            icon="🟣",
        )

    with st.expander("Latest events (most recent first)", expanded=True):
        st.dataframe(
            _latest_events_table(df, n=20),
            use_container_width=True,
            hide_index=True,
        )

    if auto_refresh:
        time.sleep(refresh_s)
        st.rerun()


if __name__ == "__main__":
    main()