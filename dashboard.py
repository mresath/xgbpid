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
import sys
import time
from pathlib import Path

import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

_LATENCY_BUDGET_US = 1_000.0   # µs — not user-configurable
_PARTICLE_COLORS   = {"electron": "#4C9BE8", "pion": "#E8704C"}


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
_TELEMETRY_PATH = Path(_cfg["logging"]["output_dir"]).parent / "live_telemetry.parquet"
_CONF_THRESHOLD = float(_cfg["model"]["confidence_threshold"])
_DEFAULT_REFRESH = int(_cfg["dashboard"]["refresh_seconds"])
_DEFAULT_WINDOW  = int(_cfg["dashboard"]["latency_window_events"])
_LABEL_MAP       = {int(k): v for k, v in _cfg["model"]["labels"].items()}
_VALIDATION_WINDOW = int(_cfg.get("validation", {}).get("window", 500))


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


def _metric_row(df: pl.DataFrame) -> None:
    total      = len(df)
    n_electron = (df["label"] == "electron").sum()
    n_pion     = (df["label"] == "pion").sum()
    e_pct      = 100.0 * n_electron / total if total else 0.0
    mean_lat   = df["latency_us"].mean() or 0.0
    p99_lat    = df["latency_us"].quantile(0.99) or 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total events",  f"{total:,}")
    c2.metric("Electrons",     f"{n_electron:,}",  f"{e_pct:.1f} %")
    c3.metric("Pions",         f"{n_pion:,}",      f"{100-e_pct:.1f} %")
    c4.metric("Mean latency",  f"{mean_lat:.1f} µs")

    p99_delta = f"budget {'OK' if p99_lat <= _LATENCY_BUDGET_US else 'EXCEEDED'}"
    c5.metric(
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


def main() -> None:
    st.set_page_config(
        page_title="XGBPID | Live",
        layout="wide",
    )

    with st.sidebar:
        st.title("XGBPID")
        st.divider()

        st.subheader("Telemetry source")
        telemetry_path = st.text_input(
            "Parquet file",
            value=str(_TELEMETRY_PATH),
            help="Path to live_telemetry.parquet written by main.py",
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

    st.title("XGBPID — Live Monitor")
    st.caption(f"Telemetry file: `{telemetry_path}`")

    df = _load(telemetry_path)

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