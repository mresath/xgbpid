# xgbpid

**Real-time particle identification for the CERN T9 beamline.**

`xgbpid` classifies particles passing through a plastic scintillator (**electron**, **pion**, or **kaon**) in under 1 ms. It was built for the Beamline for Schools (BL4S) competition at CERN by Team QuaRC consisting of 9 students from Robert College, Türkiye.

Instead of a large detector array or doing offline analysis, a single scintillator connected to a cheap digitizer learns to classify particles in real time using an XGBoost gradient-boosted tree trained on pulse shapes.

**Live dashboard: [xgbpid.streamlit.app](https://xgbpid.streamlit.app/)**

---

## Quick Start

```bash
poetry run python3 runner.py
```

This launches three parallel processes:

1. **Main acquisition loop**: Reads waveforms from the Red Pitaya (or simulation), extracts features, and classifies each particle
2. **Background retrainer**: Periodically retrains the XGBoost model from accumulated labeled data
3. **Streamlit dashboard**: Live monitoring UI at `http://localhost:8501`

Stop everything with `Ctrl-C`.

To use a different config file or port:

```bash
poetry run python3 runner.py --config configs/experiment_v1.yaml --port 8501
```

---

## Physics Background

At 5–10 GeV/c, all three species are ultra-relativistic, making traditional Cherenkov-based separation by velocity difficult. This is where pulse-shape discrimination comes into play: the differences in how each particle deposits energy in the scintillator (rise time, tail length, decay shape) provide species information even when velocity-based methods break down. Electrons deposit energy electromagnetically (fast, narrow pulses). Pions and kaons deposit energy hadronically (slower, longer tails). Kaons are heavier than pions, produce a broader hadronic shower, and have a noticeably slower rise and longer tail. These waveform differences are what the XGBoost model learns to distinguish.

## AI Usage & Technical Disclaimer
This project incorporates artificial intelligence in the following capacities:

- **Core Logic:** The XGBoost particle identification model is trained to perform real-time inference on raw waveforms to differentiate between electrons, pions, and kaons.
- **Documentation & Refinement:** Generative AI tools were utilized moderately to assist with the implementation and its technical documentation.
- **Authenticity Note:** While AI assisted in code organization and presentation, the experimental design, physics logic, and the proposal were developed and written entirely by the students of team QuaRC.
- **Safety & Reliability:** This code is designed for research purposes within the T9 beamline environment and adheres to the "reliability and autonomy" pillar of the [CERN AI Strategy](https://home.cern/news/official-news/knowledge-sharing/general-principles-use-ai-cern).

---

## How It Works

### The "Teacher-Student" Architecture

The experiment uses two detector layers.

The **teacher** is a tiered array of existing detectors that provides ground-truth particle labels:

- **XCET-1** (low-pressure Cherenkov counter): Fires for electrons and pions
- **XCET-2** (high-pressure Cherenkov counter): Also fires for kaons
- **Lead-glass calorimeter**: Confirms electrons by detecting EM showers

The **student** is a Red Pitaya STEMlab 125-14 connected to an EJ-200 plastic scintillator. It digitizes waveforms at 125 MS/s with 14-bit resolution. It doesn't know the particle type, it just sees a 1024-sample voltage trace. The goal is to train a model that can predict the label from only the trace. Once trained, the student can run standalone without the teacher.

---

### Step 1 - Data Acquisition (`xgbpid/core/daq.py`)

When a particle triggers the detector, the Red Pitaya captures a 1024-sample waveform (~8 µs snapshot at 8 ns/sample). The DAQ module handles the SCPI/TCP communication with the hardware and returns an `AcquisitionBuffer` containing the raw ADC samples, a timestamp, and the teacher label (if available).

In simulation mode (`use_simulation: true`), a `MockDAQ` generates simulated pulses with realistic noise and timing jitter for each species. This allows for development and training without being at an actual accelerator.

---

### Step 2 - Feature Extraction (`xgbpid/core/processor.py`)

Raw waveforms aren't fed into the model directly. Instead, six features are computed from each pulse:

| Feature | What it captures |
|---|---|
| `rise_time_ns` | Time for the pulse to rise from 10% to 90% of peak - faster for electrons |
| `fwhm_ns` | Full width at half maximum - narrower pulses from EM showers |
| `tail_to_total` | Pulse-shape discrimination ratio (tail integral / total) - kaons have longer tails |
| `auc` | Total area under the pulse - proportional to deposited energy |
| `peak_amplitude` | Maximum voltage |
| `baseline_rms` | Pre-trigger noise - used to detect noisy conditions |

Pile-up detection runs first. If more than one distinct pulse is found above threshold, the event is discarded. The check requires at least 5 consecutive samples above threshold to count as a second pulse, preventing kaon long-tails from being falsely rejected.

---

### Step 3 - Particle Classification (`xgbpid/core/inference.py`)

The six features are handed to an XGBoost classifier. The model outputs a probability for each class and returns the one with the highest confidence.

A configurable confidence threshold (default: `0.70`) flags low-confidence predictions with a `[LOW CONF]` marker in the log. These events are still recorded but can be filtered.

The model supports **hot-reloading**: if `models/xgbpid.json` is updated on disk by the background retrainer, the classifier detects the changed modification time and reloads automatically. The main loop never needs to restart.

---

### Step 4 - Model Training (`scripts/train.py`)

The initial model is trained using `MockDAQ` with balanced class sizes (equal numbers of pions, electrons, and kaons). Balanced training ensures the model learns pulse-shape differences, and not beam fractions. At runtime, the real beam (~85% pions, ~8% electrons, ~1.5% kaons) is naturally reflected in the event stream.

The pulse shapes used for simulation:

| Species | Amplitude | Rise time | Decay time | Notes |
|---|---|---|---|---|
| Electron | 0.8 V | 1.5 ns | 20 ns | Fast, narrow - EM shower |
| Pion | 0.6 V | 3.0 ns | 45 ns | Minimum-ionizing particle |
| Kaon | 0.65 V | 5.0 ns | 80 ns | Slower rise, long hadronic tail |

During a live run, the **background retrainer** wakes up every 60 seconds (configurable), loads all accumulated Parquet run files, filters events with confirmed teacher labels, and retrains the XGBoost model. The new model is atomically written to `models/xgbpid.json`, where the classifier picks it up on the next hot-reload check.

---

### Step 5 - Telemetry & Dashboard (`dashboard.py`)

Every 50 events (configurable), the main loop overwrites `data/live_telemetry.parquet` using a write-to-temp-then-rename pattern. The rename is atomic on POSIX systems, so the dashboard always reads a complete file, no locking, is needed.

Kaon events get their own separate file (`data/kaon_telemetry.parquet`) and are flushed immediately on every occurrence, because they're rare (~1.5% of the beam) and there's no guarantee they'll appear in the next general flush window.

The Streamlit dashboard auto-refreshes every 2 seconds and shows:

- **Rolling particle composition**: Charts of predicted labels
- **Confidence distribution**: Per-class histogram with the threshold line
- **Latency plot**: Per-event classification time (target: < 1 ms)
- **Validation metrics**: Rolling accuracy, electron efficiency (ε_e), kaon efficiency (ε_K), and pion rejection factor (PRF) when validation mode is on
- **Retrainer status**: Last retrain time, training set size, model version
- **Kaon event log**: A dedicated section for rare kaon candidates

The relay (`xgbpid/core/relay.py`) mirrors the Parquet snapshot to a Supabase storage bucket in a background thread for the public dashboard at [xgbpid.streamlit.app](https://xgbpid.streamlit.app/).

---

### Validation Mode

When `validation.enabled: true` in the config, each prediction is compared against the teacher label. Rolling metrics are logged every flush cycle and shown on the dashboard:

- **ε_e (electron efficiency)**: Fraction of true electrons correctly identified
- **ε_K (kaon efficiency)**: Fraction of true kaons correctly identified
- **PRF (pion rejection factor)**: `1 / pion_misid_rate` - higher is better

A target electron efficiency (default: `0.90`) is configurable and shown as a reference line.

---

## Environment Variables

The Supabase relay reads credentials from a `.env` file in the project root.

```env
SUPABASE_URL=https://project.supabase.co
SUPABASE_KEY=anon-or-service-role-key
```

If either variable is missing, the relay disables itself and the rest of the system runs normally. The `.env` file is never committed to source control.

---

## Project Structure

```
xgbpid/
├── assets/                    # assets such as the team logo
├── configs/
│   └── experiment_v1.yaml     # all experiment parameters
├── dashboard.py               # local dashboard
├── public/  
│   └── dashboard.py           # public dashboard
├── main.py                    # acquisition loop
├── runner.py                  # process launcher (start here)
├── logs/                      # timestamped log files (gitignored)
├── models/                    # trained XGBoost models (gitignored)
│   └── xgbpid.json
├── data/                      # telemetry files (gitignored)
│   ├── runs/
│   ├── live_telemetry.parquet
│   ├── kaon_telemetry.parquet
│   └── retrain_status.json
├── scripts/
│   └── train.py               # model training & background retrainer
└── xgbpid/core/
    ├── daq.py                 # data acquisition
    ├── processor.py           # feature extraction
    ├── inference.py           # XGBoost classifier with hot-reload
    ├── relay.py               # Supabase uploader
    └── redpitaya_scpi.py      # SCPI protocol driver
```

---

## Configuration

Everything lives in `configs/experiment_v1.yaml`.

### `hardware`

Settings for the Red Pitaya digitizer. Only used when `simulation.use_simulation` is `false`.

| Key | Default | Description |
|---|---|---|
| `host` | `"192.168.1.100"` | IP address of the Red Pitaya |
| `port` | `5000` | SCPI TCP port |
| `channel` | `"CH1"` | Digitizer input channel wired to the scintillator |
| `decimation` | `1` | Sample rate divider - `1` = full 125 MHz (8 ns/sample) |
| `buffer_size` | `1024` | Samples captured per trigger event |
| `trigger_level` | `0.1` | Voltage threshold (V) for the TLU trigger |
| `trigger_delay` | `0` | Samples to store before the trigger point |
| `xcet1_source` | `2` | SCPI source index for XCET-1 (fires on pions and electrons) |
| `xcet2_source` | `3` | SCPI source index for XCET-2 (also fires on kaons) |
| `calorimeter_source` | `4` | SCPI source index for the lead-glass calorimeter |
| `calorimeter_threshold` | `0.30` | Minimum calorimeter voltage (V) to confirm an electron |

### `simulation`

Controls the `MockDAQ` used for offline development and initial training.

| Key | Default | Description |
|---|---|---|
| `use_simulation` | `true` | Set `false` to connect to real hardware |
| `sample_rate_mhz` | `125.0` | Simulated sample rate |
| `noise_sigma_frac` | `0.02` | Gaussian noise as a fraction of peak amplitude (2%) |
| `electron_fraction` | `0.08` | Fraction of simulated beam that is electrons (~8% at T9) |
| `kaon_fraction` | `0.015` | Fraction of simulated beam that is kaons (~1.5% at T9) |
| `onset_jitter_samples` | `5` | Uniform ±N sample timing jitter on the pulse onset |

### `pulse_shapes`

Defines the simulated waveform shape for each species in simulation. Each species has:

| Key | Description |
|---|---|
| `amplitude_v` | Mean peak voltage |
| `amplitude_spread_frac` | Spread as a fraction of amplitude (Gaussian for electrons, LogNormal for pions/kaons) |
| `rise_ns` | Pulse rise time in nanoseconds |
| `decay_ns` | Pulse decay time in nanoseconds |

Default settings mirror reality: electrons have fast, narrow pulses, pions are slower, and kaons are the slowest with the longest tail.

### `model`

| Key | Default | Description |
|---|---|---|
| `path` | `"models/xgbpid.json"` | Path to the XGBoost model file |
| `labels` | `{0: pion, 1: electron, 2: kaon}` | Integer-to-name mapping for classifier output |
| `confidence_threshold` | `0.70` | Predictions below this confidence are flagged `[LOW CONF]` |

### `processing`

Feature extraction and pile-up rejection settings, need to be tuned to the noise floor of the beamline.

| Key | Default | Description |
|---|---|---|
| `pileup_threshold_v` | `0.050` | Voltage above which a pulse is considered a real signal (V) |
| `pileup_min_width` | `5` | Minimum consecutive samples above threshold to count as a distinct second pulse |
| `rise_low_frac` | `0.10` | Lower fraction of peak amplitude used for rise-time measurement |
| `rise_high_frac` | `0.90` | Upper fraction of peak amplitude used for rise-time measurement |
| `baseline_window` | `50` | Number of pre-trigger samples used to estimate the baseline |

### `logging`

| Key | Default | Description |
|---|---|---|
| `level` | `"INFO"` | Python logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `output_dir` | `"data/runs"` | Directory where completed run Parquet files are written |
| `log_dir` | `"logs"` | Directory where timestamped `.log` files are written |

### `dashboard`

| Key | Default | Description |
|---|---|---|
| `refresh_seconds` | `2` | Default auto-refresh interval in the sidebar |
| `latency_window_events` | `500` | Number of recent events shown in the rolling latency plot |
| `flush_interval` | `50` | Write `live_telemetry.parquet` every N accepted events |
| `rolling_window` | `10000` | Maximum rows kept in the live telemetry file |

### `validation`

| Key | Default | Description |
|---|---|---|
| `enabled` | `true` | Compare predictions against teacher labels and compute rolling metrics |
| `window` | `500` | Rolling window size for accuracy and PRF calculations |
| `target_electron_efficiency` | `0.90` | ε_e target shown as a reference line on the dashboard |

### `retraining`

| Key | Default | Description |
|---|---|---|
| `interval_seconds` | `60` | How often the background retrainer wakes up |
| `n_events` | `30000` | Training set size per cycle when running in simulation mode |

### `relay`

| Key | Default | Description |
|---|---|---|
| `enabled` | `true` | Enable Supabase upload for the public dashboard |
| `upload_interval_seconds` | `60` | How often the relay thread uploads the Parquet snapshot |