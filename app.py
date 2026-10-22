"""CardioSignal – ECG Denoising & Rhythm Analysis (Streamlit). Run: streamlit run app.py"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dsp import (FILTERS, run_filters, frequency_response, magnitude_spectrum, snr_mse, pick_best_filter,
                 detect_r_peaks, rr_analysis, evaluate_detection, add_simulated_noise)
from samples import SAMPLES, make_sample

st.set_page_config(page_title="CardioSignal", page_icon="🫀", layout="wide")

TEAL, BLUE, RED, GREY = "#0f8b8d", "#2f5d9e", "#d62839", "#9aa5b1"
COLORS = {"FIR Hann": TEAL, "FIR Hamming": BLUE, "IIR Butterworth": "#7b5ea7"}
DESC = {
    "FIR Hann": "Linear-phase band-pass FIR (window method). Hann window: strong side-lobe suppression, wider transition band.",
    "FIR Hamming": "Linear-phase band-pass FIR (window method). Hamming window: narrower transition band, slightly higher side lobes.",
    "IIR Butterworth": "4th-order band-pass IIR with maximally flat passband. Very efficient; applied forward-backward for zero phase.",
}

st.markdown("""
<style>
.stApp {background:#f6f8fa;}
.block-container {padding-top:1.5rem; max-width:1250px;}
div[data-testid="stVerticalBlockBorderWrapper"] {background:#fff; border-radius:14px; box-shadow:0 2px 8px rgba(15,60,80,.06);}
.hdr {display:flex; align-items:center; gap:14px; margin-bottom:.6rem;}
.hdr h1 {margin:0; font-size:1.9rem; color:#0b3c49;}
.hdr p {margin:0; color:#5b6b78;}
.pill {display:inline-block; padding:.35rem .8rem; border-radius:999px; font-weight:600; font-size:.9rem;}
.ok {background:#e0f4f1; color:#0a6b6d;} .warn {background:#fdeaea; color:#b01e2c;}
.tbl {width:100%; border-collapse:collapse; background:#fff; border-radius:12px; overflow:hidden;}
.tbl th, .tbl td {padding:.5rem .9rem; border-bottom:1px solid #e6eaee; text-align:right;}
.tbl thead th {background:#eaf5f5; color:#0b3c49;} .tbl tbody th {text-align:left; color:#0b3c49;}
.foot {text-align:center; color:#7a8791; font-size:.85rem; padding:1.5rem 0 .5rem;}
</style>
<div class="hdr">
<svg width="46" height="46" viewBox="0 0 46 46"><rect width="46" height="46" rx="12" fill="#0f8b8d"/>
<polyline points="5,24 14,24 18,12 24,36 29,18 32,24 41,24" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>
<div><h1>CardioSignal</h1><p>ECG Denoising &amp; Rhythm Analysis</p></div></div>
""", unsafe_allow_html=True)


# ---------------- Input handling ----------------
def load_csv(file):
    """Parse an ECG CSV. Accepts a time column (optional), a noisy/ECG column, and an optional clean column."""
    df = pd.read_csv(file)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all").dropna()
    if df.shape[1] == 0 or len(df) < 100:
        raise ValueError("Need at least 100 numeric samples.")
    cols = {c: str(c).strip().lower() for c in df.columns}
    tcol = next((c for c, n in cols.items() if n in ("time", "t", "seconds", "time_s")), None)
    refcol = next((c for c, n in cols.items() if n in ("clean", "reference", "ref", "ground_truth")), None)
    sigcols = [c for c in df.columns if c not in (tcol, refcol)]
    if not sigcols:
        raise ValueError("No ECG column found.")
    sigcol = next((c for c in sigcols if cols[c] in ("noisy", "ecg", "signal", "raw")), sigcols[0])
    fs = 360.0
    if tcol is not None:
        dt = np.median(np.diff(df[tcol].to_numpy()))
        if dt > 0:
            fs = round(1 / dt, 3)
    return dict(name="Uploaded file", x=df[sigcol].to_numpy(float),
                ref=df[refcol].to_numpy(float) if refcol is not None else None,
                fs=fs, r_true=None, source="upload")


def use_sample(name):
    st.session_state.sig = make_sample(name)
    st.session_state.view = FILTERS[0]


if "sig" not in st.session_state:
    use_sample("Normal rhythm (~72 bpm)")
st.session_state.setdefault("view", FILTERS[0])

with st.container(border=True):
    st.markdown("##### ECG input")
    c1, c2, c3 = st.columns([1.3, 1.3, 1])
    up = c1.file_uploader("Upload clean ECG reference (CSV)", type=["csv"],
                          help="Columns: optional 'time' (s), the ECG signal, optional 'clean'. A single-ECG file is treated as the "
                               "clean reference and noise is simulated on it (as in the Colab notebook).")
    choice = c2.selectbox("Built-in sample ECG", list(SAMPLES))
    c2.button("Load selected sample", on_click=use_sample, args=(choice,), use_container_width=True)
    c3.write("")
    c3.button("Load example signal", on_click=use_sample, args=("Normal rhythm (~72 bpm)",), use_container_width=True, type="primary")

    if up is not None and st.session_state.get("last_upload") != up.file_id:
        try:
            st.session_state.sig = load_csv(up)
            st.session_state.view = FILTERS[0]
        except Exception as e:  # invalid uploads
            st.error(f"Could not read this file: {e}")
        st.session_state.last_upload = up.file_id

    sig = st.session_state.sig
    if sig["source"] == "upload":
        sig["fs"] = c3.number_input("Sampling frequency (Hz)", 50.0, 5000.0, float(sig["fs"]), step=10.0)

    # Optional: treat an uploaded (clean) ECG as the reference and add simulated noise
    sim_on = False
    if sig["source"] == "upload" and sig["ref"] is None:
        with st.expander("Simulated noise (uploaded ECG is used as the clean reference)", expanded=True):
            sim_on = st.checkbox("Add simulated noise: baseline wander + power-line + Gaussian", value=True)
            n1, n2, n3 = st.columns(3)
            bw_f = n1.number_input("Baseline wander freq (Hz)", 0.05, 2.0, 0.3, 0.05)
            bw_a = n1.number_input("Baseline wander amplitude (mV)", 0.0, 2.0, 0.20, 0.05)
            pl_f = n2.number_input("Power-line freq (Hz)", 40.0, 70.0, 50.0, 10.0)
            pl_a = n2.number_input("Power-line amplitude (mV)", 0.0, 1.0, 0.05, 0.01)
            noise_std = n3.number_input("Gaussian noise std (mV)", 0.0, 1.0, 0.05, 0.01)
            seed = n3.number_input("Random seed", 0, 99999, 42, 1)

x, ref, fs = sig["x"], sig["ref"], float(sig["fs"])
if sim_on:
    if pl_f >= fs / 2:
        st.warning("Power-line frequency is above Nyquist for this sampling rate; skipping it.")
        pl_a = 0.0
    ref = x.copy()  # the original ECG becomes the clean reference
    x = add_simulated_noise(ref, fs, bw_f, bw_a, pl_f, pl_a, noise_std, seed)
if ref is not None and len(ref) != len(x):
    ref = None
    st.warning("Reference length differs from ECG length — ignoring reference.")
n, dur = len(x), len(x) / fs
t = np.arange(n) / fs
m1, m2, m3, m4 = st.columns(4)
m1.metric("Signal", "Uploaded + noise" if sim_on else sig["name"].split(" (")[0])
m2.metric("Sampling frequency", f"{fs:g} Hz")
m3.metric("Duration", f"{dur:.1f} s")
m4.metric("Clean reference", "Available" if ref is not None else "Not available")

# ---------------- Filtering & analysis ----------------
with st.sidebar:
    st.header("Filter settings")
    low = st.number_input("Low cutoff (Hz)", 0.1, 5.0, 0.5, 0.1)
    high = st.number_input("High cutoff (Hz)", 10.0, 100.0, 40.0, 5.0)
    taps = st.slider("FIR length (taps)", 101, 2001, 101, 100)
    order = st.slider("Butterworth order", 1, 8, 4)
    st.header("Evaluation")
    dc_fair = st.checkbox("Ignore DC offset when scoring", value=False,
                          help="Off = exactly like the notebook (reference compared as-is). On = subtract the mean of reference "
                               "and output before computing SNR/MSE, so a constant offset is not counted as error.")
    skip_edges = st.checkbox("Exclude filter edge samples when scoring", value=False,
                             help="Off = score the whole signal like the notebook.")

if high >= fs / 2:
    st.warning(f"High cutoff is above Nyquist ({fs/2:g} Hz); it will be reduced automatically.")
try:
    filtered, coeffs = run_filters(x, fs, low, high, taps, order)
except Exception as e:
    st.error(f"Filtering failed: {e}")
    st.stop()

edge = min(taps // 2, n // 4) if skip_edges else 0  # optionally ignore filter edge transients when scoring
sl = slice(edge, n - edge)
true_r = sig["r_true"][(sig["r_true"] >= edge) & (sig["r_true"] < n - edge)] if sig["r_true"] is not None else None

results = {}
for name, y in filtered.items():
    peaks = detect_r_peaks(y[sl], fs, offset=edge)
    results[name] = dict(
        y=y, peaks=peaks, rr=rr_analysis(peaks, fs),
        metrics=snr_mse(ref[sl], y[sl], dc_fair) if ref is not None else None,
        det=evaluate_detection(peaks, true_r, fs) if true_r is not None else None)

noisy_m = snr_mse(ref[sl], x[sl], dc_fair) if ref is not None else None
best = pick_best_filter({f: results[f]["metrics"] for f in FILTERS}) if ref is not None else None
ref_plot = (ref - ref.mean() if dc_fair else ref) if ref is not None else None


# ---------------- Plot helpers ----------------
def style(fig, h=380, ytitle="Amplitude (mV, arb.)"):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), template="plotly_white",
                      legend=dict(orientation="h", y=1.12), xaxis_title="Time (s)", yaxis_title=ytitle,
                      plot_bgcolor="#fff", hovermode="x unified")
    fig.update_xaxes(showgrid=True, gridcolor="#eef1f4")
    fig.update_yaxes(showgrid=True, gridcolor="#eef1f4")
    return fig


def window_range():
    return st.slider("Visible window (s) — drag on the plot to zoom, double-click to reset",
                     0.0, float(dur), (0.0, float(min(dur, 8.0))), step=0.5)


def set_view(name):
    st.session_state.view = name


def show_table(data):
    """Render {row: {column: value}} as an HTML table (no pandas/pyarrow, which some Windows policies block)."""
    cols = list(next(iter(data.values())))
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = "".join(
        f"<tr><th>{r}</th>" + "".join(f"<td>{'—' if v is None else v}</td>" for v in vals.values()) + "</tr>"
        for r, vals in data.items())
    st.markdown(f'<table class="tbl"><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table>',
                unsafe_allow_html=True)


tab_dash, tab_cmp, tab_hb, tab_about = st.tabs(["Dashboard", "Filter Comparison", "Heartbeat Analysis", "About"])

# ================= Dashboard =================
with tab_dash:
    view = st.session_state.view
    if best is not None:
        bm = results[best]["metrics"]
        with st.container(border=True):
            st.markdown("##### Recommended filter")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Best filter", best)
            r2.metric("SNR", f"{bm['snr']:.2f} dB", f"{bm['snr'] - noisy_m['snr']:+.2f} dB vs noisy input")
            r3.metric("MSE", f"{bm['mse']:.5f}", f"{bm['mse'] - noisy_m['mse']:+.5f} vs noisy input", delta_color="inverse")
            r4.metric("Noisy input SNR", f"{noisy_m['snr']:.2f} dB")
            if bm["snr"] < noisy_m["snr"]:
                st.caption("With these settings no filter beats the noisy input's SNR. The band-pass also removes the reference's own "
                           "DC offset and drift below the low cutoff, which counts as error — try 'Ignore DC offset when scoring' in the sidebar.")
            else:
                st.caption(f"{best} gives the highest SNR / lowest MSE against the clean reference for this signal and these settings.")
    with st.container(border=True):
        st.markdown(f"##### ECG waveform — showing **{view}**")
        win = window_range()
        fig = go.Figure()
        fig.add_scatter(x=t, y=x, name="Original noisy ECG", line=dict(color="#e58a8a", width=1))
        fig.add_scatter(x=t, y=results[view]["y"], name=f"Filtered ({view})", line=dict(color=TEAL, width=1.8))
        if ref is not None:
            fig.add_scatter(x=t, y=ref_plot, name="Clean reference", line=dict(color=BLUE, width=1.2, dash="dot"))
        fig.update_xaxes(range=list(win))
        st.plotly_chart(style(fig), use_container_width=True)

    st.markdown("##### Filter comparison")
    for col, name in zip(st.columns(3), FILTERS):
        with col, st.container(border=True):
            st.markdown(f"**{name}**")
            st.caption(DESC[name])
            m = results[name]["metrics"]
            a, b = st.columns(2)
            a.metric("SNR", f"{m['snr']:.2f} dB" if m else "—", f"{m['snr'] - noisy_m['snr']:+.2f} dB" if m else None)
            b.metric("MSE", f"{m['mse']:.5f}" if m else "—")
            if not m:
                st.caption("SNR/MSE need a clean reference signal.")
            st.button("View filtered waveform", key=f"v_{name}", on_click=set_view, args=(name,),
                      use_container_width=True, type="primary" if name == view else "secondary")

    with st.container(border=True):
        st.markdown("##### Performance comparison")
        if ref is None:
            st.info("SNR and MSE are only calculated when a clean reference ECG is available. "
                    "Load a built-in sample or upload a CSV with a `clean` column.")
        else:
            fig = make_subplots(rows=1, cols=2, subplot_titles=("SNR (dB) — higher is better", "MSE — lower is better"))
            vals = [[results[f]["metrics"][k] for f in FILTERS] for k in ("snr", "mse")]
            for i, v in enumerate(vals, 1):
                fig.add_bar(x=FILTERS, y=v, marker_color=[COLORS[f] for f in FILTERS], row=1, col=i,
                            text=[f"{q:.3g}" for q in v], textposition="outside", showlegend=False)
            st.plotly_chart(fig.update_layout(height=340, template="plotly_white", margin=dict(t=40, b=10)),
                            use_container_width=True)
            st.caption("Higher SNR and lower MSE are generally preferable when all filters are evaluated "
                       "against the same valid reference signal. The small delta under each SNR is the change vs the noisy input.")

    with st.container(border=True):
        st.markdown("##### Results summary")
        rows = []
        if ref is not None:
            rows.append(("Highest SNR", max(FILTERS, key=lambda f: results[f]["metrics"]["snr"])))
            rows.append(("Lowest MSE", min(FILTERS, key=lambda f: results[f]["metrics"]["mse"])))
        if true_r is not None:
            rows.append(("Best R-peak detection (F1)", max(FILTERS, key=lambda f: results[f]["det"]["f1"])))
        if rows:
            for col, (label, winner) in zip(st.columns(len(rows)), rows):
                col.metric(label, winner)
        else:
            st.info("No reference or annotations available — only heart-rate estimates can be compared (see Heartbeat Analysis).")
        st.caption("The best filter can differ depending on the noise type, signal characteristics and filter settings; "
                   "these results apply to this signal only.")

    # Rhythm verdict, computed on the best filter's output (or the selected view if there is no reference)
    rf = best if best is not None else view
    ra = results[rf]["rr"]
    with st.container(border=True):
        st.markdown(f"##### Heartbeat check — based on **{rf}**")
        if ra is None:
            st.warning("Too few R-peaks detected for rhythm analysis (need at least 3).")
        else:
            h1, h2, h3 = st.columns(3)
            h1.metric("Heart rate", f"{ra['bpm']:.0f} BPM", ra["rate_note"], delta_color="off")
            h2.metric("RR variability (CV)", f"{ra['cv']*100:.1f} %")
            h3.metric("Largest RR deviation", f"{ra['max_dev']*100:.1f} %")
            st.markdown(f'<span class="pill {"warn" if ra["irregular"] else "ok"}">{ra["label"]}</span>', unsafe_allow_html=True)
            st.caption("Screening rule on RR intervals only (CV > 10% or any RR >20% off the median). It is not a diagnosis; "
                       "see the Heartbeat Analysis tab for the R-peak plot.")

# ================= Filter comparison =================
with tab_cmp:
    with st.container(border=True):
        st.markdown("##### All filters overlaid")
        win2 = st.slider("Visible window (s)", 0.0, float(dur), (0.0, float(min(dur, 6.0))), step=0.5, key="w2")
        fig = go.Figure()
        fig.add_scatter(x=t, y=x, name="Noisy", line=dict(color="#e9b3b3", width=1))
        if ref is not None:
            fig.add_scatter(x=t, y=ref_plot, name="Clean reference", line=dict(color="#222", width=1.2, dash="dot"))
        for f in FILTERS:
            fig.add_scatter(x=t, y=results[f]["y"], name=f, line=dict(color=COLORS[f], width=1.4))
        fig.update_xaxes(range=list(win2))
        st.plotly_chart(style(fig, 420), use_container_width=True)
    with st.container(border=True):
        st.markdown("##### Frequency responses")
        fig = go.Figure()
        for f in FILTERS:
            fr, mag = frequency_response(f, coeffs[f], fs)
            fig.add_scatter(x=fr, y=mag, name=f, line=dict(color=COLORS[f]))
        fig.update_xaxes(range=[0, min(100, fs / 2)])
        fig.update_yaxes(range=[-100, 5])
        st.plotly_chart(style(fig, 340, "Magnitude (dB)").update_layout(xaxis_title="Frequency (Hz)"), use_container_width=True)
        st.caption("Single-pass design responses (as in the notebook). Filters run forward-backward, so the dB attenuation actually "
                   "applied is doubled. Note the 50 Hz power-line region and the low cutoff that removes baseline wander.")
    with st.container(border=True):
        st.markdown("##### Frequency spectra (FFT magnitude)")
        fig = go.Figure()
        spectra = {"Noisy ECG": (x, "#e58a8a")}
        if ref is not None:
            spectra = {"Reference ECG": (ref, "#222"), **spectra}
        spectra.update({f: (results[f]["y"], COLORS[f]) for f in FILTERS})
        for label, (sig_, col) in spectra.items():
            fr, mag = magnitude_spectrum(sig_, fs)
            fig.add_scatter(x=fr, y=mag, name=label, line=dict(color=col, width=1.3), opacity=0.85)
        fig.update_xaxes(range=[0, min(100, fs / 2)])
        st.plotly_chart(style(fig, 340, "Magnitude").update_layout(xaxis_title="Frequency (Hz)"), use_container_width=True)
        st.caption("The noisy spectrum shows the 50 Hz power-line spike and the low-frequency baseline-wander component; "
                   "the filtered spectra show how each filter removes them.")
    if ref is not None:
        tbl = {"Noisy ECG (input)": {"SNR (dB)": round(noisy_m["snr"], 2), "SNR improvement (dB)": 0.0,
                                     "MSE": round(noisy_m["mse"], 6)}}
        tbl.update({f: {"SNR (dB)": round(results[f]["metrics"]["snr"], 2),
                        "SNR improvement (dB)": round(results[f]["metrics"]["snr"] - noisy_m["snr"], 2),
                        "MSE": round(results[f]["metrics"]["mse"], 6)} for f in FILTERS})
        show_table(tbl)

# ================= Heartbeat analysis =================
with tab_hb:
    sel = st.selectbox("Filtered ECG used for R-peak detection", FILTERS, key="view")
    r = results[sel]
    with st.container(border=True):
        st.markdown("##### Detected R-peaks")
        win3 = st.slider("Visible window (s)", 0.0, float(dur), (0.0, float(min(dur, 8.0))), step=0.5, key="w3")
        fig = go.Figure()
        fig.add_scatter(x=t, y=r["y"], name=f"Filtered ({sel})", line=dict(color=TEAL, width=1.5))
        fig.add_scatter(x=r["peaks"] / fs, y=r["y"][r["peaks"]], mode="markers", name="Detected R-peaks",
                        marker=dict(color=RED, size=9, symbol="diamond"))
        fig.update_xaxes(range=list(win3))
        st.plotly_chart(style(fig), use_container_width=True)

    a = r["rr"]
    if a is None:
        st.warning("Too few R-peaks detected for rhythm analysis (need at least 3).")
    else:
        with st.container(border=True):
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Heart rate", f"{a['bpm']:.0f} BPM")
            k2.metric("Mean RR", f"{a['mean_rr']*1000:.0f} ms")
            k3.metric("RR variability (SDNN)", f"{a['sdnn_ms']:.0f} ms")
            k4.metric("RMSSD", f"{a['rmssd_ms']:.0f} ms")
            cls = "warn" if a["irregular"] else "ok"
            st.markdown(f'<span class="pill {cls}">{a["label"]}</span>', unsafe_allow_html=True)
            st.caption(f"Rule used: flagged if RR coefficient of variation > 10% (here {a['cv']*100:.1f}%) or any RR deviates "
                       f">20% from the median (here {a['max_dev']*100:.1f}%). Irregular RR intervals are not a confirmed arrhythmia.")
        with st.container(border=True):
            st.markdown("##### RR intervals")
            fig = go.Figure(go.Bar(x=np.arange(1, len(a["rr"]) + 1), y=a["rr"] * 1000, marker_color=TEAL))
            fig.add_hline(y=a["mean_rr"] * 1000, line_dash="dot", line_color=RED, annotation_text="mean")
            st.plotly_chart(style(fig, 280, "RR interval (ms)").update_layout(xaxis_title="Beat #"), use_container_width=True)

    rows = {}
    for f in FILTERS:
        q, d = results[f]["rr"], results[f]["det"]
        rows[f] = {"R-peaks": len(results[f]["peaks"]),
                   "BPM": round(q["bpm"], 1) if q else None,
                   "SDNN (ms)": round(q["sdnn_ms"], 1) if q else None,
                   "Sensitivity": round(d["sensitivity"], 3) if d else None,
                   "Precision": round(d["precision"], 3) if d else None}
    st.markdown("##### Per-filter detection summary")
    show_table(rows)
    if true_r is None:
        st.caption("Sensitivity/precision require annotated R-peaks; they are available for the built-in samples only.")

# ================= About =================
with tab_about:
    st.markdown("""
**CardioSignal** compares three band-pass ECG denoising filters (default 0.5–40 Hz) for a DSP course project.

- **Workflow** – upload a clean ECG reference → simulated noise is added (0.3 Hz baseline wander, 50 Hz power-line, Gaussian; same model and defaults as the Colab notebook) → three filters → SNR/MSE → recommended filter → heartbeat check.
- **FIR Hann / Hamming** – `scipy.signal.firwin` (window method, 101 taps by default) applied with `filtfilt` (zero phase).
- **IIR Butterworth** – `scipy.signal.butter` (order 4, SOS form) applied with `sosfiltfilt` (zero phase).
- **SNR** = 10·log10(mean(ref²) / mean((ref − filtered)²)), **MSE** = mean((ref − filtered)²) — computed only with a clean reference. **Best filter** = highest SNR (lowest MSE).
- **R-peaks** – threshold + 300 ms refractory period on the filtered signal; scored against annotations (±50 ms) when available.
- **Rhythm** – RR intervals, BPM, SDNN, RMSSD and a simple regularity rule.

**CSV format:** optional `time` column (seconds), an ECG column (`ecg`, `noisy`, `signal` or first column), optional `clean` column.

Built-in samples are synthetic (sum of Gaussian P-Q-R-S-T waves plus baseline wander, 50 Hz interference and white noise),
so a ground-truth reference and R-peak annotations exist. Real-data results will differ.
""")

st.markdown('<div class="foot">Educational DSP project only. This application is not intended for medical diagnosis.</div>',
            unsafe_allow_html=True)
