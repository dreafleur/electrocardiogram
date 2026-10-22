"""DSP core for CardioSignal: filter design, metrics, R-peak detection, rhythm analysis."""
import numpy as np
from scipy import signal

FILTERS = ["FIR Hann", "FIR Hamming", "IIR Butterworth"]


# ---------- Filters (all band-pass: remove baseline wander + high-freq/50 Hz noise) ----------
def design_fir(window, fs, numtaps, low, high):
    """Linear-phase FIR band-pass using the window method (window = 'hann' or 'hamming')."""
    return signal.firwin(int(numtaps) | 1, [low, high], pass_zero=False, window=window, fs=fs)


def design_iir(fs, order, low, high):
    """Butterworth band-pass as second-order sections (numerically stable)."""
    return signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")


def apply_fir(x, h):
    # forward-backward filtering (same as scipy.signal.filtfilt in the Colab notebook): zero phase, no R-peak shift
    return signal.filtfilt(h, [1.0], x, padlen=min(3 * len(h), len(x) - 1))


def apply_iir(x, sos):
    # forward-backward filtering gives zero phase distortion (no R-peak shift)
    return signal.sosfiltfilt(sos, x)


def run_filters(x, fs, low=0.5, high=40.0, numtaps=101, order=4, remove_dc=False):
    """Return ({name: filtered signal}, {name: filter coefficients}).
    Defaults match the Colab notebook: 0.5-40 Hz, 101 taps, order 4, no DC removal before filtering."""
    high = min(high, 0.45 * fs)  # keep cutoff safely below Nyquist
    x0 = x - np.mean(x) if remove_dc else np.asarray(x, dtype=float)
    h_hann = design_fir("hann", fs, numtaps, low, high)
    h_hamm = design_fir("hamming", fs, numtaps, low, high)
    sos = design_iir(fs, order, low, high)
    out = {"FIR Hann": apply_fir(x0, h_hann),
           "FIR Hamming": apply_fir(x0, h_hamm),
           "IIR Butterworth": apply_iir(x0, sos)}
    coeffs = {"FIR Hann": h_hann, "FIR Hamming": h_hamm, "IIR Butterworth": sos}
    return out, coeffs


def frequency_response(name, coeffs, fs):
    """Single-pass design magnitude response in dB (as plotted in the Colab notebook).
    Because the filters run forward-backward, the dB attenuation actually applied is doubled."""
    if name == "IIR Butterworth":
        f, h = signal.sosfreqz(coeffs, worN=4096, fs=fs)
    else:
        f, h = signal.freqz(coeffs, worN=4096, fs=fs)
    return f, 20 * np.log10(np.abs(h) + 1e-12)


def magnitude_spectrum(x, fs):
    """One-sided magnitude spectrum (same definition as the Colab notebook)."""
    x = np.asarray(x, dtype=float)
    spec = np.abs(np.fft.rfft(x)) / len(x)
    if len(spec) > 2:
        spec[1:-1] *= 2
    return np.fft.rfftfreq(len(x), d=1 / fs), spec


# ---------- Metrics (only valid with a clean reference) ----------
def snr_mse(ref, est, remove_dc=False):
    """Colab definitions: SNR (dB) = 10*log10(mean(ref^2) / mean((ref-est)^2)); MSE = mean((ref-est)^2).
    remove_dc=True subtracts each signal's mean first (ignores constant offset differences)."""
    ref = np.asarray(ref, dtype=float)
    est = np.asarray(est, dtype=float)
    if remove_dc:
        ref, est = ref - ref.mean(), est - est.mean()
    err = ref - est
    mse = float(np.mean(err ** 2))
    p_sig = float(np.mean(ref ** 2))
    snr = float("inf") if mse == 0 else float(10 * np.log10(p_sig / mse)) if p_sig > 0 else float("-inf")
    return {"snr": snr, "mse": mse}


def pick_best_filter(metrics):
    """metrics: {name: {'snr':..,'mse':..}}. Best = highest SNR (equivalently lowest MSE); MSE breaks ties."""
    return max(metrics, key=lambda k: (round(metrics[k]["snr"], 6), -metrics[k]["mse"]))


# ---------- R-peaks and rhythm ----------
def detect_r_peaks(y, fs, offset=0):
    """Simple threshold + refractory-period detector on the filtered ECG."""
    thr = 0.5 * np.percentile(y, 99.5)
    peaks, _ = signal.find_peaks(y, height=thr, distance=int(0.3 * fs))
    return peaks + offset


def rr_analysis(peaks, fs):
    """RR intervals, heart rate, variability and a neutral regularity label."""
    if len(peaks) < 3:
        return None
    rr = np.diff(peaks) / fs
    mean_rr = rr.mean()
    cv = rr.std() / mean_rr
    max_dev = np.max(np.abs(rr - np.median(rr))) / np.median(rr)
    irregular = cv > 0.10 or max_dev > 0.20
    return {
        "rr": rr, "bpm": 60 / mean_rr, "mean_rr": mean_rr,
        "sdnn_ms": rr.std() * 1000,
        "rmssd_ms": np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000 if len(rr) > 1 else 0.0,
        "cv": cv, "max_dev": max_dev, "irregular": bool(irregular),
        "rate_note": ("Below 60 BPM (slow)" if 60 / mean_rr < 60
                      else "Above 100 BPM (fast)" if 60 / mean_rr > 100 else "Within 60–100 BPM"),
        "label": ("Irregular rhythm — RR intervals vary abnormally (further evaluation recommended)"
                  if irregular else "Normal / regular rhythm — RR intervals are consistent"),
    }


def evaluate_detection(peaks, truth, fs, tol=0.05):
    """Sensitivity / precision / F1 against annotated R-peaks (tolerance 50 ms)."""
    tp, used = 0, set()
    for r in truth:
        if len(peaks) == 0:
            break
        j = int(np.argmin(np.abs(peaks - r)))
        if abs(peaks[j] - r) <= tol * fs and j not in used:
            used.add(j)
            tp += 1
    sens = tp / len(truth) if len(truth) else 0.0
    prec = tp / len(peaks) if len(peaks) else 0.0
    f1 = 2 * sens * prec / (sens + prec) if (sens + prec) else 0.0
    return {"sensitivity": sens, "precision": prec, "f1": f1, "tp": tp}


# ---------- Simulated noise (same model as the course notebook) ----------
def add_simulated_noise(clean, fs, bw_f=0.3, bw_a=0.20, pl_f=50.0, pl_a=0.05, noise_std=0.05, seed=42):
    """clean + baseline wander (sine) + power-line interference (sine) + Gaussian noise.
    RandomState(seed).randn matches np.random.seed(seed); np.random.randn in the notebook."""
    t = np.arange(len(clean)) / fs
    gauss = noise_std * np.random.RandomState(int(seed)).randn(len(clean))
    return clean + bw_a * np.sin(2 * np.pi * bw_f * t) + pl_a * np.sin(2 * np.pi * pl_f * t) + gauss
