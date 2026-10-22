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
    # mode='same' removes the constant group delay of a linear-phase FIR
    return signal.fftconvolve(x, h, mode="same")


def apply_iir(x, sos):
    # forward-backward filtering gives zero phase distortion (no R-peak shift)
    return signal.sosfiltfilt(sos, x)


def run_filters(x, fs, low=0.5, high=40.0, numtaps=1001, order=4):
    """Return ({name: filtered signal}, {name: filter coefficients})."""
    high = min(high, 0.45 * fs)  # keep cutoff safely below Nyquist
    x0 = x - np.mean(x)
    h_hann = design_fir("hann", fs, numtaps, low, high)
    h_hamm = design_fir("hamming", fs, numtaps, low, high)
    sos = design_iir(fs, order, low, high)
    out = {"FIR Hann": apply_fir(x0, h_hann),
           "FIR Hamming": apply_fir(x0, h_hamm),
           "IIR Butterworth": apply_iir(x0, sos)}
    coeffs = {"FIR Hann": h_hann, "FIR Hamming": h_hamm, "IIR Butterworth": sos}
    return out, coeffs


def frequency_response(name, coeffs, fs):
    """Effective magnitude response in dB (IIR squared because of forward-backward filtering)."""
    if name == "IIR Butterworth":
        f, h = signal.sosfreqz(coeffs, worN=4096, fs=fs)
        return f, 40 * np.log10(np.abs(h) + 1e-12)
    f, h = signal.freqz(coeffs, worN=4096, fs=fs)
    return f, 20 * np.log10(np.abs(h) + 1e-12)


# ---------- Metrics (only valid with a clean reference) ----------
def snr_mse(ref, est):
    """SNR (dB) = 10*log10(sum(ref^2) / sum((ref-est)^2)); MSE = mean((ref-est)^2)."""
    ref = ref - np.mean(ref)
    err = ref - est
    mse = float(np.mean(err ** 2))
    snr = float(10 * np.log10(np.sum(ref ** 2) / np.sum(err ** 2)))
    return {"snr": snr, "mse": mse}


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
        "label": ("Irregularity detected — further evaluation recommended"
                  if irregular else "Relatively regular pattern"),
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
