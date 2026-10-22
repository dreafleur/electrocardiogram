"""Built-in synthetic ECG samples (clean reference + known R-peak times)."""
import numpy as np

FS = 360
SAMPLES = {
    "Normal rhythm (~72 bpm)": dict(hr=72, irregular=False, noise=1.0),
    "Irregular RR intervals": dict(hr=75, irregular=True, noise=1.0),
    "High-noise ECG": dict(hr=70, irregular=False, noise=2.5),
}

# (offset from R peak in s, amplitude, width in s) for P, Q, R, S, T waves
WAVES = [(-0.20, 0.15, 0.025), (-0.04, -0.12, 0.010), (0.0, 1.0, 0.012),
         (0.04, -0.25, 0.010), (0.28, 0.30, 0.045)]


def make_sample(name, duration=10.0, seed=7):
    cfg = SAMPLES[name]
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * FS)) / FS

    # beat times
    beats, tb = [], 0.6
    while tb < duration - 0.5:
        beats.append(tb)
        base = 60 / cfg["hr"]
        tb += rng.uniform(0.6, 1.25) * base if cfg["irregular"] else base * (1 + 0.03 * rng.standard_normal())

    clean = np.zeros_like(t)
    for b in beats:
        for off, amp, w in WAVES:
            clean += amp * np.exp(-0.5 * ((t - b - off) / w) ** 2)
    clean -= clean.mean()

    k = cfg["noise"]
    baseline = 0.4 * np.sin(2 * np.pi * 0.25 * t + 1.0) + 0.2 * np.sin(2 * np.pi * 0.12 * t)
    powerline = 0.15 * k * np.sin(2 * np.pi * 50 * t)
    noisy = clean + baseline + powerline + 0.04 * k * rng.standard_normal(t.size)

    return dict(name=name, x=noisy, ref=clean, fs=float(FS),
                r_true=np.round(np.array(beats) * FS).astype(int), source="sample")
