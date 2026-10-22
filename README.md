# CardioSignal – ECG Denoising & Heartbeat Analysis

Streamlit app comparing FIR (Hann), FIR (Hamming) and IIR (Butterworth) band-pass ECG filters using SNR, MSE, waveform quality and R-peak detection, plus RR-interval rhythm analysis.

## Run locally

```bash
cd cardiosignal
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501. A built-in sample loads automatically, so no upload is needed for a demo.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI (dashboard, tabs, plots) |
| `dsp.py` | Filters, SNR/MSE, R-peak detection, RR analysis, detection scoring |
| `samples.py` | Synthetic ECG generator (clean reference + known R-peaks) |

## CSV upload format

- Optional `time` column (seconds) – used to infer sampling frequency (otherwise set it manually).
- ECG column named `ecg`, `noisy`, `signal`, or simply the first column.
- Optional `clean` column – enables SNR/MSE.

## Notes

- SNR/MSE appear only when a clean reference exists.
- Sensitivity/precision appear only when annotated R-peaks exist (built-in samples).
- Educational DSP project only. Not intended for medical diagnosis.
