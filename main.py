from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import io
import subprocess
import os
import uuid
import librosa
import matplotlib.pyplot as plt
import scipy.signal
import joblib  # for loading the regression model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
model = joblib.load("tension_inverse_poly_model.pkl")

@app.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    thickness: float = Form(...),
    length: float = Form(...)
):
    input_path = None
    output_path = None

    try:
        contents = await file.read()

        input_path = f"temp_input_{uuid.uuid4().hex}.dat"
        output_path = f"converted_{uuid.uuid4().hex}.wav"
        with open(input_path, "wb") as f:
            f.write(contents)

        result = subprocess.run([
            "ffmpeg", "-y", "-i", input_path, "-ar", "44100", "-ac", "1", output_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise Exception(f"ffmpeg failed: {result.stderr.decode()}")

        audio_data, sr = librosa.load(output_path, sr=None)

        # Bandpass filter
        def bandpass_filter(signal, sr, lowcut=600, highcut=1700):
            nyquist = 0.5 * sr
            b, a = scipy.signal.butter(4, [lowcut / nyquist, highcut / nyquist], btype='band')
            return scipy.signal.filtfilt(b, a, signal)

        audio_data = bandpass_filter(audio_data, sr)

        # Frame & hop
        frame_length = 2048
        hop_length = 512

        # RMS energy
        rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
        peak_idx = np.argmax(rms)
        start_idx = max(0, peak_idx - 2)
        end_idx = min(len(rms), peak_idx + 3)

        # Frequency estimation
        f0 = librosa.yin(audio_data, fmin=600, fmax=1700, sr=sr,
                         frame_length=frame_length, hop_length=hop_length)
        f0 = np.nan_to_num(f0, nan=0.0)
        focused_f0 = f0[start_idx:end_idx]
        nonzero_freqs = focused_f0[focused_f0 > 0]

        if len(nonzero_freqs) == 0:
            frequency = 0
        else:
            frequency = np.max(nonzero_freqs)

        print("Estimated frequency:", frequency)

        # ---- Use regression model for tension ----
        if frequency > 0 and thickness > 0:
            inv_thickness = 1 / thickness
            freq_squared = frequency ** 2
            tension = model.predict([[inv_thickness, freq_squared]])[0]
        else:
            tension = 0

        return {
            "frequency": round(frequency, 2),
            "tension": round(tension, 2)
        }

    except Exception as e:
        print("Exception occurred:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        for path in [input_path, output_path]:
            if path and os.path.exists(path):
                os.remove(path)

if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.getenv("WEBSITES_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)