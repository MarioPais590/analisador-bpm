from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import librosa
import soundfile as sf
import os
import uuid

app = FastAPI()

@app.post("/analisar")
async def analisar_e_alterar_bpm(
    file: UploadFile = File(...),
    novo_bpm: int = Form(...)
):
    temp_id = str(uuid.uuid4())
    original_filename = f"{temp_id}_{file.filename}"
    processed_filename = f"processed_{temp_id}.wav"

    with open(original_filename, "wb") as f:
        f.write(await file.read())

    try:
        y, sr = librosa.load(original_filename)
        bpm_original, _ = librosa.beat.beat_track(y=y, sr=sr)

        # Calcular fator de mudança de tempo
        fator = bpm_original / novo_bpm
        y_stretched = librosa.effects.time_stretch(y, rate=fator)

        # Salvar novo áudio
        sf.write(processed_filename, y_stretched, sr)

        # Apagar original
        os.remove(original_filename)

        return {
            "bpm_original": round(bpm_original),
            "bpm_novo": novo_bpm,
            "download_url": f"/baixar/{processed_filename}"
        }

    except Exception as e:
        if os.path.exists(original_filename):
            os.remove(original_filename)
        return {"erro": str(e)}

@app.get("/baixar/{filename}")
async def baixar(filename: str):
    return FileResponse(path=filename, filename=filename, media_type='audio/wav')
