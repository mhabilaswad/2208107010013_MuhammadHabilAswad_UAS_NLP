from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import shutil
import os
import uuid
import logging

# Import fungsi dari modul lain
from stt import transcribe_speech_to_text
from llm import generate_response
from tts import transcribe_text_to_speech  # tidak perlu impor COQUI_MODEL_PATH, dst

app = FastAPI()

# Folder sementara untuk simpan audio
TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/voice-chat")
async def voice_chat(file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File harus berupa audio")

    try:
        input_audio_path = os.path.join(TEMP_DIR, f"input_{uuid.uuid4().hex}.wav")
        with open(input_audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        with open(input_audio_path, "rb") as f:
            audio_bytes = f.read()

        recognized_text = transcribe_speech_to_text(audio_bytes, file_ext=".wav")
        print("Recognized text:", recognized_text)
        response_text = generate_response(recognized_text)
        print("Response from Gemini:", response_text)
        output_audio_path = transcribe_text_to_speech(response_text)

        return FileResponse(output_audio_path, media_type="audio/wav", filename="response.wav")

    except Exception as e:
        logging.exception("Error di /voice-chat:")  # Ini akan print stacktrace ke console
        raise HTTPException(status_code=500, detail=str(e))
