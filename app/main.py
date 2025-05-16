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
    # Validasi tipe file, hanya menerima file dengan tipe konten audio
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File harus berupa audio")

    try:
        # Simpan file audio yang diunggah ke direktori sementara dengan nama unik
        input_audio_path = os.path.join(TEMP_DIR, f"input_{uuid.uuid4().hex}.wav")
        with open(input_audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Buka kembali file audio untuk dibaca sebagai byte stream
        with open(input_audio_path, "rb") as f:
            audio_bytes = f.read()

        # Transkripsikan audio ke teks menggunakan whisper.cpp
        recognized_text = transcribe_speech_to_text(audio_bytes, file_ext=".wav")
        print("Recognized text:", recognized_text)

        # Kirim teks ke model Gemini untuk mendapatkan respons dalam bentuk teks
        response_text = generate_response(recognized_text)
        print("Response from Gemini:", response_text)

        # Ubah teks respons menjadi audio menggunakan TTS
        output_audio_path = transcribe_text_to_speech(response_text)

        # Kembalikan file audio hasil respons ke klien
        return FileResponse(output_audio_path, media_type="audio/wav", filename="response.wav")

    except Exception as e:
        # Tangani dan log error jika terjadi kegagalan dalam proses
        logging.exception("Error di /voice-chat:")
        raise HTTPException(status_code=500, detail=str(e))
