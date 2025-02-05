import torch
from fastapi import FastAPI, UploadFile, File, Form
from datetime import datetime
import whisper
import os

app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173"],  # Add your SvelteKit dev server URL
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# Load Whisper model (You can specify a model like 'base', 'small', etc.)
model = whisper.load_model("base")
# Optionally, you can check if CUDA is available and move the model to GPU
# if torch.cuda.is_available():
#     model = model.to("cuda")
# else:
#     print("CUDA not available, using CPU.")

@app.post("/transcribe/{user_id}")
async def transcribe_audio(
        user_id: str,
        audio_file: UploadFile = File(...),
        patient_name: str = Form(...),
        session_date: datetime = Form(...)
):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_file_path = os.path.join(script_dir, "tmp", audio_file.filename)

    try:
        print(f"Received transcription request:")
        print(f"Patient Name: {patient_name}")
        print(f"Session Date: {session_date}")
        print(f"Audio File: {audio_file.filename}")

        os.makedirs(os.path.join(script_dir, "tmp"), exist_ok=True)

        with open(temp_file_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)

        print(f"File saved to {temp_file_path}")
        result = model.transcribe(temp_file_path)

        return {
            "user_id": user_id,
            "patient_name": patient_name,
            "session_date": session_date,
            "transcription": result['text']
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
