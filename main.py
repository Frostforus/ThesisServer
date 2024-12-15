import torch
from fastapi import FastAPI, UploadFile, File
import whisper
import os

app = FastAPI()

# Load Whisper model (You can specify a model like 'base', 'small', etc.)
model = whisper.load_model("base")
# Optionally, you can check if CUDA is available and move the model to GPU
# if torch.cuda.is_available():
#     model = model.to("cuda")
# else:
#     print("CUDA not available, using CPU.")

@app.post("/transcribe/{user_id}")
async def transcribe_audio(user_id: str, audio_file: UploadFile = File(...)):
    # Get the directory of the current script (main.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the temporary file path next to main.py
    temp_file_path = os.path.join(script_dir,"tmp", audio_file.filename)

    try:
        print(f"Transcribing audio file {audio_file.filename} for user {user_id}")
        # Save the uploaded audio file temporarily
        with open(temp_file_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        print(f"File saved to {temp_file_path}")
        # Transcribe the audio file using Whisper
        result = model.transcribe(temp_file_path)
    except Exception as e:
        return {"error": str(e)}
    finally:
        pass
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)  # Remove the temporary file after transcription

    return {
        "user_id": user_id,
        "transcription": result['text']
    }

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
