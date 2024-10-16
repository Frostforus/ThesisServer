from fastapi import FastAPI, UploadFile, File
import whisper
import os

app = FastAPI()

# Load Whisper model (You can specify a model like 'base', 'small', etc.)
model = whisper.load_model("base")


@app.post("/transcribe/{user_id}")
async def transcribe_audio(user_id: str, audio_file: UploadFile = File(...)):
    # Save the uploaded audio file temporarily
    temp_file_path = f"/tmp/{audio_file.filename}"

    with open(temp_file_path, "wb") as f:
        content = await audio_file.read()  # Read the file
        f.write(content)  # Write the file to disk

    # Transcribe the audio file using Whisper
    result = model.transcribe(temp_file_path)

    # Remove the temporary file after transcription
    os.remove(temp_file_path)

    # Return the transcription in JSON format
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
