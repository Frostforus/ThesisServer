import torch
from fastapi import FastAPI, UploadFile, File, Form
from datetime import datetime
import whisper
import os
from pocketbase import PocketBase
from dotenv import load_dotenv
import time
from pydub import AudioSegment

load_dotenv()

app = FastAPI()
print(f"PyTorch version: {torch.__version__}")
print(f"Whisper version: {whisper.__version__}")
# Authenticate with PocketBase
try:

    pb = PocketBase(os.getenv('POCKETBASE_URL'))
    pb.admins.auth_with_password(
        os.getenv('POCKETBASE_ADMIN_EMAIL'),
        os.getenv('POCKETBASE_ADMIN_PASSWORD')
    )
    print("Successfully authenticated with PocketBase as admin")
except Exception as e:
    print(f"Failed to authenticate with PocketBase: {str(e)}")
    raise e

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173"],  # Add your SvelteKit dev server URL
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# Load Whisper model (You can specify a model like 'base', 'small', etc.)
model = whisper.load_model("tiny")


# Optionally, you can check if CUDA is available and move the model to GPU
if torch.cuda.is_available():
    model = model.to("cuda")
    print("Model loaded successfully on GPU")
else:
    print("CUDA not available, using CPU.")

@app.post("/transcribe/{user_id}")
async def transcribe_audio(
        audio_file: UploadFile = File(...),
        patient_name: str = Form(...),
        session_date: datetime = Form(...)
):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_file_path = os.path.join(script_dir, "tmp", audio_file.filename)

    try:
        print(f"Transcribing audio file: {audio_file.filename}")
        # Save audio file temporarily
        os.makedirs(os.path.join(script_dir, "tmp"), exist_ok=True)
        content = await audio_file.read()
        with open(temp_file_path, 'wb') as f:
            f.write(content)

        # Transcribe audio
        result = model.transcribe(temp_file_path)
        transcription_text = result['text']
        print(f"Transcription: {transcription_text}")
        # Create session record
        session_data = {
            "date": session_date.isoformat(),
        }

        session_record = pb.collection('sessions').create(session_data)

        # Create transcription record with file
        file_data = None
        with open(temp_file_path, 'rb') as f:
            file_data = f.read()

        transcription_data = {
            "content": transcription_text,
            "model_used": "whisper",
            "generation_time_secs": result.get('generation_time', 0),
            "status": "transcribed",
            "session": session_record.id,
            "recording": (audio_file.filename, file_data, audio_file.content_type)
        }

        transcription_record = pb.collection('transcriptions').create(
            transcription_data
        )

        print(f"Created session record: {session_record.id}")
        print(f"Created transcription record: {transcription_record.id}")

        return {
            "success": True,
            "session_id": session_record.id,
            "transcription_id": transcription_record.id,
            "transcription": transcription_text
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/transcribe/benchmark/{user_id}")
async def transcribe_audio(
        audio_file: UploadFile = File(...),
        patient_name: str = Form(...),
        session_date: datetime = Form(...)
):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_file_path = os.path.join(script_dir, "tmp", audio_file.filename)

    try:


        # Save audio file temporarily
        os.makedirs(os.path.join(script_dir, "tmp"), exist_ok=True)
        content = await audio_file.read()
        with open(temp_file_path, 'wb') as f:
            f.write(content)
        print(f"Transcribing audio file: {audio_file.filename}")
        audio = AudioSegment.from_file(temp_file_path)
        audio_length = len(audio) / 1000.0  # Convert milliseconds to seconds
        print(f"Audio length: {audio_length:.2f} seconds")
        start_time = time.time()  # Start timing

        # Transcribe audio with English specification
        result = model.transcribe(
            temp_file_path,
            language="en",  # Force English language
            task="transcribe"  # Specifically set task as transcription
        )
        execution_time = time.time() - start_time  # Calculate total execution time

        transcription_text = result['text']

        print(f"Transcription completed in {execution_time:.2f} seconds")
        transcription_speed_ratio =  audio_length / execution_time
        print(f"Transcription speed ratio: {transcription_speed_ratio:.2f}x")
        print(f"Transcription: {transcription_text}")

        # Create session record
        session_data = {
            "date": session_date.isoformat(),
            "patient_name": patient_name
        }

        session_record = pb.collection('sessions').create(session_data)

        # Create transcription record with file
        file_data = None
        with open(temp_file_path, 'rb') as f:
            file_data = f.read()

        transcription_data = {
            "content": transcription_text,
            "model_used": "whisper",
            "generation_time_secs": execution_time,  # Use actual measured time
            "status": "transcribed",
            "session": session_record.id,
            "recording": (audio_file.filename, file_data, audio_file.content_type)
        }

        transcription_record = pb.collection('transcriptions').create(
            transcription_data
        )

        print(f"Created session record: {session_record.id}")
        print(f"Created transcription record: {transcription_record.id}")

        return {
            "success": True,
            "session_id": session_record.id,
            "transcription_id": transcription_record.id,
            "transcription": transcription_text,
            "execution_time": execution_time  # Include execution time in response
        }

    except Exception as e:
        print(f"Error: {str(e)}")
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
