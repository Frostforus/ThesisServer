import torch
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form, HTTPException
from datetime import datetime
import whisper
import os
from pocketbase import PocketBase
from dotenv import load_dotenv
import time
from pydub import AudioSegment
import requests
import logging
import colorlog
from pathlib import Path
import magic
from typing import Optional
# Ignore Magic SyntaxWarning
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)


# Configure logging with colors
def setup_logger():
    """Set up the logger with color formatting"""
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(levelname)s:\t  %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    ))

    logger = colorlog.getLogger('app')
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


logger = setup_logger()

load_dotenv()

app = FastAPI()
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"Whisper version: {whisper.__version__}")

# Authenticate with PocketBase
try:
    pb = PocketBase(os.getenv('POCKETBASE_URL'))
    pb.admins.auth_with_password(
        os.getenv('POCKETBASE_ADMIN_EMAIL'),
        os.getenv('POCKETBASE_ADMIN_PASSWORD')
    )
    logger.info("Successfully authenticated with PocketBase as admin")
except Exception as e:
    logger.error(f"Failed to authenticate with PocketBase: {str(e)}")
    raise e

# Global variables
MODEL_VERSION = "tiny"
model: Optional[whisper.Whisper] = None


@app.on_event("startup")
async def startup_event():
    load_model(MODEL_VERSION)


def process_audio_transcription(
        file_path: str,
        filename: str,
        content_type: str,
        session_id: str
):
    try:
        logger.info(f"Starting transcription for: {filename}")
        audio = AudioSegment.from_file(file_path)
        audio_length = len(audio) / 1000.0  # Convert milliseconds to seconds
        logger.info(f"Audio length: {audio_length:.2f} seconds")
        start_time = time.time()

        result = model.transcribe(
            file_path,
            language="en",
            task="transcribe"
        )
        execution_time = time.time() - start_time

        transcription_text = result['text']

        logger.info(f"Transcription completed in {execution_time:.2f} seconds")
        transcription_speed_ratio = audio_length / execution_time
        logger.debug(f"Transcription speed ratio: {transcription_speed_ratio:.2f}x")
        logger.debug(f"Transcription: {transcription_text}")

        url = "http://localhost:8090/api/collections/transcriptions/records"

        with open(file_path, 'rb') as f:
            mime_type = magic.from_buffer(f.read(2048), mime=True)
            f.seek(0)  # Reset file pointer to beginning
            files = {
                'recording': (Path(file_path).name, f, mime_type)
            }

            data = {
                "content": transcription_text,
                "model_used": f"whisper - {MODEL_VERSION}",
                "generation_time_secs": str(execution_time),
                "transcription_speed_ratio": str(transcription_speed_ratio),
                "status": "transcribed",
                "session": session_id
            }

            headers = {
                'Authorization': pb.auth_store.token
            }

            response = requests.post(
                url,
                files=files,
                data=data,
                headers=headers
            )
            response.raise_for_status()

            record = response.json()

            logger.info(f"Created transcription record: {record['id']}")

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error occurred: {str(e)}")
        logger.error(f"Response content: {e.response.text}")

    except Exception as e:
        logger.error(f"Error processing transcription: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


def load_model(version: str):
    global model, MODEL_VERSION
    MODEL_VERSION = version
    model = whisper.load_model(MODEL_VERSION)

    if torch.cuda.is_available():
        model = model.to("cuda")
        logger.info(f"Model {version} loaded successfully on GPU")
    else:
        logger.warning(f"CUDA not available, using CPU for model {version}")

    return {"status": "success", "model": version}


# FastAPI endpoint
@app.post("/api/model")
async def change_model(model_version: str):
    try:
        valid_models = ["tiny", "base", "small", "medium", "large", "turbo"]
        if model_version not in valid_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model version. Must be one of: {', '.join(valid_models)}"
            )

        result = load_model(model_version)
        return result
    except Exception as e:
        logger.error(f"Error changing model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe/{user_id}")
async def transcribe_audio(
        background_tasks: BackgroundTasks,
        audio_file: UploadFile = File(...),
        patient_name: str = Form(...),
        session_date: datetime = Form(...)
):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(script_dir, "tmp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, audio_file.filename)

    try:
        content = await audio_file.read()
        with open(temp_file_path, 'wb') as f:
            f.write(content)

        session_data = {
            "date": session_date.isoformat(),
            "patient": patient_name
        }
        session_record = pb.collection('sessions').create(session_data)
        logger.info(f"Created session record: {session_record.id}")

        background_tasks.add_task(
            process_audio_transcription,
            temp_file_path,
            audio_file.filename,
            audio_file.content_type,
            session_record.id
        )

        return {
            "status": "processing",
            "message": "Transcription started",
            "session_id": session_record.id
        }

    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        logger.error(f"Error in transcribe_audio: {str(e)}")
        return {"error": str(e)}


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
