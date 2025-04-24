import time
import threading
import queue
import numpy as np
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from real_time_diarization import RealTimeTranscriber, DEFAULT_LANGUAGE
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # Determine language from query parameters (default to DEFAULT_LANGUAGE)
    language = websocket.query_params.get("lang", DEFAULT_LANGUAGE)
    # Get number of speakers from query parameters (default to 2)
    speakers = int(websocket.query_params.get("speakers", 2))
    
    print(f"WebSocket connection accepted, language={language}, speakers={speakers}")
    logger.info(f"WebSocket connection accepted, language={language}, speakers={speakers}")
    
    # Initialize transcriber with selected language and speaker count
    transcriber = RealTimeTranscriber(config={
        "language": language,
        "max_speakers": speakers
    })
    transcriber.start_time = time.time()
    transcriber.is_running = True
    # Start processing threads (VAD, diarization, transcription)
    threads = []
    for func in [
        transcriber.vad_processing_thread,
        transcriber.diarization_thread,
        transcriber.transcription_thread
    ]:
        t = threading.Thread(target=func, daemon=True)
        t.start()
        threads.append(t)

    # Notify client that models are initialized and ready
    try:
        await websocket.send_json({"type": "ready"})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected before models ready")
        return
    # Background task to send transcription results
    async def send_loop():
        while transcriber.is_running:
            try:
                result = transcriber.output_queue.get(timeout=1)
                message = {
                    "speaker_id": result["speaker_id"],
                    "role": result["role"],
                    "text": result["text"],
                    "timestamp": result["timestamp"] - transcriber.start_time
                }
                await websocket.send_json(message)
                transcriber.output_queue.task_done()
            except queue.Empty:
                await asyncio.sleep(0.1)

    send_task = asyncio.create_task(send_loop())
    try:
        # Receive audio chunks and feed to transcriber
        while True:
            data = await websocket.receive_bytes()
            # Expecting float32 PCM audio data
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            try:
                transcriber.audio_queue.put(audio_chunk, block=False)
            except queue.Full:
                continue
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    finally:
        transcriber.stop()
        send_task.cancel() 