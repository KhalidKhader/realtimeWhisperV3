#!/usr/bin/env python3
"""
Simple WebSocket client to test the real-time transcription server.
This client connects to the WebSocket server and sends audio from the microphone.
"""

import asyncio
import websockets
import pyaudio
import numpy as np
import json
import time
import argparse

# Audio parameters
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 1024 * 2  # 2048 samples per chunk (128ms at 16kHz)
FORMAT = pyaudio.paInt16

async def send_audio_stream():
    """Connect to the WebSocket server and stream audio from the microphone."""
    uri = "ws://localhost:8000/listen"
    
    print(f"Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to the server!")
            
            # Initialize PyAudio
            p = pyaudio.PyAudio()
            
            # Open audio stream
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            print("Streaming audio from microphone. Press Ctrl+C to stop.")
            
            # Send audio data
            try:
                while True:
                    # Read audio data from microphone
                    audio_data = stream.read(CHUNK, exception_on_overflow=False)
                    
                    # Convert to numpy array
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    
                    # Send audio data as binary
                    await websocket.send(audio_data)
                    
                    # Receive and print transcription results
                    try:
                        result = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                        result_json = json.loads(result)
                        
                        # Print transcription if available
                        if "text" in result_json:
                            speaker = result_json.get("speaker_id", "Unknown")
                            text = result_json.get("text", "")
                            if text.strip():
                                print(f"{speaker}: {text}")
                    except asyncio.TimeoutError:
                        # No message received within timeout, continue
                        pass
                    
                    # Small delay to prevent flooding the server
                    await asyncio.sleep(0.01)
            
            except KeyboardInterrupt:
                print("\nStopping audio stream...")
            finally:
                # Close audio stream
                stream.stop_stream()
                stream.close()
                p.terminate()
                print("Audio stream closed.")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Real-time Transcription Test Client")
    print("===================================")
    print("This client will stream audio from your microphone to the transcription server.")
    print("Make sure the server is running before starting this client.")
    print("Press Ctrl+C to stop the client.")
    
    # Run the client
    asyncio.run(send_audio_stream())
