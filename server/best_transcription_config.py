if __name__ == "__main__":
    # Add diagnostic print statement
    print("Starting real-time transcription system...")
    
    # Configuration focused on quality
    config = {
        "language": "en",                       # Use English for transcription
        "use_mps": True,                        # Use Apple Silicon MPS acceleration
        "silence_threshold": 0.01,             # Balanced voice detection threshold
        "chunk_size": 6000,                     # Larger chunks for better context (500ms)
        "min_voice_duration": 0.7,              # Minimum speech segment for quality
        "min_silence_duration": 0.3,            # Better segmentation between utterances
        "speaker_similarity_threshold": 0.9,    # Higher threshold for better speaker distinction
        "max_speakers": 4                      # Limit to 3 speakers for clearer diarization
    }
    
    print("Creating transcriber object...")
    # Create and start the transcriber
    try:
        transcriber = RealTimeTranscriber(config)
        print("Transcriber created successfully. Starting...")
        transcriber.start()
    except Exception as e:
        print(f"ERROR: Failed to start transcriber: {e}")
        import traceback
        traceback.print_exc()