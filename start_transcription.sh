#!/bin/bash

# Activate the virtual environment
source whisper_env/bin/activate

# Parse command line arguments
LANGUAGE="en"
SPEAKERS=2

# Process command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -l|--language)
      LANGUAGE="$2"
      shift 2
      ;;
    -s|--speakers)
      SPEAKERS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--language LANG] [--speakers NUM]"
      exit 1
      ;;
  esac
done

# Start the transcriber directly
echo "Starting transcription with language=${LANGUAGE} and speakers=${SPEAKERS}..."
python real_time_diarization.py --language ${LANGUAGE} --speakers ${SPEAKERS} 