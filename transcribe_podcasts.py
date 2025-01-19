import whisper
import datetime
import json
from pathlib import Path
import ssl

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    return str(datetime.timedelta(seconds=round(seconds)))

def transcribe_audio_with_timestamps(audio_path, model_size="base", language=None):
    """
    Transcribe an audio file with timestamps using Whisper
    
    Parameters:
        audio_path (str): Path to the audio file
        model_size (str): Size of the model ("tiny", "base", "small", "medium", "large")
        language (str): Language code (e.g., "en" for English) or None for auto-detection
    """
    # Add this line at the beginning of the function to disable SSL verification
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # Create output paths
    audio_path = Path(audio_path)
    base_path = audio_path.with_suffix('')
    txt_path = base_path.with_suffix('.txt')
    json_path = base_path.with_name(f"{base_path.stem}_full.json")

    # Load model
    print(f"Loading Whisper {model_size} model...")
    model = whisper.load_model(model_size)

    # Transcribe
    print("Starting transcription...")
    result = model.transcribe(
        str(audio_path),
        language="en",
        verbose=True,  # Show progress
    )

    # Save full result as JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Save formatted transcript with timestamps
    with open(txt_path, 'w', encoding='utf-8') as f:
        for segment in result["segments"]:
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"].strip()
            f.write(f'[{start} --> {end}] {text}\n')

    print(f"\nTranscription completed!")
    print(f"Plain transcript saved to: {txt_path}")
    print(f"Full data saved to: {json_path}")

if __name__ == "__main__":
    # Example usage
    audio_file = "podcasts/5e26c02b2af435e1158b1a0f8e81c404.mp3"
    transcribe_audio_with_timestamps(
        audio_file,
        model_size="base",  # Options: tiny, base, small, medium, large
        language="en"       # Optional: specify language or None for auto-detection
    )