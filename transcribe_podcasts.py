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
    txt_path = Path("transcripts") / f"{base_path.stem}.txt" 
    #json_path = base_path.with_name(f"{base_path.stem}_full.json")

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
    #with open(json_path, 'w', encoding='utf-8') as f:
     #   json.dump(result, f, indent=2, ensure_ascii=False)

    # Save formatted transcript with timestamps
    with open(txt_path, 'w', encoding='utf-8') as f:
        for segment in result["segments"]:
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"].strip()
            f.write(f'[{start} --> {end}] {text}\n')

    print(f"\nTranscription completed!")
    print(f"Plain transcript saved to: {txt_path}")
    #print(f"Full data saved to: {json_path}")

if __name__ == "__main__":
    # Example usage
    podcast_dir = Path("podcasts")
    transcript_dir = Path("transcripts")
    audio_files = list(podcast_dir.glob("*.mp3"))

    print(f"Found {len(audio_files)} audio files")

   # Process each file
    for audio_file in audio_files:
        # Check if transcript already exists
        transcript_path = transcript_dir / f"{audio_file.stem}.txt"
        
        if transcript_path.exists():
            print(f"Skipping {audio_file.name} - transcript already exists")
            continue
            
        print(f"\nProcessing: {audio_file}")
        try:
            transcribe_audio_with_timestamps(
                str(audio_file),
                model_size="base",
                language="en"
            )
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue
    
    print("\nAll files processed!")