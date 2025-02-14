import os
import json

transcripts = []
transcript_dir = "cleaned_transcripts"

for filename in os.listdir(transcript_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(transcript_dir, filename), "r") as f:
            text = f.read()
            transcripts.append({
                "id": filename.split(".")[0],
                "text": text,
                "title": f"Episode {filename.split('.')[0]}",
                "date": "2025-01-01"
            }
            )

def add_basic_metadata(transcripts):
    for transcript in transcripts:
        transcript["metadata"] = {
            "length": len(transcript["text"].split()),
            "source": "Podcast",
            "file_name": f"{transcript['id']}.txt"
        }
    return transcripts

enriched_transcripts = add_basic_metadata(transcripts)

with open("enriched_transcripts.json", "w") as f:
    json.dump(enriched_transcripts, f, indent=4)
