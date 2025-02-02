import os
import re
import string


# ------------------------------------------------------------------------------
# 1. Define the cleaning function
# ------------------------------------------------------------------------------

FILLER_WORDS = ["um", "uh", "like", "ah", "er", "yeah", "good", "oh", "yes", "bye"]


def clean_line(line: str) -> str:
   
    line_no_timestamps = re.sub(r"\[.*?\]", "", line)
    allowed_chars = r"[^a-z0-9,.?!'\-\s]"
    line_clean_punct = re.sub(allowed_chars, "", line_no_timestamps.lower())
    
    filler_pattern = r'\b(?:' + '|'.join(re.escape(word) for word in FILLER_WORDS) + r')\b'
    
    line_no_fillers = re.sub(filler_pattern, '', line_clean_punct, flags=re.IGNORECASE)

    tokens = line_no_fillers.split()

    line_final = " ".join(tokens)
    
    return line_final

# ------------------------------------------------------------------------------
# try to combine and segment for each episode
# ------------------------------------------------------------------------------
def segment_transcript(cleaned_lines, chunk_size=5):

    segments = []
    for i in range(0, len(cleaned_lines), chunk_size):
        chunk = cleaned_lines[i : i + chunk_size]
        # Join lines with a space or newline
        segment_text = " ".join(chunk)
        segments.append(segment_text)
    return segments


input_folder = "transcripts"
output_folder = "cleaned_transcripts"

for filename in os.listdir(input_folder):
        if filename.endswith("_2025.txt"):
             input_path = os.path.join(input_folder, filename)
             output_path = os.path.join(output_folder, filename.replace("_2025.txt", "_2025_cleaned.txt"))

             cleaned_lines = []  
             with open(input_path, "r", encoding="utf-8") as infile:
                for line in infile:
                     cleaned_line = clean_line(line)
                     if cleaned_line.strip(): 
                          cleaned_lines.append(cleaned_line)

             segments = segment_transcript(cleaned_lines, chunk_size=5)  

             with open(output_path, "w", encoding="utf-8") as outfile:
                  for segment in segments:
                       outfile.write(segment + "\n\n")

print("Separate cleaned and segmented files created for each podcast episode.")

