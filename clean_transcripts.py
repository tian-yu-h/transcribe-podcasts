import os
import re
import string

# ------------------------------------------------------------------------------
# 1. Define the cleaning function
# ------------------------------------------------------------------------------
FILLER_WORDS = ["um", "uh", "like", "ah", "er", "yeah", "good", "oh", "yes", "bye"]

def clean_line(line: str) -> str:
    """
    Cleans a single line of transcript text by:
    - Removing timestamps (e.g., [0:00:00 --> 0:00:06])
    - Optionally removing extra punctuation or sponsor tags
    - Normalizing spaces
    - Converting to lowercase
    """
    
    # 1. Remove timestamp format [0:00:00 --> 0:00:06]
    #    Regex explanation: 
    #      \[          => literal '['
    #      .*?         => non-greedy match for anything
    #      \]          => literal ']'
    #    The question mark makes it non-greedy, so it doesn't over-match
    line_no_timestamps = re.sub(r"\[.*?\]", "", line)
    
    # 2. Convert to lowercase (optional: if you need consistent casing)
    line_lower = line_no_timestamps.lower()
    
    # 3. Remove extra punctuation if you like (beyond standard .!?)
    #    Example: remove everything except alpha-numeric, some punctuation, and spaces
    #    We'll keep typical punctuation like . , ! ? - ' 
    #    If you want to remove them all, adjust the regex accordingly.
    #    For a simpler approach, let's just remove "excess" punctuation, but
    #    keep basic punctuation for readability.
    #    In many NLP tasks, you might remove them entirely or do tokenization separately.
    
    #    This pattern replaces sequences of [^a-z0-9,.?!' -] with an empty string
    #    The dash needs to be escaped or placed at the end in the class
    allowed_chars = r"[^a-z0-9,.?!'\-\s]"
    line_clean_punct = re.sub(allowed_chars, "", line_lower)
    
    #4 Tokenize   
    tokens = line_clean_punct.split()

    # 5. Remove filler words (single-word approach)
    tokens_no_fillers = [t for t in tokens if t not in FILLER_WORDS]

    # 6. Rejoin tokens
    line_no_fillers = " ".join(tokens_no_fillers)

    # 7. Remove extra spaces
    line_final = re.sub(r"\s+", " ", line_no_fillers)
    
    # Now you have a cleaned line
    return line_final

# ------------------------------------------------------------------------------
# 2. Combine lines from multiple text files
# ------------------------------------------------------------------------------
def read_and_combine_transcripts(file_list):
    """
    Reads multiple .txt files and combines them into a single list of raw lines.
    """
    all_lines = []
    for file_path in file_list:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # We can strip each line here or do it later in cleaning
            lines = [line.strip() for line in lines]
            all_lines.extend(lines)
    return all_lines

# ------------------------------------------------------------------------------
# 3. Apply cleaning to each line
# ------------------------------------------------------------------------------
def clean_transcripts(lines):
    """
    Apply the clean_line function to each line and return the cleaned lines.
    """
    cleaned = [clean_line(line) for line in lines if line.strip()]
    # Filter out empty lines (after cleaning) if desired
    cleaned = [c for c in cleaned if c]
    return cleaned

# ------------------------------------------------------------------------------
# 4. Optional: Segmenting the combined transcript into chunks
# ------------------------------------------------------------------------------
def segment_transcript(cleaned_lines, chunk_size=5):
    """
    Breaks the cleaned transcripts into chunks of `chunk_size` lines each.
    Returns a list of segments (each segment is a single string).
    
    Example: If chunk_size=5, each segment is 5 lines combined into 1 paragraph.
    """
    segments = []
    for i in range(0, len(cleaned_lines), chunk_size):
        chunk = cleaned_lines[i : i + chunk_size]
        # Join lines with a space or newline
        segment_text = " ".join(chunk)
        segments.append(segment_text)
    return segments

# ------------------------------------------------------------------------------
# 5. Main usage example
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Example: define your transcript file paths
    transcript_files = [
        "transcripts/MSR901_ACast.txt",
        "transcripts/TheFoodProgramme-20250103.txt",
        "transcripts/VMP9180343055.txt"
    ]
    
    # 1. Read & combine
    raw_lines = read_and_combine_transcripts(transcript_files)
    
    # 2. Clean
    cleaned_lines = clean_transcripts(raw_lines)
    
    # 3. Segment (optional)
    #    Let's say we want every 5 lines as one chunk
    segmented_texts = segment_transcript(cleaned_lines, chunk_size=5)
    
    # 4. Print or save the results
    print("===== CLEANED LINES (first 10) =====")
    for i, line in enumerate(cleaned_lines[:10]):
        print(f"{i+1:02d}: {line}")
    
    print("\n===== SEGMENTED TEXT (first 3 chunks) =====")
    for i, chunk in enumerate(segmented_texts[:3]):
        print(f"--- Segment {i+1} ---")
        print(chunk)
        print()
    
    # 5. (Optional) Save your cleaned or segmented output to a file
    with open("cleaned_transcript_all.txt", "w", encoding="utf-8") as out_f:
        for line in cleaned_lines:
            out_f.write(line + "\n")
    
    with open("segmented_transcript_all.txt", "w", encoding="utf-8") as out_f:
        for seg in segmented_texts:
            out_f.write(seg + "\n\n")  # separate chunks by blank line
