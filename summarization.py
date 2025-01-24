from dotenv import load_dotenv
from openai import OpenAI
import os

folder_path = 'cleaned_transcripts'
output_folder_path = 'summary'

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI()

def summarize_transcript(transcript_text):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates concise summaries of podcast transcripts. Focus on collecting details about food, beverages, flavors, ingredients, and recipes mentioned in the podcast. Summarize the content by topics, using 200 words per topic. If the podcast doesn't primarily discuss food-related subjects, summarize the main themes instead. Prioritize the most significant or frequently mentioned items if there's more content than the word limit allows."},
                {"role": "user", "content": f"Please summarize this podcast transcript:\n{transcript_text}"}
            ],
            max_tokens=500,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"
    
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            transcript_text = file.read()

        # Summarize the transcript
        summary = summarize_transcript(transcript_text)

        # Save the summary to a new file
        summary_file_path = os.path.join(output_folder_path, f"summary_{filename}")
        with open(summary_file_path, 'w', encoding='utf-8') as summary_file:
            summary_file.write(summary)

print("Summarization complete. Summaries saved in the same folder as the transcripts.")