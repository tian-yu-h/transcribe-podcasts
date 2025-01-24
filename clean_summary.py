import os
import re

def filter_and_combine_topics(file_path, keywords):
    combined_content = ""
    current_topic = None
    include_content = False

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.endswith(":"):
                current_topic = line[:-1]  # Remove the colon
                normalized_topic = current_topic.lower().replace('and', '').replace('&', '')
                include_content = any(keyword.lower() in normalized_topic.lower() for keyword in keywords)
            elif include_content and line:
                combined_content += line + "\n"

    return combined_content

def combine_files(directory, keywords):
    combined_text = ""
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            filtered_text = filter_and_combine_topics(file_path, keywords)
            if filtered_text:  
                combined_text += f"From {filename}:\n{filtered_text}\n\n"
    return combined_text.strip()

# Define the keywords to include
topics_keywords = ["food", "flavor", "beverage", "recipe", "ingredient", "scent"]

# Specify the directory containing your text files
directory = "summary"

# Check if the directory exists
if not os.path.exists(directory):
    print(f"Error: The directory '{directory}' does not exist.")
else:
    # Combine and filter the files
    combined_text = combine_files(directory, topics_keywords)

    # Save the combined text to a new file
    output_file_path = "combined_filtered_topics.txt"
    with open(output_file_path, "w") as outfile:
        outfile.write(combined_text)

    print(f"Filtered and combined content has been saved to {output_file_path}")

#the combined_filtered_topics.txt needs to be reviewed before proceeding.
#clean the combined topics

def clean_content(text):
    # Remove titles "From ... .txt:"
    cleaned_text = re.sub(r"From .*?\.txt:\n", "", text)
    
    # Remove "- " from the beginning of lines
    cleaned_text = re.sub(r"^- ", "", cleaned_text, flags=re.MULTILINE)
    
    # Remove extra blank lines (more than one consecutive newline)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
    return cleaned_text.strip()

# Read the input file
with open('combined_filtered_topics.txt', 'r', encoding='utf-8') as file:
    content = file.read()

# Clean the content
cleaned_content = clean_content(content)

# Write the cleaned content to a new file
with open('cleaned_combined_topics.txt', 'w', encoding='utf-8') as file:
    file.write(cleaned_content)



#perpare for bertopic modeling
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

with open('cleaned_combined_topics.txt', 'r', encoding='utf-8') as file:
    combined_text = file.read()

def preprocess_for_bertopic(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    
    return " ".join(tokens)

# Preprocess the combined text
preprocessed_text = preprocess_for_bertopic(combined_text)

# Save the preprocessed text to a new file
with open("preprocessed_for_bertopic.txt", "w") as outfile:
    outfile.write(preprocessed_text)
