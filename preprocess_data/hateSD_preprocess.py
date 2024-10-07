import re
import fasttext
from huggingface_hub import hf_hub_download
import json

# Download and load the fastText model for language identification
model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
model = fasttext.load_model(model_path)

def preprocess_comments(json_path, output_csv_path):

    def is_english(comment):
        """ Function to filter non-English comments """
        prediction = model.predict(comment, k=2) # Get the top 2 languages present
        for label in prediction[0]:
            if label == "__label__eng_Latn":
                return True
        return False

    # Load the JSON file with utf-8-sig encoding
    with open(json_path, 'r', encoding='utf-8-sig') as f:
        comments_data = json.load(f)
    comments = comments_data

    cleaned_comments = []
    number_to_letter_map = {
        '0': 'o',
        '1': 'l',
        '3': 'e',
        '4': 'a',
        '5': 's',
        '7': 't',
        '9': 'g'
    }

    for comment in comments:
        # Remove punctuation (including emoji, except necessary ones like: f**k, sh!t, @$$)
        comment = comment.replace('\n', ' ')
        comment = re.sub(r'[^\w\s*?!@$]', '', comment)
        # Splits the comment into words, then checks each word to see if it contains any letters. If yes -> perform the leetspeak replacements.
        # case: 94y -> gay but not 1994 cause pure numeric
        comment = ' '.join([
            ''.join([number_to_letter_map.get(c, c) if c.isdigit() and any(ch.isalpha() for ch in word) else c for c in word]) 
            for word in comment.split()
        ])

        # Check if the comment contains at least 2 letters
        letters = [char for char in comment if char.isalpha()]
        if len(letters) >= 2 and is_english(comment):
            cleaned_comments.append(comment)

    import csv

    # Save cleaned comments directly to a CSV file
    with open(output_csv_path, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(["cleaned_comments"])  # Write the header
        for comment in cleaned_comments:
            writer.writerow([comment])  # Write each comment as a row
