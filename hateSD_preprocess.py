import re
import emoji
import fasttext
from huggingface_hub import hf_hub_download
import json
import csv
import pandas as pd

# Download and load the fastText model for language identification
model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
model = fasttext.load_model(model_path)

def preprocess(comment_path, transcript_path, output_comment_path, output_transcript_path):

    def is_english(text):
        """ Function to filter non-English text """
        prediction = model.predict(text, k=2) # Get the top 2 languages present
        for label in prediction[0]:
            if label == "__label__eng_Latn":
                return True
        return False

    # read csv file
    df = pd.read_csv(comment_path, header=None, encoding='utf-8-sig')
    comments = df[0].tolist()
    times = df[1].tolist()
    df = pd.read_csv(transcript_path, header=None, encoding='utf-8-sig')
    transcript = df[0].tolist()

    # ===== clean comments =====
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

    for comment, time in zip(comments, times):
        comment = emoji.replace_emoji(comment, replace=' ')

        # Remove punctuation (except necessary ones like: f**k, sh!t, @$$)
        comment = comment.replace('\n', ' ')
        comment = re.sub(r'[^\w\s*?!@$\']', '', comment)

        # Splits the comment into words, then checks each word to see if it contains any letters. If yes -> perform the leetspeak replacements.
        # case: 94y -> gay but not 1994 cause pure numeric
        comment = ' '.join([
            ''.join([number_to_letter_map.get(c, c) if c.isdigit() and any(ch.isalpha() for ch in word) else c for c in word])
            for word in comment.split()
        ])

        # Check if the comment contains at least 2 letters
        letters = [char for char in comment if char.isalpha()]
        if len(letters) >= 2 and is_english(comment):
            cleaned_comments.append((comment, time))

    # Save cleaned comments directly to a CSV file
    with open(output_comment_path, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        for comment, time in cleaned_comments:
            writer.writerow([comment, time])  # Write each comment as a row

    # ===== clean transcript =====
    cleaned_transcript = []

    for sentence in transcript:
        letters = [char for char in sentence]
        if len(letters) >= 2 and is_english(sentence):
            cleaned_transcript.append(sentence)

    # Save cleaned transcript directly to a CSV file
    with open(output_transcript_path, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        for text in cleaned_transcript:
            writer.writerow([text])  # Write each comment as a row