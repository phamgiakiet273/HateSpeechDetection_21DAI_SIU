import re
import emoji
import json
import csv
import torch
import pandas as pd
from langdetect import detect

from emoji import demojize
from nltk.tokenize import TweetTokenizer
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


tokenizer = TweetTokenizer()

def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return ""
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return ""
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token

def normalizeTweet(tweet):
    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    normTweet = (
        normTweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )

    return " ".join(normTweet.split())

if __name__ == "__main__":
    print(
        normalizeTweet(
            "SC has first two presumptive cases of coronavirus, DHEC confirms https://postandcourier.com/health/covid19/sc-has-first-two-presumptive-cases-of-coronavirus-dhec-confirms/article_bddfe4ae-5fd3-11ea-9ce4-5f495366cee6.html?utm_medium=social&utm_source=twitter&utm_campaign=user-share… via @postandcourier"
        )
    )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("TweebankNLP/bertweet-tb2_wnut17-ner")
model = AutoModelForTokenClassification.from_pretrained("TweebankNLP/bertweet-tb2_wnut17-ner")
#ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1)

def remove_entities(comment):
    # Get named entities in the comment
    ner_results = ner_pipeline(comment)

    # Extract entity words to remove
    entity_words = [result['word'].replace('@@', '') for result in ner_results]  # Remove '@@' from subword tokens

    # Create a regex pattern to match entity words
    pattern = r'\b(' + '|'.join(re.escape(word) for word in entity_words) + r')\b'

    # Replace entity words with an empty string
    new_comment = re.sub(pattern, '', comment)

    # Clean up extra spaces
    new_comment = re.sub(r'\s+', ' ', new_comment).strip()

    # Brute-force remove '@@ ' if it's still in the comment
    new_comment = new_comment.replace('@@ ', '')

    # Remove any words containing '@' in them
    new_comment = re.sub(r'\S*@\S*', '', new_comment)

    return new_comment


def preprocess(comment_path, transcript_path, output_comment_path, output_transcript_path, output_NER_comment, output_NER_transcript):

    def is_english(text):
        """Function to filter non-English text using langdetect."""
        detected_lang = detect(text)
        return detected_lang == 'en'


    def translate_time_to_english(time_str):
        time_translation_map = {
            'giây': 'seconds',
            'phút': 'minutes',
            'giờ': 'hours',
            'ngày': 'days',
            'tuần': 'weeks',
            'tháng': 'months',
            'năm': 'years',
            'trước': 'ago'
        }

        # Define the regex to capture time patterns (e.g., '5 năm trước', '2 tuần trước')
        time_pattern = r'(\d+)\s+(giây|phút|giờ|ngày|tuần|tháng|năm)\s+trước'
        # Search for the pattern in the string
        match = re.search(time_pattern, time_str)

        if match:
            # Extract the number and the time unit from the matched string
            number = match.group(1)
            vietnamese_time_unit = match.group(2)
            # Translate the time unit and format the final result
            english_time_unit = time_translation_map.get(vietnamese_time_unit, vietnamese_time_unit)
            return f"{number} {english_time_unit} ago"
        # Return the original string if no match is found
        return time_str

    # Read CSV files
    df = pd.read_csv(comment_path, header=None, encoding='utf-8-sig')
    comments = df[0].tolist()
    times = df[1].tolist()  # Assuming times are in column 1
    df = pd.read_csv(transcript_path, header=None, encoding='utf-8-sig')
    transcript = df[0].tolist()

    # ===== Clean Comments =====
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
        comment = comment.replace('\n', ' ')
        comment = re.sub(r'[^\w\s*?!@$\']', '', comment)

        # Leetspeak replacement
        comment = ' '.join([
            ''.join([number_to_letter_map.get(c, c) if c.isdigit() and any(ch.isalpha() for ch in word) else c for c in word])
            for word in comment.split()
        ])

        # Translate time to English
        time = translate_time_to_english(time)

        # Check if the comment contains at least 2 words and is in English
        word_count = len(comment.split())
        if word_count >= 2 and is_english(comment):
            cleaned_comments.append((comment, time))

    # Save cleaned comments to CSV with 3 columns: 'index', 'content', 'time'
    with open(output_comment_path, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['index', 'content', 'time'])
        for idx, (comment, time) in enumerate(cleaned_comments, 1):
            writer.writerow([idx, comment, time])

    # ===== Clean Transcript =====
    cleaned_transcript = []

    for sentence in transcript:
      try:
          letters = [char for char in sentence]  # Check if 'sentence' is iterable as a string
          if len(letters) >= 2 and is_english(sentence):
              cleaned_transcript.append(sentence)
      except Exception as e:
          continue

    with open(output_transcript_path, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['index', 'content', 'time'])
        for idx, sentence in enumerate(cleaned_transcript, 1):
            writer.writerow([idx, sentence, ''])  # No time for transcript

    # ===== Add NER =====
    cleaned_comments_df = pd.DataFrame(cleaned_comments, columns=['content', 'time'])
    cleaned_transcript_df = pd.DataFrame(cleaned_transcript)

    NER_cleaned_comments = cleaned_comments_df['content'].apply(normalizeTweet)
    NER_cleaned_transcript = cleaned_transcript_df[0].apply(normalizeTweet)

    # Apply NER and replace words with the cleaned comments
    NER_cleaned_comments = NER_cleaned_comments.apply(remove_entities)
    NER_cleaned_transcript = NER_cleaned_transcript.apply(remove_entities)

    # Save NER-processed comments
    with open(output_NER_comment, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['index', 'content', 'time'])
        for idx, (comment, time) in enumerate(zip(NER_cleaned_comments, cleaned_comments_df['time']), 1):
            writer.writerow([idx, comment, time])

    # Save NER-processed transcript
    with open(output_NER_transcript, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['index', 'content', 'time'])
        for idx, sentence in enumerate(NER_cleaned_transcript, 1):
            writer.writerow([idx, sentence, ''])  # No time for transcript