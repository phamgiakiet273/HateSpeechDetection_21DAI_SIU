import pandas as pd
from transformers import pipeline
import re

def split_text_into_sentences(text):
    """Tách văn bản thành các câu."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    return sentences

def detect_hate_speech(comments_file_path, transcription_file_path, output_comments_path, output_transcription_path):
    # Load the CSV file for YouTube comments with additional columns for user and time
    df_comments = pd.read_csv(comments_file_path)

    # Load the CSV file for video transcription with a timestamp for each sentence
    df_transcription = pd.read_csv(transcription_file_path)

    # Load the pre-trained hate speech detection model
    classifier = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-offensive")

    # Apply the classifier to YouTube comments
    comments_predictions = df_comments['comments'].apply(
        lambda x: classifier(x) if pd.notnull(x) else [{"label": "No comment", "score": 0}]
    )
    df_comments['Comment_Prediction'] = comments_predictions.apply(lambda x: x[0]['label'])
    df_comments['Comment_Score'] = comments_predictions.apply(lambda x: x[0]['score'])

    # Process transcription text: split into sentences and classify each sentence
    transcription_sentences = []
    predictions = []
    scores = []
    timestamps = []

    for index, row in df_transcription.iterrows():
        text = row['video_transcription']
        timestamp = row['timestamp']  # Assumes timestamp column exists

        if pd.notnull(text):
            sentences = split_text_into_sentences(text)
            transcription_sentences.extend(sentences)
            timestamps.extend([timestamp] * len(sentences))  # Assign timestamp to each sentence

            sentence_predictions = [classifier(sentence) for sentence in sentences]
            predictions.extend([pred[0]['label'] for pred in sentence_predictions])
            scores.extend([pred[0]['score'] for pred in sentence_predictions])

    # Create a new DataFrame for split transcription sentences with predictions, scores, and timestamps
    df_transcription_sentences = pd.DataFrame({
        'Sentence': transcription_sentences,
        'Timestamp': timestamps,
        'Transcription_Prediction': predictions,
        'Transcription_Score': scores
    })

    # Save the results to CSV files (comments and transcription sentences)
    df_comments.to_csv(output_comments_path, index=False)
    df_transcription_sentences.to_csv(output_transcription_path, index=False)
    print(f"Kết quả bình luận đã được lưu vào {output_comments_path}")
    print(f"Kết quả câu chuyển thể đã được lưu vào {output_transcription_path}")

# File paths (sửa đường dẫn cho phù hợp)
comments_file_path = r'\offensive_detection\data\input\comments.csv'
transcription_file_path = r'\offensive_detection\data\input\transcription.csv'
output_comments_path = r'\offensive_detection\data\output\output_comments_results.csv'
output_transcription_path = r'\offensive_detection\data\output\output_transcription_sentences_results.csv'

# Run the function
detect_hate_speech(comments_file_path, transcription_file_path, output_comments_path, output_transcription_path)
