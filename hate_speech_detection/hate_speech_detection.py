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

    # Load the pre-trained hate speech detection models
    classifier_offensive = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-offensive")
    classifier_hate = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-hate")

    # Apply the classifiers to YouTube comments
    comments_predictions_offensive = df_comments['comments'].apply(
        lambda x: classifier_offensive(x) if pd.notnull(x) else [{"label": "No comment", "score": 0}]
    )
    comments_predictions_hate = df_comments['comments'].apply(
        lambda x: classifier_hate(x) if pd.notnull(x) else [{"label": "No comment", "score": 0}]
    )

    df_comments['Comment_Offensive_Prediction'] = comments_predictions_offensive.apply(lambda x: x[0]['label'])
    df_comments['Comment_Offensive_Score'] = comments_predictions_offensive.apply(lambda x: x[0]['score'])
    df_comments['Comment_Hate_Prediction'] = comments_predictions_hate.apply(lambda x: x[0]['label'])
    df_comments['Comment_Hate_Score'] = comments_predictions_hate.apply(lambda x: x[0]['score'])

    # Process transcription text: split into sentences and classify each sentence
    transcription_sentences = []
    predictions_offensive = []
    scores_offensive = []
    predictions_hate = []
    scores_hate = []
    timestamps = []

    for index, row in df_transcription.iterrows():
        text = row['video_transcription']
        timestamp = row['timestamp']  # Giả định rằng cột timestamp có mặt

        if pd.notnull(text):
            sentences = split_text_into_sentences(text)
            transcription_sentences.extend(sentences)
            timestamps.extend([timestamp] * len(sentences))  # Gán dấu thời gian cho mỗi câu

            sentence_predictions_offensive = [classifier_offensive(sentence) for sentence in sentences]
            sentence_predictions_hate = [classifier_hate(sentence) for sentence in sentences]

            predictions_offensive.extend([pred[0]['label'] for pred in sentence_predictions_offensive])
            scores_offensive.extend([pred[0]['score'] for pred in sentence_predictions_offensive])

            predictions_hate.extend([pred[0]['label'] for pred in sentence_predictions_hate])
            scores_hate.extend([pred[0]['score'] for pred in sentence_predictions_hate])

    # Create a new DataFrame for split transcription sentences with predictions, scores, and timestamps
    df_transcription_sentences = pd.DataFrame({
        'Sentence': transcription_sentences,
        'Timestamp': timestamps,
        'Transcription_Offensive_Prediction': predictions_offensive,
        'Transcription_Offensive_Score': scores_offensive,
        'Transcription_Hate_Prediction': predictions_hate,
        'Transcription_Hate_Score': scores_hate
    })

    # Save the results to CSV files (comments and transcription sentences)
    df_comments.to_csv(output_comments_path, index=False)
    df_transcription_sentences.to_csv(output_transcription_path, index=False)
    print(f"Kết quả bình luận đã được lưu vào {output_comments_path}")
    print(f"Kết quả câu chuyển thể đã được lưu vào {output_transcription_path}")


# File paths (sửa đường dẫn cho phù hợp)
comments_file_path = r'\hate_speech_detection\data\input\comments.csv'
transcription_file_path = r'\hate_speech_detection\data\input\transcription.csv'
output_comments_path = r'\hate_speech_detection\data\output\output_comments_results.csv'
output_transcription_path = r'\hate_speech_detection\data\output\output_transcription_sentences_results.csv'

# Run the function
detect_hate_speech(comments_file_path, transcription_file_path, output_comments_path, output_transcription_path)
