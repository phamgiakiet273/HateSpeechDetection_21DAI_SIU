# Offensive and Hate Speech Detection in YouTube Comments and Video Transcriptions

## Overview

This project detects offensive and hate speech content from YouTube comments and video transcriptions. It leverages the Hugging Face `transformers` library and two pre-trained models:
- `cardiffnlp/twitter-roberta-base-offensive`: For detecting offensive content.
- `cardiffnlp/twitter-roberta-base-hate`: For detecting hate speech.

The input data includes:
- A CSV file containing YouTube comments.
- A CSV file containing video transcriptions.

The output consists of two CSV files:
- One with the comments classified as either offensive or not, hate speech or not, along with their corresponding scores.
- One with the transcriptions split into sentences, each classified as either offensive or not, hate speech or not, along with their corresponding scores.

## Features

- Detect offensive content in YouTube comments.
- Detect hate speech in YouTube comments.
- Detect both offensive and hate speech in video transcriptions by splitting the text into sentences and classifying each one.
- Outputs the classification labels (`OFFENSIVE`, `NOT_OFFENSIVE`, `HATE`, `NOT_HATE`) and a confidence score for each category.

## File Structure

```bash
hate_speech_detection/
│
├── data/
│   ├── input/
│   │   ├── comments.csv                   # Input file for YouTube comments
│   │   └── transcription.csv              # Input file for video transcriptions
│   ├── output/
│   │   ├── output_comments_results.csv     # Output file with classified comments
│   │   └── output_transcription_sentences_results.csv # Output file with classified transcription sentences
│
├── hate_speech_detection.py                # Main Python script for detection
├── requirements.txt                        # Required libraries to run the project
└── README.md                               # This file
```

## Requirements

To set up and run the project, you will need the following libraries:

- `pandas`: For handling CSV files and data manipulation.
- `transformers`: To load the pre-trained RoBERTa models for offensive and hate speech detection.
- `torch`: Required for running models from the Hugging Face library.
- `gdown`: (Optional) Used if downloading files from Google Drive is necessary.

You can install the required libraries using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Input Files

1. **comments.csv**: Contains YouTube comments. View format in comments.csv!

2. **transcription.csv**: Contains video transcriptions with timestamps. View format in transcription.csv!

## How to Run the Project

1. **Prepare the input files**: Ensure `comments.csv` and `transcription.csv` are placed in the `data/input/` directory.
   
2. **Run the script**:
   Execute the `hate_speech_detection.py` script to classify both comments and transcription sentences:
   
   ```bash
   python hate_speech_detection.py
   ```

3. **View the output**: After running the script, two output files will be generated in the `data/output/` directory:
   - `output_comments_results.csv`: Classified results for YouTube comments, including predictions for both offensive and hate speech.
   - `output_transcription_sentences_results.csv`: Classified results for video transcription sentences, including predictions for both offensive and hate speech.

## Notes

- Ensure your input files have the correct format as shown above for smooth processing.
