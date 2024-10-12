# Offensive Content Detection in YouTube Comments and Video Transcriptions

## Overview

This project detects offensive or hate speech content from YouTube comments and video transcriptions. The project leverages the Hugging Face `transformers` library and a pre-trained model (`cardiffnlp/twitter-roberta-base-offensive`) to classify text into offensive or non-offensive categories. The input data includes:
- A CSV file containing YouTube comments.
- A CSV file containing video transcriptions.

The output consists of two CSV files:
- One with the comments classified as offensive or not.
- One with the transcriptions split into sentences and classified accordingly.

## Features

- Detect offensive content in YouTube comments.
- Detect offensive content in video transcriptions by splitting the transcription into sentences and classifying each one.
- Outputs the classification label (`OFFENSIVE` or `NOT_OFFENSIVE`) and a confidence score.

## File Structure

```bash
offensive_detection/
│
├── data/
│   ├── input/
│   │   ├── comments.csv                   # Input file for YouTube comments
│   │   └── transcription.csv              # Input file for video transcription
│   ├── output/
│   │   ├── output_comments_results.csv     # Output file with classified comments
│   │   └── output_transcription_sentences_results.csv # Output file with classified transcription sentences
│
├── offensive_detection.py                  # Main Python script for detection
├── requirements.txt                        # Required libraries to run the project
└── README.md                               # This file
```

## Requirements

To set up and run the project, you will need the following libraries:

- `pandas`: For handling CSV files and data manipulation.
- `transformers`: To load the pre-trained RoBERTa model for hate speech detection.
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

1. **Prepare the input files**: Make sure `comments.csv` and `transcription.csv` are placed in the `data/input/` directory.
   
2. **Run the script**:
   Execute the `offensive_detection.py` script to classify both comments and transcription sentences:
   
   ```bash
   python offensive_detection.py
   ```

3. **View the output**: After running the script, two output files will be generated in the `data/output/` directory:
   - `output_comments_results.csv`: Classified results for YouTube comments.
   - `output_transcription_sentences_results.csv`: Classified results for video transcription sentences.

## Notes

- Make sure your input files have the correct format as shown above for smooth processing.

