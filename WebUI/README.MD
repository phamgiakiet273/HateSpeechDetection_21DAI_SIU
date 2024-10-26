

# Installation

Clone the repository:
```
git clone https://github.com/phamgiakiet273/HateSpeechDetection_21DAI_SIU.git
```

Install the required packages:
```
pip install -r requirements.txt
```

# Flask app
## Routes

`/` - render a `home.html` template

`/process` **In progress**

- Method: POST
- Parameter: `video_path` - Using video URL to render video comments and transcripts

Description: Results will store as files in `/dataset`.

`/predict`

- Method: POST
- Parameter:   
    - `path_csv` - Path to the input CSV file in `/dataset` after speech to text preprocessing.

Description: Runs the hate speech detection. Results are saved to `result.csv` **(Current result can just save in one path file)**

`/download` - Downloads the result.csv file containing the classification results.

## Functions

`def lstm_word2vec(comment):` Detect toxic comments and speech
 - Input: comment and transcript in string
 - Output: an initial text with 7 positive decimal number (respective with 7 labels)

`def makeData(path_csv):` Detect all comments and transcripts to 'result.csv` file
 - Input: comment or transcript files path
 - Output: `DataFrame` of the result

# Toxic Comment Detection Web Application

This web application allows users to detect toxic comments in YouTube videos by transcribing the video's audio and analyzing the resulting comments for harmful language. The application leverages a model trained to classify comments as toxic, severe toxic, obscene, threat, insult, identity hate, or neutral.

## Features

- **Load YouTube Videos**: Enter a YouTube URL to load the video and display it on the page.
- **Transcription**: Automatically transcribes the video's audio into text.
- **Comment Analysis**: Detects and categorizes the comments from the transcription results.
- **NER Results**: Displays Named Entity Recognition results from the comments.
- **Download Results**: Option to download the processed results as a CSV file.

## How to Use

1. **Load a Video**:
   - Enter a valid YouTube URL in the input field.
   - Click the "Load Video" button to display the video.

2. **Process Comments**:
   - Click the "Process" button to transcribe the video's audio and analyze the comments for toxicity.

3. **Test Input**:
   - Paste a file URL in the provided section to detect hate speech using the model.
   - Click the "Detect Hate Speech" button to run the analysis.

4. **View Results**:
   - The results will be displayed in the "Results" section.
   - Click the "Download Results" button to save the analysis results.

