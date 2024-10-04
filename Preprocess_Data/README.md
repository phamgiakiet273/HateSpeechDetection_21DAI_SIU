# Preprocessing data for Hate Speech Detection in YouTube Comments

## Overview

This module preprocesses YouTube comments/transcripts for hate speech detection models. It includes steps to clean, filter, and standardize comments, making them ready for further analysis or model input. The preprocessing ensures that comments retain key features important for detecting offensive or hate speech, while irrelevant elements are removed or adjusted.

Input: a list of comments/transcripts

Output: a list of preprocessed comments/transcripts

## Features

-   **Removing unnecessary punctuation**: Only essential punctuation (`*?!@$&`) is retained, which helps preserve context, especially in comments that may contain offensive language (ex: f**ck, sh!t, @$$, etc).
-   **Removing emojis**: Emojis can often carry sentiment or sarcasm, but for hate speech detection, it's reasonable to remove them since we're focusing on text-based hate.
-   **Replacing numbers with letters**: The code detects and replaces leetspeak (e.g., "h4t3" becomes "hate") to ensure that hate speech terms disguised by numbers are recognized.
-   **Filtering non-English comments**: Using a pre-trained FastText model, the script filters out comments that are not in English (`__label__eng_Latn`).
-   **Ensuring comments contain at least 2 words**: This avoids processing very short comments that are unlikely to be informative for hate speech detection (ex: 'first', 'yep', 'lol', etc).

**No spell-checking**: In toxic comments, there might be a correlation between misspelled text and toxicity. Spell-checking system might auto-correct hate speech terms / misspelled texts into completely different words.

## Requirements

To set up and run the project, install the following Python libraries using `pip`:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes:

-   `fasttext==0.9.2`: For language identification.
-   `emoji==2.8.0`: For handling emojis.
-   `huggingface_hub==0.16.4`: To download the FastText model from Hugging Face.
-   `numpy==1.24.3`

## How to use this module

1. **Prepare your comments**: Insert the list of YouTube comments you wish to preprocess.

2. **Run the script**: Use the `NLP_HateSD_Preprocess.py` script to clean and preprocess the comments.

    ```bash
    python NLP_HateSD_Preprocess.py
    ```
    
3. **Output**: The script will return a list of cleaned comments, ready for use in hate speech detection models.
