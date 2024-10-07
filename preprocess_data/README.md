# Preprocessing data for Hate Speech Detection in YouTube Comments

## Overview

This module preprocesses YouTube comments/transcripts for hate speech detection models. It includes steps to clean, filter, and standardize comments, making them ready for further analysis or model input. The preprocessing ensures that comments retain key features important for detecting offensive or hate speech, while irrelevant elements are removed or adjusted.

Input: a list of comments/transcripts
Output: a list of preprocessed comments/transcripts

## Features

-   **Filtering non-English comments**: Filters out comments that are not in English (label isn't `__label__eng_Latn`) using facebook/fasttext-language-identification.
-  **Cleaning comments**: The function performs several cleaning tasks, including the removal of unnecessary punctuation using RegEx,  replacing numbers with corresponding letters to handle leetspeak (e.g., "h4t3" becomes "hate"), and ensuring that comments contain at least 2 letters.

**No spell-checking**: In toxic comments, there might be a correlation between misspelled text and toxicity. Spell-checking system might auto-correct hate speech terms / misspelled texts into completely different words.

## Requirements

To set up and run the project, install the following Python libraries using `pip`:

```bash
pip install -r requirements.txt
```

## How to use this module

```python
from hateSD_preprocess import preprocess_comments

"""
Change path as need
"""
json_path = "/content/drive/MyDrive/Colab Notebooks/datasets/trivial/save_comment_example.json"
output_csv_path = "/content/drive/MyDrive/Colab Notebooks/datasets/trivial/cleaned_comments.csv"

preprocess_comments(json_path, output_csv_path)
 ```

