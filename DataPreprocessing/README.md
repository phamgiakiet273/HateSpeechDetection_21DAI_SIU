# Preprocessing data for Hate Speech Detection in YouTube Comments

## Overview

This module preprocesses YouTube comments/transcripts for hate speech detection models. It includes steps to clean, filter, and standardize comments, making them ready for further analysis or model input. The preprocessing ensures that comments retain key features important for detecting offensive or hate speech, while irrelevant elements are removed or adjusted.

- Input: 2 files. Each contains a list of comments/transcripts

- Output: 4 files. The first 2 files contain a list of preprocessed comments/transcripts and the last 2 files contain a list of preprocessed comments/transcripts with NER (NER [TweebankNLP/bertweet-tb2_wnut17-ner] is included for future data analysis)

Example:

- Input: ["""I voted for politicians and now I regret it."" LMFAO🤣", "@Etrajbe its a f*cking joke relax omg!!", "h", "thepiratebay is still up isn't it :)", "are you s.t.u.p.i.d?"]

- Output: ["I voted for politicians and now I regret it LMFAO", "its a f*cking joke relax omg!!", "thepiratebay is still up isn't it", "are you stupid?"]

## Features

-   **Filtering non-English texts**: Filters out comments / sentences in transcript that are not in English (label isn't `__label__eng_Latn`) using *facebook/fasttext-language-identification*.
-  **Cleaning comments**: The function performs several cleaning tasks, including the removal of unnecessary punctuation using RegEx, removing emoji, replacing numbers with corresponding letters to handle leetspeak (e.g., "h4t3" becomes "hate"), and ensuring that comments contain at least 2 letters.

**No spell-checking**: In toxic comments, there might be a correlation between misspelled text and toxicity. Spell-checking system might auto-correct hate speech terms / misspelled texts into completely different words.

## Requirements

To set up and run the project, install the following Python libraries using `pip`:

```bash
pip install -r requirements.txt
```

## How to use this module

```python
from hateSD_preprocess import preprocess

"""
Change path as need
"""
comment_path = "/content/drive/MyDrive/Colab Notebooks/datasets/trivial/save_comment_example.csv"
transcript_path = "/content/drive/MyDrive/Colab Notebooks/datasets/trivial/save_example.csv"
output_comment_path = "/content/drive/MyDrive/Colab Notebooks/datasets/trivial/cleaned_comments.csv"
output_transcript_path = "/content/drive/MyDrive/Colab Notebooks/datasets/trivial/cleaned_transcripts.csv"
output_NER_comment = "/content/drive/MyDrive/Colab Notebooks/datasets/trivial/NER_cleaned_comments.csv"
output_NER_transcript = "/content/drive/MyDrive/Colab Notebooks/datasets/trivial/NER_cleaned_transcripts.csv"

preprocess(comment_path, transcript_path, output_comment_path, output_transcript_path, output_NER_comment, output_NER_transcript)
 ```

