# Hate Speech Detection on YouTube 

This project provides a Python script for detecting hate speech in YouTube comments using two pre-trained LSTM models: **Word2Vec** and **GloVe**. The models were trained using the dataset from the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge). The script reads comments from a CSV file, processes them through both models, averages the predictions, and exports the results into a new CSV file with added predictions.

## Project Structure

- `models/`: Folder containing the pre-trained models (`LSTM_word2vec` and `LSTM_Glove`).
- `token.pkl`: A saved Keras tokenizer used for text preprocessing.
- `detection.py`: The Python script that handles reading, processing, and predicting comments from the CSV file.

## Requirements

To run this project, you will need the following Python packages installed:

- `pandas`
- `tensorflow`
- `keras`
- `pickle`

You can install the required packages using the following command:

```bash
pip install pandas tensorflow keras

## Dataset

The models were trained using the dataset from the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge). This dataset contains comments labeled with various categories related to toxic speech, including:

- **Toxic**
- **Severe toxic**
- **Obscene**
- **Threat**
- **Insult**
- **Identity hate**

The dataset contains millions of rows of labeled comments, making it a great resource for detecting hate speech in YouTube comments.

## Usage

1. Place the required LSTM models (`LSTM_word2vec` and `LSTM_Glove`) in the `models/` directory.
2. Ensure that the tokenizer (`token.pkl`) is located in the specified path.
3. Prepare the CSV file containing YouTube comments. The CSV should have the following columns:
    - `index`: Unique identifier for each row.
    - `content`: The text of the YouTube comment to be analyzed.
    - `time`: The timestamp of the comment (this field is optional and will not affect the prediction).

## Running the Script

The detection script is located in the file `detection.py`. To run the script, use the following command:

```bash
python detection.py

### The script will:

1. Read the comments from the CSV file (ensure to update the `path_csv` variable with the correct path).
2. Process each comment through both the **Word2Vec** and **GloVe** models.
3. Average the predictions from both models.
4. Export the result to a new CSV file named `result_<input_filename>.csv` (e.g., `result_youtube_comments.csv`).

The default configuration uses both **Word2Vec** and **GloVe** models to improve prediction accuracy. However, if you wish to use only one model, you can customize the script by commenting out the model you don’t want to use in the `process_comments` function within `detection.py`.

### Input CSV Format

Your input CSV file should have the following structure:

| index | content           | time         |
|-------|-------------------|--------------|
| 1     | Example comment 1 | 1 year ago   |
| 2     | Example comment 2 | 2 year ago   |

The `content` column contains the text of the YouTube comment that will be analyzed for hate speech. The `time` column is optional and will not be used in the analysis.

### Output

The script will generate a CSV file named `result_<input_filename>.csv` containing the original data along with the following predicted labels:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`
- `neutral`: The neutral probability based on model predictions.

The `neutral` label represents the remaining probability that a comment is non-toxic or neutral. After the models predict the probabilities for the six hate speech categories, the `neutral` score is calculated as the remaining probability to ensure the total probability sums up to 100%.

For example, if a comment has the following predicted probabilities for hate speech categories:

```python
predictions = [0.12, 0.05, 0.08, 0.01, 0.03, 0.02]  # toxic, severe_toxic, etc.

The sum of the predictions for the six categories is:

```python
sum_predictions = sum(predictions)  # 0.31

Then, the neutral score is calculated as:

```python
neutral_score = max(0, 1 - sum_predictions)  # 0.69

Finally, the predictions are normalized so that all categories (including neutral) sum to 100%. The resulting percentages for each category are:

```python
predictions.append(neutral_score)
predictions = [float((float(item) / (sum_predictions + neutral_score)) * 100) for item in predictions]

### Example Output

The output CSV will have the following format:

| index | content           | time         | toxic | severe_toxic | obscene | threat | insult | identity_hate | neutral |
|-------|-------------------|--------------|-------|--------------|---------|--------|--------|---------------|---------|
| 1     | Example comment 1 | 1 year ago   | 12.3  | 5.4          | 8.7     | 0.1    | 3.5    | 2.4           | 67.6    |
| 2     | Example comment 2 | 2 year ago   | 0.5   | 0.2          | 0.1     | 0.0    | 0.3    | 0.1           | 98.8    |
