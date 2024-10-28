import os
import pandas as pd
import pickle
from flask import Flask, render_template, send_file, request, redirect, url_for
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

path_parent = os.getcwd()


path_req = path_parent + '/models' #path to models
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'neutral']
path_req = os.path.join(path_parent, 'models')
tokenizer_path = path_req + '/Keras_Tokenizer/token.pkl'
tokenizer = pickle.load(open(tokenizer_path, 'rb'))
loaded_model_word2vec = keras.models.load_model(path_req + "/LSTM_word2vec")
# loaded_model_Glove = keras.models.load_model(path_req + "/LSTM_Glove")

def lstm_word2vec(comment):
    data = [comment]
    comment_data = pd.DataFrame(data, columns = ['Comment'])
    list_tokenized_test = tokenizer.texts_to_sequences(comment_data['Comment'])
    X_te = pad_sequences(list_tokenized_test, maxlen=200)
    preds_test = loaded_model_word2vec.predict(X_te)
    return preds_test

def makeData(path_csv):
    df = pd.read_csv(path_csv)
    final_results = []

    for index, row in df.iterrows():
        comment = row['content']
        if pd.isna(comment):
            final_results.append([None] * 7) 
            continue

        # Use Detection model
        predictions = lstm_word2vec(comment)
        sum_predictions = 0
        arr = ['{:f}'.format(item) for item in predictions[0]]
        
        for i in arr:
            sum_predictions += float(i)
        
        arr.append(str(max([0, (1 - sum_predictions)])))
        sum_predictions += float(arr[6])
        arr = [float((float(item) / sum_predictions) * 100) for item in arr]
        final_results.append(arr)

    results_df = pd.DataFrame(final_results, columns=labels)
    df = df.join(results_df)
    df.to_csv('result.csv', index=False) 

    return df




@app.route('/')
def main():
    return render_template('home.html')

@app.route('/download')
def download_file():
    path = path_parent + '/result.csv'
    return send_file(path, as_attachment=True)

@app.route('/process', methods=['POST'])
def video_processing():
    video_path = request.form.get('youtube_url')
    # Perform processing on video_path if necessary
    print(video_path)
    return "Processing complete for video"

@app.route('/predict', methods=['POST'])
def custom():
    path_csv = request.form.get('task')
    if path_csv:
        makeData(path_csv)
        # return "Processing complete. You can download the results."
    # return "Error: No CSV file path provided"

if __name__ == "__main__":
    app.run(port=5002, debug=True)
