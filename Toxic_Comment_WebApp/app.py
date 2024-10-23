import os
import pandas as pd
import pickle
from flask import Flask, render_template, send_file, request
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

path_parent = os.getcwd()
path_req = path_parent + '/models'
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
path_parent = os.getcwd()
path_req = os.path.join(path_parent, 'models')
tokenizer_path = '/workspace/ai_intern/phuong/project/test/Toxic_Comment_WebApp/models/Keras_Tokenizer/token.pkl'
tokenizer = pickle.load(open(tokenizer_path, 'rb'))
loaded_model_word2vec = keras.models.load_model(path_req + "/LSTM_word2vec")
loaded_model_Glove = keras.models.load_model(path_req + "/LSTM_Glove")

def lstm_word2vec(comment):
    data = [comment]
    comment_data = pd.DataFrame(data, columns = ['Comment'])
    list_tokenized_test = tokenizer.texts_to_sequences(comment_data['Comment'])
    X_te = pad_sequences(list_tokenized_test, maxlen=200)
    preds_test = loaded_model_word2vec.predict(X_te)
    return preds_test

def lstm_glove(comment):
    data = [comment]
    comment_data = pd.DataFrame(data, columns = ['Comment'])
    list_tokenized_test = tokenizer.texts_to_sequences(comment_data['Comment'])
    X_te = pad_sequences(list_tokenized_test, maxlen=200)
    preds_test = loaded_model_Glove.predict(X_te)
    return preds_test

def makeData(path_csv, model_type):
    # Read the CSV file
    df = pd.read_csv(path_csv)
    
    # Initialize an empty list to store the final predictions
    final_results = []

    # Iterate over each row in the CSV file
    for index, row in df.iterrows():
        comment = row['content']
        
        # Skip if the comment is empty or NaN
        if pd.isna(comment):
            final_results.append([None] * 7)  # Append empty result if no comment
            continue

        # Make predictions using the selected model
        if model_type == 1:  # GloVe model
            predictions = lstm_glove(comment)
        else:  # Word2Vec model
            predictions = lstm_word2vec(comment)

        # Process predictions
        sum_predictions = 0
        arr = ['{:f}'.format(item) for item in predictions[0]]
        
        for i in arr:
            sum_predictions += float(i)
        
        # Add final prediction
        arr.append(str(max([0, (1 - sum_predictions)])))
        sum_predictions += float(arr[6])
        
        # Normalize predictions to percentages
        arr = [float((float(item) / sum_predictions) * 100) for item in arr]

        # Append the result for the current comment
        final_results.append(arr)

    # Create new DataFrame with results
    results_df = pd.DataFrame(final_results, columns=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'neutral'])

    # Add the predictions to the original DataFrame
    df = df.join(results_df)

    # Write the result to a CSV file named 'result.csv'
    df.to_csv(path_parent + 'result.csv', index=False)

    return df


@app.route('/')
def main():
    return render_template('home.html')

@app.route('/download')
def download_file():
    path = '/workspace/ai_intern/phuong/project/test/Toxic_Comment_WebAppresult.csv'
    return send_file(path, as_attachment=True)

@app.route('/predict', methods=['POST'])
def custom():
    path_csv = request.form['task']
    arr = []
    if request.method == 'POST':
        if request.form['submit-button'] == 'glove':
            arr = makeData(path_csv, 1)
        elif request.form['submit-button'] == 'word2vec':
            print("word2vec found")
            arr = makeData(path_csv, 2)
        arr.append(path_csv)
        return

if __name__ == "__main__":
    app.run(port=5001, debug=True)