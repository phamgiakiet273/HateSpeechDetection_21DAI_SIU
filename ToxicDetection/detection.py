import os
import pandas as pd
import pickle
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

path_parent = os.getcwd()
path_req = os.path.join(path_parent, 'models')
tokenizer_path = '/models/Keras_Tokenizer/token.pkl'

tokenizer = pickle.load(open(tokenizer_path, 'rb'))
loaded_model_word2vec = keras.models.load_model(path_req + "/LSTM_word2vec")
loaded_model_Glove = keras.models.load_model(path_req + "/LSTM_Glove")

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def lstm_word2vec(comment):
    data = [comment]
    comment_data = pd.DataFrame(data, columns=['Comment'])
    list_tokenized_test = tokenizer.texts_to_sequences(comment_data['Comment'])
    X_te = pad_sequences(list_tokenized_test, maxlen=200)
    preds_test = loaded_model_word2vec.predict(X_te)
    return preds_test

def lstm_glove(comment):
    data = [comment]
    comment_data = pd.DataFrame(data, columns=['Comment'])
    list_tokenized_test = tokenizer.texts_to_sequences(comment_data['Comment'])
    X_te = pad_sequences(list_tokenized_test, maxlen=200)
    preds_test = loaded_model_Glove.predict(X_te)
    return preds_test

def process_comments(path_csv):
    df = pd.read_csv(path_csv)
    
    final_results = []

    for index, row in df.iterrows():
        comment = row['content']
        
        if pd.isna(comment):
            final_results.append([None] * 7)
            continue

        predictions_word2vec = lstm_word2vec(comment)
        predictions_glove = lstm_glove(comment)

        combined_predictions = (predictions_word2vec + predictions_glove) / 2

        sum_predictions = 0
        arr = ['{:f}'.format(item) for item in combined_predictions[0]]
        
        for i in arr:
            sum_predictions += float(i)
        
        arr.append(str(max([0, (1 - sum_predictions)])))
        sum_predictions += float(arr[6])
        
        arr = [float((float(item) / sum_predictions) * 100) for item in arr]

        final_results.append(arr)

    results_df = pd.DataFrame(final_results, columns=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'neutral'])

    df = df.join(results_df)

    input_filename = os.path.basename(path_csv)
    output_filename = 'result_' + os.path.splitext(input_filename)[0] + '.csv'
    output_path = os.path.join(path_parent, output_filename)

    df.to_csv(output_path, index=False)

    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    path_csv = '/dataset/cleaned_comments.csv'

    process_comments(path_csv)
