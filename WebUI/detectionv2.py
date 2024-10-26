import os
import pandas as pd
import pickle
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Đường dẫn đến các mô hình và tokenizer
path_parent = os.getcwd()
path_req = os.path.join(path_parent, 'models')
tokenizer_path = '/workspace/ai_intern/phuong/project/test/Toxic_Comment_WebApp/models/Keras_Tokenizer/token.pkl'

# Tải tokenizer và mô hình
tokenizer = pickle.load(open(tokenizer_path, 'rb'))
loaded_model_word2vec = keras.models.load_model(path_req + "/LSTM_word2vec")
loaded_model_Glove = keras.models.load_model(path_req + "/LSTM_Glove")

# Danh sách nhãn dự đoán
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Hàm sử dụng mô hình Word2Vec
def lstm_word2vec(comment):
    data = [comment]
    comment_data = pd.DataFrame(data, columns=['Comment'])
    list_tokenized_test = tokenizer.texts_to_sequences(comment_data['Comment'])
    X_te = pad_sequences(list_tokenized_test, maxlen=200)
    preds_test = loaded_model_word2vec.predict(X_te)
    return preds_test

# Hàm sử dụng mô hình GloVe
def lstm_glove(comment):
    data = [comment]
    comment_data = pd.DataFrame(data, columns=['Comment'])
    list_tokenized_test = tokenizer.texts_to_sequences(comment_data['Comment'])
    X_te = pad_sequences(list_tokenized_test, maxlen=200)
    preds_test = loaded_model_Glove.predict(X_te)
    return preds_test

# Hàm đọc CSV, xử lý dự đoán, và xuất kết quả
def process_comments(path_csv):
    # Đọc file CSV
    df = pd.read_csv(path_csv)
    
    # Danh sách lưu kết quả dự đoán cuối cùng
    final_results = []

    # Duyệt qua từng dòng bình luận
    for index, row in df.iterrows():
        comment = row['content']
        
        # Bỏ qua nếu bình luận trống
        if pd.isna(comment):
            final_results.append([None] * 7)  # Thêm kết quả rỗng nếu không có bình luận
            continue

        # Dự đoán bằng cả hai mô hình
        predictions_word2vec = lstm_word2vec(comment)
        predictions_glove = lstm_glove(comment)

        # Kết hợp dự đoán bằng cách tính trung bình các giá trị
        combined_predictions = (predictions_word2vec + predictions_glove) / 2

        # Xử lý kết quả dự đoán
        sum_predictions = 0
        arr = ['{:f}'.format(item) for item in combined_predictions[0]]
        
        for i in arr:
            sum_predictions += float(i)
        
        # Tính toán nhãn "neutral" nếu tổng xác suất nhỏ hơn 1
        arr.append(str(max([0, (1 - sum_predictions)])))
        sum_predictions += float(arr[6])
        
        # Chuyển dự đoán thành phần trăm
        arr = [float((float(item) / sum_predictions) * 100) for item in arr]

        # Thêm kết quả của bình luận hiện tại vào danh sách
        final_results.append(arr)

    # Tạo DataFrame cho kết quả dự đoán
    results_df = pd.DataFrame(final_results, columns=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'neutral'])

    # Gộp kết quả với DataFrame ban đầu
    df = df.join(results_df)

    # Lấy tên file input và thêm "result_" vào trước tên
    input_filename = os.path.basename(path_csv)
    output_filename = 'result_' + os.path.splitext(input_filename)[0] + '.csv'
    output_path = os.path.join(path_parent, output_filename)

    # Lưu kết quả vào file CSV với tên mới
    df.to_csv(output_path, index=False)

    print(f"Results saved to {output_path}")

# Hàm main để chạy chương trình
if __name__ == "__main__":
    # Đường dẫn đến file CSV
    path_csv = '/workspace/ai_intern/phuong/project/test/Toxic_Comment_WebApp/dataset/cleaned_comments.csv'  # Thay thế bằng đường dẫn thực tế của bạn

    # Gọi hàm xử lý và dự đoán
    process_comments(path_csv)
