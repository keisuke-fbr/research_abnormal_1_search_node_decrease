import pickle
import os

def store(result,num):
    # キャッシュファイルのパスを設定
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(current_dir, f".././cache/{num}.pkl")
    with open(file_name, "wb") as f:
        pickle.dump(result, f)

def read(file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(current_dir, f".././cache/{file_name}.pkl")
    with open(file_name, 'rb') as f:
        results_array = pickle.load(f)
    return results_array
