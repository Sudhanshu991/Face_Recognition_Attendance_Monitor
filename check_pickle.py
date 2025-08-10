# check_pickle.py
import pickle

with open("dataset/encodings.pkl", "rb") as f:
    data = pickle.load(f)
    print(data)
    print(type(data))
