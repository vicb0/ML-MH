import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def RF(data):
    X = data.drop(columns=['CLASS'])
    y = data['CLASS']

    rf = RandomForestClassifier()

    with open('random_forest_pickle', 'rb') as f:
        rf = pickle.load(f)

    y_pred = rf.predict(X)
    accuracy = accuracy_score(y, y_pred)

    return {
        "samples": len(X),
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix(y, y_pred)
    }

def main(file_path, fragments):
    results = []
    for i in range(1, fragments+1):
        data = pd.read_hdf(f'{file_path}_{i}.h5')

        print(f"Testando para o fragmento {i}")
        result = RF(data)
        results.append(result)

    with open('results-mh1M.pkl', 'wb') as f:
        pickle.dump(results, f)
    
if __name__ == "__main__":
    main(file_path='./mh_1m_fragments/fragment', fragments=14)