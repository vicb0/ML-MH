import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from utils import save_results

def RF(data):
    X = data.drop(columns=['CLASS'])
    y = data['CLASS']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print(confusion_matrix(y_true=y_test, y_pred=y_pred))

    return [
        accuracy_score(y_test, y_pred),
        classification_report(y_test, y_pred),
        confusion_matrix(y_true=y_test, y_pred=y_pred),
    ]

def main(fragmented):

    if fragmented:
        maligno = pd.read_csv('./fragments/malware_fragment_filtered.csv', delimiter=';')
        for i in range(0, 11):
            benigno = pd.read_csv(f'./fragments/benign_fragment_{i}_filtered.csv', delimiter=';')
            maligno["CLASS"] = 1
            benigno["CLASS"] = 0

            data = pd.concat([benigno, maligno], join="outer").fillna(0)

            data = data.drop(columns=['SHA256', 'NOME', 'PACOTE', 'API_MIN', 'API'])
            print(f"Testando para o arquivo {i} benigno")
            results = RF(data)
            save_results("random_forest_results", title=f"Benigno {i}", content=results)
    else:
        data = pd.read_hdf('./data.h5')
        data['CLASS'] = np.where(data['vt_detection'] < 4, 0, 1)

        data = data.drop(columns=['SHA256', 'NOME', 'PACOTE', 'API_MIN', 'API'])

        results = RF(data)
        save_results("random_forest_results", title=f"Dataset completo", content=results)

if __name__ == "__main__":
    main(fragmented=True)