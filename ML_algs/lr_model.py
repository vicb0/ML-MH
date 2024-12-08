import pandas as pd
import numpy as np
import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from utils import save_results
 

def LR(data):

    X = data.drop(columns=['CLASS'])
    y = data['CLASS']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    LR = LogisticRegression(max_iter=500).fit(X_train, y_train)
    y_pred = LR.predict(X_test)


    cm = confusion_matrix(y_test, y_pred)
    ac = accuracy_score(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    print("accuracy: ", ac)
    print("Confusion Matrix: \n", cm)
    print("Classification Report: \n", cr)

    return [
        ac,
        cr,
        cm,
    ]

def main(fragmented):
    if fragmented:
        maligno = pd.read_csv('./fragments/malware_fragment_filtered.csv', delimiter=';')
        for i in range(0, 11):
            benigno = pd.read_csv(f'./fragments/benign_fragment_{i}_filtered.csv', delimiter=';')
            maligno["CLASS"] = 1
            benigno["CLASS"] = 0

            data = pd.concat([benigno, maligno], join="outer").fillna(0)

            data = data.drop(columns=['SHA256', 'NOME', 'PACOTE', 'API_MIN', 'API', 'vt_detection', "VT_Malware_Deteccao", "AZ_Malware_Deteccao"])
            print(f"Testando para o arquivo {i} benigno")
            results = LR(data)
            save_results("logistic_regression_results", title=f"Benigno {i}", content=results)
    else:
        data = pd.read_hdf('./data.h5')
        data['CLASS'] = np.where(data['vt_detection'] < 4, 0, 1)

        data = data.drop(columns=['SHA256', 'NOME', 'PACOTE', 'API_MIN', 'API', 'vt_detection', "VT_Malware_Deteccao", "AZ_Malware_Deteccao"])

        results = LR(data)
        save_results("logistic_regression_results", title=f"Dataset completo", content=results)

if __name__ == "__main__":
    main(fragmented=True)