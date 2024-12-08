import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, LearningCurveDisplay, learning_curve

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from utils import save_results

def KNN(data):
    X = data.drop(columns=['CLASS'])
    y = data['CLASS']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    KNN = KNeighborsClassifier(n_neighbors=7)
    KNN = KNN.fit(X_train, y_train)

    y_pred = KNN.predict(X_test)

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

    # train_sizes, train_scores, test_scores = learning_curve(KNN, X, y)
    # display = LearningCurveDisplay(
    #     train_sizes=train_sizes,
    #     train_scores=train_scores,
    #     test_scores=test_scores, score_name="Score")
    # display.plot()
    #plt.show()


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
            results = KNN(data)
            save_results("k_neighbors_results", title=f"Benigno {i}", content=results)
    else:
        data = pd.read_hdf('./data.h5')
        data['CLASS'] = np.where(data['vt_detection'] < 4, 0, 1)

        data = data.drop(columns=['SHA256', 'NOME', 'PACOTE', 'API_MIN', 'API', 'vt_detection', "VT_Malware_Deteccao", "AZ_Malware_Deteccao"])

        results = KNN(data)
        save_results("k_neighbors_results", title=f"Dataset completo", content=results)

if __name__ == "__main__":
    main(fragmented=True)

