import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix

from utils import save_results

def GB(data):

    X = data.drop(columns=['CLASS'])
    y = data['CLASS']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    GBC = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=0, min_samples_split=10,
        min_samples_leaf=4).fit(X_train, y_train)

    score = GBC.score(X_test, y_test)
    print(score)

    return [score]

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
            results = GB(data)
            save_results("gradient_boost_results", title=f"Benigno {i}", content=results)
    else:
        data = pd.read_hdf('./data.h5')
        data['CLASS'] = np.where(data['vt_detection'] < 4, 0, 1)

        data = data.drop(columns=['SHA256', 'NOME', 'PACOTE', 'API_MIN', 'API', 'vt_detection', "VT_Malware_Deteccao", "AZ_Malware_Deteccao"])

        results = GB(data)
        save_results("gradient_boost_results", title=f"Dataset completo", content=results)

if __name__ == "__main__":
    main(fragmented=True)