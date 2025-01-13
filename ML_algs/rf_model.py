import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from utils import save_results

def RF(data):

    X = data.drop(columns=['CLASS'])
    y = data['CLASS']

    # Divisão em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42) # 75% teste e 25% validação

    best_model = RandomForestClassifier() # modelo que obteve melhor resultado
    best_score = 0
    validation_scores = []

    for i, (train_index, val_index) in enumerate(kf.split(X_train, y_train), start=1):
        print(f"Fold {i}:")

        #Divisão em treinamento e validação
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        #Treinamento
        model = RandomForestClassifier(random_state=42)
        model = model.fit(X_train_fold, y_train_fold)

        # Avaliando o modelo
        y_pred = model.predict(X_val_fold)
        accuracy = accuracy_score(y_val_fold, y_pred)

        if accuracy > best_score:
            best_score, best_model = accuracy, model

        validation_scores.append(accuracy)
        print("Accuracy: ", accuracy)

    #Média dos resultados
    print("Mean: ", np.mean(validation_scores))

    # Teste para o melhor modelo
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Melhor modelo: ")
    print("Acurácia: ", accuracy)
    print(classification_report(y_test, y_pred))

def main(fragmented):

    if fragmented:
        for i in range(1, 12):
            data = pd.read_hdf(f'./fragments/fragment_{i}.h5')
            data['CLASS'] = np.where(data['vt_detection'] < 4, 0, 1)

            data = data.drop(columns=['SHA256', 'NOME', 'PACOTE', 'API_MIN', 'API', 'vt_detection', "VT_Malware_Deteccao", "AZ_Malware_Deteccao"])
            print(f"Testando para o fragmento {i}")
            results = RF(data)
            
            #save_results("random_forest_results", title=f"Fragmento {i}", content=results)
    else:
        data = pd.read_hdf('./data.h5')
        data['CLASS'] = np.where(data['vt_detection'] < 4, 0, 1)

        data = data.drop(columns=['SHA256', 'NOME', 'PACOTE', 'API_MIN', 'API', 'vt_detection', "VT_Malware_Deteccao", "AZ_Malware_Deteccao"])

        results = RF(data)
        save_results("random_forest_results", title=f"Dataset completo", content=results)

if __name__ == "__main__":
    main(fragmented=True)