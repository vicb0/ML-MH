import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

from utils import drop_low_var_by_col

def RF(data, variance):

    X = data.drop(columns=['CLASS'])
    y = data['CLASS']

    # Divisão em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42) # 75% teste e 25% validação
    
    # params ={
    #     "n_estimators": [50, 100, 250], #Número de árvores na floresta
    #     "criterion": ["gini", "entropy", "log_loss"], #Medição da qualidade de uma decisão tomada em um nó
    #     "max_depth":[10, 20, 50, 100, 150, 200], # altura máxima da árvore
    #     "min_samples_split":[8, 16, 32, 48], # Número mínimo de registros para dividir um nó em outro
    #     "min_samples_leaf":[2, 4, 8, 16, 32], # Número mínimo de folhas
    # }

    params ={
        "n_estimators": [100], #Número de árvores na floresta
        "criterion": ["log_loss"], #Medição da qualidade de uma decisão tomada em um nó
        "max_depth":[50], # altura máxima da árvore
        "min_samples_split":[8], # Número mínimo de registros para dividir um nó em outro
        "min_samples_leaf":[2], # Número mínimo de folhas
    }

    rf = GridSearchCV(
        estimator = RandomForestClassifier(),
        param_grid = params,
        cv = kf,
        verbose=2,
        n_jobs=8,
    )

    rf.fit(X_train, y_train)

    best_model = rf.best_estimator_

    with open('random_forest_pickle', 'wb') as f:
        pickle.dump(best_model, f)

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy best model: {accuracy}") 

    results = pd.DataFrame.from_dict(rf.cv_results_)
    #results.to_csv(f'random_forest_cv_results{variance}.csv', sep=';', index=False)  
    print(results)
    print(classification_report(y_test, y_pred))

def main(variance=0.01, col=4000):
    
    for i in range(1, 12):
        data = pd.read_hdf(f'./fragments/fragment_{i}.h5')

        data = drop_low_var_by_col(data, col=col)

        data['CLASS'] = np.where(data['vt_detection'] < 4, 0, 1)
        
        data = data.drop(columns=['SHA256', 'NOME', 'PACOTE', 'API_MIN', 'API', 'vt_detection', "VT_Malware_Deteccao", "AZ_Malware_Deteccao"])
        print(f"Testando para o fragmento {i}")

        RF(data, variance)

        break
    

if __name__ == "__main__":
    main(variance=0.19) # 1972