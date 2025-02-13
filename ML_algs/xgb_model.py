import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

from utils import drop_low_var_by_col, drop_low_var

def GB(data, variance, col):

    X = data.drop(columns=['CLASS'])
    y = data['CLASS']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    params ={
        "n_estimators": [50, 100, 250], #Número de árvores na floresta
        "criterion": ["friedman_mse", "squared_error"], #Medição da qualidade de uma decisão tomada em um nó
        "max_depth":[10, 20, 50, 100, 150, 200], # altura máxima da árvore
        "min_samples_split":[8, 16, 32, 48], # Número mínimo de registros para dividir um nó em outro
        "min_samples_leaf":[2, 4, 8, 16, 32], # Número mínimo de folhas
    }

    GB = GridSearchCV(
        estimator = GradientBoostingClassifier(),
        param_grid = params,
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42), # 75% teste e 25% validação
        verbose=2,
        n_jobs=8,
    )

    GB.fit(X_train, y_train)

    best_model = GB.best_estimator_

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy best model: {accuracy}") 

    results = pd.DataFrame.from_dict(GB.cv_results_)
    if col:
        results.to_csv(f'xgb_cv_results{col}.csv', sep=';', index=False)  
    else:
        results.to_csv(f'xgb_cv_results{variance}.csv', sep=';', index=False)  
    print(results)
    
    print(classification_report(y_test, y_pred))


def main(variance=0.1, col=0):
    data = pd.read_hdf(f'./fragments/fragment_1.h5')

    data['CLASS'] = np.where(data['vt_detection'] < 4, 0, 1)

    if col:
        data = drop_low_var_by_col(data, col)
    else:
        data = drop_low_var(data, variance)

    data = data.drop(columns=['SHA256', 'NOME', 'PACOTE', 'API_MIN', 'API', 'vt_detection', "VT_Malware_Deteccao", "AZ_Malware_Deteccao"])

    GB(data, variance, col)       

if __name__ == "__main__":
    main(col=4000)