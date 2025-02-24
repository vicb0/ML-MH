import shap


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from utils import drop_low_var

def RF(data):

    X = data.drop(columns=['CLASS'])
    y = data['CLASS']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    rf = RandomForestClassifier(
        criterion='log_loss',
        max_depth=50, 
        min_samples_leaf=2, 
        min_samples_split=8, 
        n_estimators=100,
        random_state=42,
    )

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy best model: {accuracy}") 
    print(classification_report(y_test, y_pred))

    explainer = shap.Explainer(rf, X_train)
    shap_values = explainer(X_train)

    plt.figure()
    shap.plots.beeswarm(shap_values[:, :, 1]) # Todo o conjunto de maligno

    plt.figure()
    shap.plots.waterfall(shap_values[2, :, 1]) # Amostra 2 do conjunto maligno
 
    plt.figure()
    shap.plots.bar(shap_values[:, :, 1].abs.mean(0)) # Média da contribuição de 1 coluna para cada amostra

    plt.figure()
    shap.plots.beeswarm(shap_values[:, :, 0])

    plt.figure()
    shap.plots.waterfall(shap_values[2, :, 0])
 
    plt.figure()
    shap.plots.bar(shap_values[:, :, 0].abs.mean(0))


def main(variance=0.01):
    data = pd.read_hdf(f'./fragments/fragment_1.h5')
    data = drop_low_var(data, var=variance)
    data['CLASS'] = np.where(data['vt_detection'] < 4, 0, 1)
    data = data.drop(columns=['SHA256', 'NOME', 'PACOTE', 'API_MIN', 'API', 'vt_detection', "VT_Malware_Deteccao", "AZ_Malware_Deteccao"])

    RF(data)

    
if __name__ == "__main__":
    main(variance=0.1)