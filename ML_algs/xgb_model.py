import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, ConfusionMatrixDisplay

from utils import drop_low_var_by_col_100k, drop_low_var_100k

import xgboost as xgb
import matplotlib.pyplot as plt

def GB(data, variance, col):

    X = data.drop(columns=['CLASS'])
    y = data['CLASS']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    depth={
        'base_score':[0.2,0.5,0.7],
        'gamma':[0],
        'booster': ['gbtree'],
        'random_state': [42],
        'n_estimators': [10, 20, 30, 40, 50, 70, 100],
        'learning_rate' : [0.1,0.01,0.05,0.0005],
        'max_depth':[5, 8, 10, 15, 20, 25, 30]
    }

    xgb_cl=xgb.XGBClassifier()

    GB = GridSearchCV(
        estimator = xgb_cl,
        param_grid = depth,
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42), # 75% teste e 25% validação
        verbose=2,
        n_jobs=8,
        refit='accuracy'
    )

    GB.fit(X_train, y_train)

    best_model = GB.best_estimator_

    y_pred = best_model.predict(X_test)
    print(f"Accuracy best model: {accuracy_score(y_test, y_pred)}")
    print(f"Precision best model: {precision_score(y_test, y_pred)}")
    print(f"Recall best model: {recall_score(y_test, y_pred)}")

    results = pd.DataFrame.from_dict(GB.cv_results_)
    print(results)
    print(classification_report(y_test, y_pred))
    #Save results
    if col:
        results.to_csv(f'xgb_cv_results{col}.csv', sep=';', index=False)  
    else:
        results.to_csv(f'xgb_cv_results{variance}.csv', sep=';', index=False)

    plt.rcParams.update({'font.size': 20})
    ax = plt.subplot()
    ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=y_pred, ax=ax, cmap="gray", display_labels=['benigno', "maligno"], colorbar=False)
    ax.set_xlabel("Rótulo previsto", fontsize=18)
    ax.set_ylabel("Rótulo real", fontsize=18) 
    ax.plot()
    plt.show()


def main(variance=0.1, col=0):
    data = pd.read_hdf(f'./fragments/fragment_1.h5')

    data['CLASS'] = np.where(data['vt_detection'] < 4, 0, 1)

    if col:
        data = drop_low_var_by_col_100k(data, col)
    else:
        data = drop_low_var_100k(data, variance)

    data = data.drop(columns=['SHA256', 'NOME', 'PACOTE', 'API_MIN', 'API', 'vt_detection', "VT_Malware_Deteccao", "AZ_Malware_Deteccao"])

    GB(data, variance, col)       

if __name__ == "__main__":
    main(col=12000)