import pandas as pd
import numpy as np
import pickle


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from utils import get_high_var_by_col_1m
from utils import drop_metadata

def RF(data):

    X = data.drop(columns=['CLASS'])
    y = data['CLASS']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    # Params used for GridSearch
    # params ={
    #     "n_estimators": [50, 100, 250],
    #     "criterion": ["gini", "entropy", "log_loss"],
    #     "max_depth":[10, 50, 100, 150, 200],
    #     "min_samples_split":[8, 16, 32, 48],
    #     "min_samples_leaf":[2, 4, 8, 16, 32],
    # }


    ## Best params found for MH-1M
    params ={
        "n_estimators": [500],
        "criterion": ["entropy"],
        "max_depth":[100],
        "min_samples_split":[8],
        "min_samples_leaf":[2],
    }

    rf = GridSearchCV(
        estimator = RandomForestClassifier(random_state=42),
        param_grid = params,
        cv = kf,
        verbose=2,
        n_jobs=4,
    )

    rf.fit(X_train, y_train)

    best_model = rf.best_estimator_

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy best model: {accuracy}")
    results = pd.DataFrame.from_dict(rf.cv_results_)
    #results.to_csv(f'random_forest_mh1m_results{cols}_1.csv', sep=';', index=False)
    print(results)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    results = []
    cm_total = np.ndarray((2, 2))

    print("Teste dos fragmentos do MH-100k")
    for i in range(1, 12):
        mh100k = pd.read_hdf(f'./100k_fragments_for_1m_model/fragment_{i}.h5')

        mh100k['CLASS'] = np.where(mh100k['vt_detection'] < 4, 0, 1)

        mh100k = mh100k[data.columns.to_list()] #  mh100k.drop(columns=['SHA256', 'vt_detection'])

        # mh100k = mh100k.reindex(columns=data.columns)

        X_test = mh100k.drop(columns=['CLASS'])
        y_test = mh100k["CLASS"]

        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy best model: {accuracy}")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        cm_total += cm
        results.append({
            "samples": len(X_test),
            "accuracy": accuracy,
            "confusion_matrix": cm
        })

    print('confusion matrix total:')
    print(cm_total)

    with open(f'results_mh1m_rf_{len(X)}rows_{len(X.columns)}cols.pkl', 'wb') as f:
        pickle.dump(results, f)

def main(col, size, filename):
    data = pd.read_hdf(f"{filename}{size}.h5")
    cols1m = data.columns.to_list()

    cols100k = next(pd.read_csv("./MH-100K/mh_100k_dataset.csv", chunksize=1))
    cols100k = pd.Series(drop_metadata(cols100k).columns).str.lower()
    cols100k = cols100k + cols100k.groupby(cols100k).cumcount().astype(str).replace({'0':''})
    cols100k = cols100k.to_list()

    common = []
    for column in cols1m:
        if column == 'CLASS':
            continue
        category, attribute = column.lower().split("::")
        if category == 'apicalls':
            attribute = f'{attribute}()'
        new_column = f"{category[:-1]}::{attribute}"

        if new_column in cols100k:
            common.append(column)
    common.append("CLASS")
    # cols1m = get_high_var_by_col_1m(col=4000).sort_index()
    # cols1m = cols + cols.groupby(cols).cumcount().astype(str).replace({'0':''})
    # cols1m = cols.tolist()
    # cols1m.append("CLASS")
    data = data[common]

    RF(data)

if __name__ == "__main__":
    main(col=4000, size=120_000, filename='./fragments1m/balanced_fragment_')
