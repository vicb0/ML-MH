import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from utils import get_high_var_by_col_1m
from utils import drop_metadata

def GB(data):

    X = data.drop(columns=['CLASS'])
    y = data['CLASS']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    params={
        'base_score':[1],
        'random_state': [42],
        'n_estimators': [250],
        'learning_rate' : [0.1],
        'max_depth':[25],
    }

    #rf = GridSearchCV(
    #    estimator =  xgb.XGBClassifier(),
    #    param_grid = params,
    #    cv = kf,
    #    verbose=2,
    #    n_jobs=4,
    #)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
    
    gb = xgb.XGBClassifier(random_state=42, base_score=0.5, n_estimators=50, learning_rate=0.1, max_depth=10, scale_pos_weight=0.5, eval_metric='logloss')

    gb.fit(eval_set=[(X_train, y_train), (X_val, y_val)])

    #best_model = gb.best_estimator_

    y_pred = gb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy best model: {accuracy}")
    #results = pd.DataFrame.from_dict(rf.cv_results_)
    #results.to_csv(f'random_forest_mh1m_results{cols}_1.csv', sep=';', index=False)
    #print(results)
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

        y_pred = gb.predict(X_test)
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

    with open(f'results_mh1m_xgboost_{len(X)}rows_{len(X.columns)}cols.pkl', 'wb') as f:
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

    GB(data)

if __name__ == "__main__":
    main(col=4000, size=120_000, filename='./fragments1m/balanced_fragment_')
