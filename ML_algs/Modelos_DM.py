#coding: utf-8

import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV, chi2
import sklearn_relief as relief
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve
from typing import Tuple, List
from random import sample
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from os.path import join
from os import getcwd, cpu_count
import joblib
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTENC, SMOTE
from sklearn.cluster import DBSCAN, KMeans
from sklearn_som.som import SOM
import sklearn_relief as sr


import warnings
warnings.filterwarnings('ignore') 

# Colocar os modelos abaixo
SUPERVISED_MODELS = {
    # "DT": {
    #     "model": DecisionTreeClassifier(),
    #     "params":{
    #         "max_depth":[3,4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    #         "min_samples_split":[2, 4, 8, 12, 16],
    #         "min_samples_leaf": [2,3,4,5,6,7,8],
    #         "criterion": [ 'gini',  'entropy'] 
    #     }
    # },
    "RF": {
        "model": RandomForestClassifier(),
        "params":{
            "max_depth":[3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50],
            "min_samples_split":[2, 4, 8, 12, 16],
            'n_estimators': [50, 100, 150, 200, 300, 500],
            'criterion': ['gini', 'entropy'],
            'max_features': ['auto', 2, 3], #, 4, 6, 8, 10 ,11],
            'min_samples_leaf' : [2, 3, 4, 5, 6, 7, 8]
        }
    },
    "XGB": {
        "model": XGBClassifier(booster='gbtree', random_state=101),
        "params":{
            'XGB__max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50],
            'XGB__n_estimators': [50, 100, 150, 200, 300, 500],
            'XGB__learning_rate': [0.01, 0.03, 0.1, 0.3, 0.5, 1, 2],
            'XGB__gamma': [0, 0.1, 0.5, 1, 2]
        }
    }
    # },
    # "ADAB": {
    #     "model": AdaBoostClassifier(random_state=101),
    #     "params": {
    #         "algorithm": ['SAMME', 'SAMME.R'],
    #         "n_estimators": [710, 720, 730],
    #         "learning_rate": [0.01, 0.1, 0.3, 0.5, 1]
    #     }

    # }
}

UNSUPERVISED_MODELS = {
    "KMEANS": {
        "params": [
            {"n_clusters": 3 },
            {"n_clusters": 4 },
            {"n_clusters": 6 },
            {"n_clusters": 10 }
        ]
    }
}
OUTPUT_PATH = ""


def train(model_type: str, Xtrain: np.array, ytrain: np.array, Xtest: np.array, ytest: np.array):
    """Faz a leitura dos dados
    Args:
        model_type: str
            tipo do modelo
        Xtrain: np.array
            Entradas para treino
        ytrain: np.array
            Label de treino
        Xtest: np.array
            Entradas para teste
        ytest: np.array
            Label de teste
    """
    global SUPERVISED_MODELS, OUTPUT_PATH
    params = {}
    model = None
    if model_type in SUPERVISED_MODELS:
        model = SUPERVISED_MODELS[model_type]["model"]
        params = SUPERVISED_MODELS[model_type]["params"]
    else:
        print("Não existe esse modelo!")
        exit()
    
    grid_search = GridSearchCV(model,param_grid=params , cv=5, scoring='accuracy', n_jobs=cpu_count()//2)
    clf = grid_search.fit(Xtrain,ytrain)

    with open(join(OUTPUT_PATH, f'{model_type}_{datetime.now():%Y-%m-%d_%H-%M-%S}.txt'), 'w', encoding='utf-8') as output_file:
        print('Best parameters found:\n', clf.best_params_)
        output_file.write('Best parameters found: %s' % clf.best_params_)
        output_file.write('\n')

        # All results
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print(f"{mean:0.3f} (+/-{(std * 2):0.03f}) for {params} ")
            output_file.write(f"media: {mean:0.3f} ")
            output_file.write(f" desvio: (+/-{(std * 2):0.03f}) ")
            output_file.write(f" param: {params} ")
            output_file.write("\n")

        ytrue, ypred = ytest, clf.predict(Xtest)

        print("Results on the test set:")
        reports = classification_report(ytrue, ypred)
        print(reports)
        cm = confusion_matrix(ytrue, ypred)
        print(cm)        
        fpr, tpr, thresholds = roc_curve(ytrue, ypred, pos_label=2)
        auc_score = auc(fpr, tpr)
        print(f"AUC: {auc_score}")
        output_file.write(f"\nClassification Report\n{reports}\n\nConfusion Matrix\n{cm}\n\nAUC: {auc_score}")

        plt.cla()
        plt.clf()
        plot_confusion_matrix(clf, Xtest, ytest)
        plt.savefig(join(OUTPUT_PATH, f'{model_type}_CM_{datetime.now():%Y-%m-%d_%H-%M-%S}.png'))
        
        plt.cla()
        plt.clf()
        plot_roc_curve(clf, Xtest, ytest)
        plt.savefig(join(OUTPUT_PATH, f'{model_type}_ROCAUC_{datetime.now():%Y-%m-%d_%H-%M-%S}.png'))
    
    joblib.dump(clf.best_estimator_, join(OUTPUT_PATH, f'{model_type}_{datetime.now():%Y-%m-%d_%H-%M-%S}.joblib'))
    return clf.best_estimator_

def normalize(Xtrain: np.array, Xtest: np.array) -> Tuple[np.array, np.array]:
    """Normaliza os dados fazendo a normalizacao MinMax
    Args:
        Xtrain: np.array
            Dados de treino
        Xtest: np.array
            Dados de teste

    Returns:
    Tuple[np.array, np.array]
        Retorna Xtrain, Xtest
    """
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(Xtrain)
    x_test_scaled = scaler.transform(Xtest)

    joblib.dump(scaler, join(OUTPUT_PATH, f'dataset.scaler')) 
    return x_train_scaled, x_test_scaled


def undersampling(X: np.array, y: np.array) ->  Tuple[np.array, np.array]:
    """Faz o undersampling dos dados
    Args:
        data: pd.DataFrame
            Dataframe com os dados de entrada
    Returns:
    Tuple[np.array, np.array]
        Retorna X, y
    """
    undersampler = RandomUnderSampler(sampling_strategy='majority')
    X_res, y_res = undersampler.fit_resample(X, y)
    return X_res, y_res

def oversampling(X: np.array, y: np.array) ->  Tuple[np.array, np.array]:
    """Faz o oversampling dos dados
    Args:
        data: pd.DataFrame
            Dataframe com os dados de entrada
    Returns:
    Tuple[np.array, np.array]
        Retorna X, y
    """
    #oversampler = RandomOverSampler(random_state=42)
    # oversampler = SMOTENC(random_state=42, categorical_features=[0, 1, 2])
    oversampler = SMOTENC(random_state=42, categorical_features=[0])
    # oversampler = SMOTE(random_state=42)
    X_res, y_res = oversampler.fit_resample(X, y)
    return X_res, y_res

def clustering(model_type: str, X: np.array, y: np.array):
    """Faz a clusterização dos dados
    Args:
        model_type: str
            Tipo de modelo de clusterização
        X: np.array
            Dados de atributos para clusterização
        y: np.array
            Dados de rótulo para clusterização
    """
    global UNSUPERVISED_MODELS, OUTPUT_PATH
    params = None
    if model_type in UNSUPERVISED_MODELS:
        params = UNSUPERVISED_MODELS[model_type]["params"]
    else:
        print("Não existe esse modelo!")
        exit()

    classes = np.unique(y)

    if model_type == "KMEANS":
        for p in params:
            k_means = KMeans()
            k_means.set_params(**p)
            k_means.fit(X)

            groups = {}
            for idx, k in enumerate(k_means.labels_):
                if k not in groups:
                    groups[k] = {str(c): 0 for c in classes}
                groups[k][str(y[idx])] += 1 
            
            print(f"{p['n_clusters']} - {str(groups)}")

            # plt.title(f"Kmenas - {str(p)}")
            # plt.savefig(join(OUTPUT_PATH, f'cluster_{model_type}_{datetime.now():%Y-%m-%d_%H-%M-%S}.png'))
    

def feature_selection(Xtrain: np.array, ytrain: np.array, attributes : List[str]):
    """Faz a seleção de atributos dos dados
    Args:
        Xtrain: np.array
            Atributos de treino
        ytrain: np.array
            Atributos de treino
        attributes: List[str]
            O nome dos atributos
    """
    global OUTPUT_PATH
    with open(join(OUTPUT_PATH, f'feature_sel_{datetime.now():%Y-%m-%d_%H-%M-%S}.txt'), 'w', encoding='utf-8') as feature_sel:
        print("RFC - Decision Tree")
        feature_sel.write("RFC - Decision Tree\n")
        model = RFECV(DecisionTreeClassifier(), cv=5)
        fit = model.fit(Xtrain, ytrain)
        print("Selected Features")
        feature_sel.write("Selected Features\n")
        for idx, name in enumerate(attributes):
            print(f"-> {name}: {fit.support_[idx]} {fit.ranking_[idx]}")
            feature_sel.write(f"-> {name}: {fit.support_[idx]} {fit.ranking_[idx]}\n")

        print("ExtraTreesClassifier")
        feature_sel.write("ExtraTreesClassifier\n")
        model = ExtraTreesClassifier(n_estimators=10)
        model.fit(Xtrain, ytrain)
        print("Selected Features")
        feature_sel.write("Selected Features\n")
        for idx, name in enumerate(attributes):
            print(f"-> {name}: {model.feature_importances_[idx]}")
            feature_sel.write(f"-> {name}: {model.feature_importances_[idx]}\n")

        print("Chi2")
        feature_sel.write("Chi2\n")
        chi2_stats, score = chi2(Xtrain, ytrain)
        print(chi2_stats)
        print("Selected Features")
        feature_sel.write("Selected Features\n")
        for idx, name in enumerate(attributes):
            print(f"-> {name}: {score[idx]}")
            feature_sel.write(f"-> {name}: {score[idx]}\n")



def preprocess(data: pd.DataFrame) -> Tuple[np.array, np.array, np.array, np.array, List[str]]: #Xtrain, ytrain, Xtest, ytest
    """Faz o préprocessamento dos dados
    Args:
        data: pd.DataFrame
            Dataframe com os dados de entrada
    Returns:
    Tuple[np.array, np.array, np.array, np.array, List[str]]
        Retorna Xtrain, ytrain, Xtest, ytest, attributes
    """
    # ------------ Montar aqui o X (input) e o y (label)
    # selected_columns = ['idade', 'sexo', 'raca', 'PCR', 'HTO01', 'LEU01', \
    #                     'LINf1', 'PLA01', 'URE', 'CRE', 'HGB', 'BAST', 'K', 'NA']
    # selected_columns = ['PCR', 'HTO01', 'LEU01', \
    #                     'LINf1', 'PLA01', 'URE', 'HGB', 'K', 'NA']
    # selected_columns = ['idade','PCR', 'HTO01', 'LEU01', \
    #                     'LINf1', 'PLA01', 'URE', 'CRE', 'HGB', 'BAST', 'K', 'NA']
    # selected_columns = ['idade', 'PCR', 'HTO01', 'LEU01', 
    #                     'LINf1', 'PLA01', 'URE', 'HGB', 'K', 'NA']
    # selected_columns = ['PCR', 'HTO01', 'LEU01', 'PLA01', 'URE', 'CRE', 'HGB', 'K']
    selected_columns = ['PCR', 'sexo', 'HTO01', 'HGB']
    sel_data = data[selected_columns]
    sel_data = sel_data.drop(sel_data[~sel_data['PCR'].str.contains('Detectado')].index)                   
    sel_data['PCR'] = sel_data['PCR'].apply(lambda x: 0 if x == "Não Detectado" else 1)
    sel_data['sexo'] = sel_data['sexo'].apply(lambda x: 1 if x == "M" else 0)
    # sel_data['raca'] = sel_data['raca'].apply(lambda x: 0 if x == "Pardo" else x)
    # sel_data['raca'] = sel_data['raca'].apply(lambda x: 1 if x == "Branco" else x)
    # sel_data['raca'] = sel_data['raca'].apply(lambda x: 2 if x == "Preto" else x)
    # sel_data['raca'] = sel_data['raca'].apply(lambda x: 3 if x == "Amarelo" else x)

    print(sel_data)
    print(sel_data.describe())

    # for col in selected_columns:
    #     if 'PCR' not in col:
    #         plt.cla()
    #         plt.clf()
    #         sel_data.boxplot(by=['PCR'], column=[col])
    #         plt.savefig(join(OUTPUT_PATH, f'boxplot_{col}_{datetime.now():%Y-%m-%d_%H-%M-%S}.png'))

    X = sel_data.loc[:,sel_data.columns!="PCR"].to_numpy()
    y = sel_data[['PCR']].to_numpy()

    attributes = selected_columns
    attributes.remove('PCR')

    # ------------ Até aqui

    # Descomentar o que quiser rodar, undersampling e oversampling
    print('Quantidade de dados por classe desbalanceado: ', np.unique(y, return_counts=True))
    # X, y = undersampling(X, y)
    X, y = oversampling(X, y)
    print('Quantidade de dados por classe balanceado: ', np.unique(y, return_counts=True))    

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.9, random_state=42)
    print('Quantidade de dados para Treino: ', np.unique(ytrain, return_counts=True))
    print('Quantidade de dados para Test:   ', np.unique(ytest, return_counts=True))

    Xtrain, Xtest = normalize(Xtrain, Xtest)

    return Xtrain, ytrain, Xtest, ytest, attributes

def read_file(filename: str) -> pd.DataFrame:
    """Faz a leitura dos dados
    Args:
        filename: str
            nome do arquivo de entrada
    Returns:
    pd.DataFrame
        Retorna o dado em um Dataframe
    """
    return pd.read_csv(open(filename, 'r', encoding='utf-8'))

def main(dataset: str, feature_sel: bool, models: str, is_supervised: bool):
    data = read_file(dataset)

    #pré-processamento
    Xtrain, ytrain, Xtest, ytest, attributes = preprocess(data)

    #Seleção de feature ou treino
    if feature_sel:
        feature_selection(Xtrain, ytrain, attributes)
    else:
        for model in models:
            if is_supervised:
                train(model, Xtrain, ytrain, Xtest, ytest)
            else:
                clustering(model, Xtrain, ytrain)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Treinamento de modelos')
    parser.add_argument('-d', "--dataset", dest="dataset", type=str, help="Arquivo csv com os dados", required=True)
    parser.add_argument("--feature_sel", dest="feature_sel", action='store_true', help="Indica se quer apenas selecionar features", required=set(['--aum', '--asm', '--model']).issubset(sys.argv))
    parser.add_argument("--asm", dest="all_supervised_models", action='store_true', help="Indica se quer testar todos os modelos definidos", required=set(['--feature_sel', '--model']).issubset(sys.argv))
    parser.add_argument("--aum", dest="all_unsupervised_models", action='store_true', help="Indica se quer testar todos os modelos definidos", required=set(['--feature_sel', '--model']).issubset(sys.argv))
    parser.add_argument('-m', "--model", dest="model", type=str, help="Tipo do modelo", required=set(['--aum', '--asm', '--feature_sel']).issubset(sys.argv))
    parser.add_argument('-o', "--output_dir", dest="output_dir", type=str, help="tip_de_modelo", required=False, default=getcwd())
    args = parser.parse_args()

    OUTPUT_PATH = args.output_dir

    selected_models = None
    is_supervised = False
    if args.all_unsupervised_models and args.all_supervised_models:
        exit()
    elif args.all_unsupervised_models:
        selected_models = UNSUPERVISED_MODELS
        is_supervised = False
    elif args.all_supervised_models:
        selected_models = SUPERVISED_MODELS
        is_supervised = True
    else:
        if args.model in SUPERVISED_MODELS:
            is_supervised = True
        else:
            is_supervised = False

    main(args.dataset, args.feature_sel, [model for model in selected_models] if selected_models is not None else [args.model], is_supervised)