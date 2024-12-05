# -*- coding: utf-8 -*-
"""

@author: Felipe
"""
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix


chunksize = 5e3

df_class_0 = pd.read_csv('./fragments/benign_fragment_0_filtered.csv', chunksize=chunksize, delimiter=';')
df_class_1 = pd.read_csv('./fragments/malware_fragment_filtered.csv', chunksize=chunksize, delimiter=';')

benigno = next(df_class_0)
maligno = next(df_class_1)

maligno["CLASS"] = 1
benigno["CLASS"] = 0

data = pd.concat([benigno, maligno], join="outer").fillna(0)

data = data.drop(columns=['SHA256', 'NOME', 'PACOTE', 'API_MIN', 'API', 'vt_detection', "VT_Malware_Deteccao", "AZ_Malware_Deteccao"])

X = data.drop(columns=['CLASS'])
y = data['CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

GBC = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=0, min_samples_split=10,
    min_samples_leaf=4).fit(X_train, y_train)

score = GBC.score(X_test, y_test)


print(score)