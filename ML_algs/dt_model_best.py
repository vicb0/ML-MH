# -*- coding: utf-8 -*-
"""

@author: Felipe
"""
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.model_selection import train_test_split, LearningCurveDisplay, learning_curve, GridSearchCV

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


chunksize = 5e3

benign_fragment_i = 4

df_class_0 = pd.read_csv(f'./fragments/benign_fragment_{benign_fragment_i}_filtered.csv', chunksize=chunksize, delimiter=';')
df_class_1 = pd.read_csv('./fragments/malware_fragment_filtered.csv', chunksize=chunksize, delimiter=';')

benigno = next(df_class_0)
maligno = next(df_class_1)

maligno["CLASS"] = 1
benigno["CLASS"] = 0

data = pd.concat([benigno, maligno], join='outer').fillna(0)

data = data.drop(columns=['SHA256', 'NOME', 'PACOTE', 'API_MIN', 'API', 'vt_detection', "VT_Malware_Deteccao", "AZ_Malware_Deteccao"])

X = data.drop(columns=['CLASS'])
y = data['CLASS']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)


depth={
    "max_depth":[10, 15],
    "min_samples_split":[4],
}
DTC = DecisionTreeClassifier()
DTC_Grid=GridSearchCV(DTC,param_grid=depth , cv=6, scoring='f1')
DTC=DTC_Grid.fit(X_train,y_train) 

print('Best parameters found: \n', DTC.best_params_)

#plot_tree(DTC)
#plt.show()

y_pred = DTC.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

ac = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("ac: ", ac)
print("f1: ", f1)

#
# train_sizes, train_scores, test_scores = learning_curve(DTC, X, y)
# display = LearningCurveDisplay(
#     train_sizes=train_sizes,
#     train_scores=train_scores,
#     test_scores=test_scores, score_name="Score")
# display.plot()
#plt.show()


