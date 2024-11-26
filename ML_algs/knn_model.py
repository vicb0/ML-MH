import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, LearningCurveDisplay, learning_curve

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


chunksize = 5e3

benign_fragment_i = 2

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

KNN = KNeighborsClassifier(n_neighbors=7)
KNN = KNN.fit(X_train, y_train)

y_pred = KNN.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

ac = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("ac: ", ac)
print("f1: ", f1)

# train_sizes, train_scores, test_scores = learning_curve(KNN, X, y)
# display = LearningCurveDisplay(
#     train_sizes=train_sizes,
#     train_scores=train_scores,
#     test_scores=test_scores, score_name="Score")
# display.plot()
#plt.show()


