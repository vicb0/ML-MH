import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

chunksize = 5e3

df_class_0 = pd.read_csv('./fragments/benign_fragment_0_filtered.csv', chunksize=chunksize, delimiter=';')
df_class_1 = pd.read_csv('./fragments/malware_fragment_filtered.csv', chunksize=chunksize, delimiter=';')

benigno = next(df_class_0)
maligno = next(df_class_1)

maligno["CLASS"] = 1
benigno["CLASS"] = 0

data = pd.concat([benigno, maligno], join="outer").fillna(0)
#data = pd.concat([benino, maligno], ignore_index=True) # sem filter

data = data.drop(columns=['SHA256', 'NOME', 'PACOTE', 'API_MIN', 'API'])

X = data.drop(columns=['CLASS'])
y = data['CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print(confusion_matrix(y_true=y_test, y_pred=y_pred))
