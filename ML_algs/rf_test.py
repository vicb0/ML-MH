import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


data = pd.read_hdf('./data.h5')
data['CLASS'] = np.where(data['vt_detection'] < 4, 0, 1)

data = data.drop(columns=['SHA256', 'NOME', 'PACOTE', 'API_MIN', 'API', "vt_detection", "VT_Malware_Deteccao", "AZ_Malware_Deteccao"])

X = data.drop(columns=['CLASS'])
y = data['CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print(confusion_matrix(y_true=y_test, y_pred=y_pred))
