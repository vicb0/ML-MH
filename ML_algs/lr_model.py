import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# chunksize = 5e3

# benign_fragment_i = 4

# df_class_0 = pd.read_csv(f'./fragments/benign_fragment_{benign_fragment_i}_filtered.csv', chunksize=chunksize, delimiter=';')
# df_class_1 = pd.read_csv('./fragments/malware_fragment_filtered.csv', chunksize=chunksize, delimiter=';')

# benigno = next(df_class_0)
# maligno = next(df_class_1)

# maligno["CLASS"] = 1
# benigno["CLASS"] = 0

# data = pd.concat([benigno, maligno], join='outer').fillna(0)

# data = data.drop(columns=['SHA256', 'NOME', 'PACOTE', 'API_MIN', 'API', 'vt_detection', "VT_Malware_Deteccao", "AZ_Malware_Deteccao"])

data = pd.read_hdf('./data.h5')
data['CLASS'] = np.where(data['vt_detection'] < 4, 0, 1)

data = data.drop(columns=['SHA256', 'NOME', 'PACOTE', 'API_MIN', 'API', "vt_detection", "VT_Malware_Deteccao", "AZ_Malware_Deteccao"])

X = data.drop(columns=['CLASS'])
y = data['CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

LR = LogisticRegression(max_iter=1000).fit(X_train, y_train)
y_pred = LR.predict(X_test)

ac = accuracy_score(y_test, y_pred)

print("ac: ", ac)