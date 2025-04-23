import pickle
import numpy as np
from sklearn.metrics import confusion_matrix

results = {}

with open('results_mh1m_120000.pkl', 'rb') as f:
    results = pickle.load(f)

samples = 0

CM = np.array([[0, 0], [0, 0]])

for i, result in enumerate(results):
    CM = CM + result["confusion_matrix"]
    samples += result['samples']

TN_total = CM[0][0]

FP_total = CM[0][1]
FN_total = CM[1][0]

TP_total = CM[1][1]


accuracy = (TP_total + TN_total) / samples
precisao_benigno = TN_total / (FN_total + TN_total)
precisao_maligno = TP_total / (FP_total + TP_total)
recall = TP_total / (TP_total + FN_total)

print("Registros", samples)
print("acurácia:", accuracy)
print("precisão benigno: ", precisao_benigno)
print("precisão maligno: ", precisao_maligno)
print("recall: ", recall)
print("Matriz de confusão: \n", CM)