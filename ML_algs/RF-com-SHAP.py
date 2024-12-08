# -*- coding: utf-8 -*-
"""

@author: Felipe
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn import tree
import shap

import warnings
warnings.filterwarnings("ignore")

def modelosRF(ds, name):
    
	#depth={"max_depth":[3,4, 5, 6, 7, 8, 9, 10],
	#	"min_samples_split":[2, 4, 8, 12, 16],
	#	'n_estimators': [ 100],
	#	'criterion': ['gini', 'entropy'],
	#	'max_features': ['auto', 2, 3, 4, 6, 8,10,12,14,16,18],
	#	'min_samples_leaf' : [2,3,4,5,6,7,8]}
	
	
	depth={"max_depth":[4],
		"min_samples_split":[8],
		'n_estimators': [100],
		'criterion': ['entropy'],
		'max_features': [8],
		'min_samples_leaf' : [3]}


	X_train,y_train=ds[0]
	X_test,y_test=ds[1]
	
	RF=RandomForestClassifier()
	
	DTC_Grid=GridSearchCV(RF, param_grid=depth , cv=5, scoring=['f1', 'accuracy', 'recall', 'roc_auc',  'precision'], refit='accuracy')
	DTC=DTC_Grid.fit(X_train,y_train)
    #RF=RandomForestClassifier(max_depth=10, min_samples_split=4, n_estimators=100, max_features=10 )
    #RF.fit(X_train,y_train)    
    #y_true, y_pred2 = y_test, RF.predict(X_test)
    #cm=confusion_matrix(y_test, y_pred2)
    #print(cm)
    #print(classification_report(y_true, y_pred2))
    #print()
   
	arquivo = open('saidaRF.txt','w')
    
    #print('Best parameters found:\n', DTC.best_params_)
    #arquivo.write('Best parameters found: %s' % DTC.best_params_)
    #arquivo.write('\n')

    # All results
	means = DTC.cv_results_['mean_test_accuracy']
	means2 = DTC.cv_results_['mean_test_precision']
	means3 = DTC.cv_results_['mean_test_recall']
	means4 = DTC.cv_results_['mean_test_f1']
	means5 = DTC.cv_results_['mean_test_roc_auc']
	stds = DTC.cv_results_['std_test_f1']
	for mean,mean2,mean3,mean4,mean5, std, params in zip(means,means2,means3,means4,means5, stds, DTC.cv_results_['params']):
		print("%0.3f (+/-%0.03f) for %r" % (mean4, std * 2, params))
		arquivo.write('acurácia: %0.3f ' % mean)
		arquivo.write('precision: %0.3f ' % mean2)
		arquivo.write('recall: %0.3f ' % mean3)
		arquivo.write('f1: %0.3f ' % mean4)
		arquivo.write('roc_auc: %0.3f ' % mean5)
		std=std*2
		#arquivo.write(' desvio: (+/-%0.03f) ' % std)
		arquivo.write(' param: %s ' % params)
		arquivo.write('\n')

	print('Best parameters found:\n', DTC.best_params_)
	arquivo.write('Best parameters found: %s' % DTC.best_params_)
	arquivo.write('\n')

	y_true, y_pred = y_test, DTC.predict(X_test)

	# Compute ROC curve and ROC area for each class
	fpr = dict()
	tpr = dict()
	roc_auc = dict()

	print('Results on the test set:')
	print(classification_report(y_true, y_pred))

	acc=accuracy_score(y_test, y_pred,normalize=True)
	precision=precision_score(y_test, y_pred)
	f1=f1_score(y_test, y_pred, average='micro')    	
	recall=recall_score(y_test, y_pred, average='weighted')
	cm=confusion_matrix(y_test, y_pred)

	print("accuracy", acc)
	print("precision", precision)
	print("recall", recall)    
	print("f1", f1)
	print("cm", cm)

	arquivo.write(classification_report(y_true, y_pred))
	arquivo.write('Resultado do test set: \n')
	arquivo.write(' accuracy: %0.03f \n' % acc)
	arquivo.write(' precision: %0.03f \n' % precision)
	arquivo.write(' recall: %0.03f \n' % recall)
	arquivo.write(' f1: %0.03f \n\n' % f1)
	arquivo.write('Matriz de onfusao: \n')
	arquivo.write(' %0.03f ' % cm[0][0])
	arquivo.write(' %0.03f \n' % cm[0][1])
	arquivo.write(' %0.03f ' % cm[1][0])
	arquivo.write(' %0.03f ' % cm[1][1])
	arquivo.write('\n\n')
	
	print()

	class_names=[]
	class_names.append("Negativo")
	class_names.append("Positivo")
    
    
	# Plot non-normalized confusion matrix
	titles_options = [("Confusion matrix, without normalization", None),("Normalized confusion matrix", 'true')]

	for title, normalize in titles_options:
		plt.savefig('Matriz_conf.png')
		disp = ConfusionMatrixDisplay.from_estimator(DTC, X_test, y_test, display_labels=class_names,cmap=plt.cm.Blues,normalize=normalize)
		disp.ax_.set_title(title)



	print(title)
	print(disp.confusion_matrix)
	arquivo.write('Matriz de onfusao normalizada: \n')
	arquivo.write(' %0.03f ' % disp.confusion_matrix[0][0])
	arquivo.write(' %0.03f \n' % disp.confusion_matrix[0][1])
	arquivo.write(' %0.03f ' % disp.confusion_matrix[1][0])
	arquivo.write(' %0.03f ' % disp.confusion_matrix[1][1])
	
	plt.savefig('Matriz_conf_norm.png')
	
	# Compute ROC curve and ROC area for each class
	fpr, tpr, _  = roc_curve(y_test, y_pred)
	roc_auc = auc(fpr, tpr)

	# Compute micro-average ROC curve and ROC area
	# fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_test_pred[i].ravel())
	# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange',
                  lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()
	plt.savefig('curva_roc.png')
	explainer = shap.Explainer(DTC.predict, X_train)
	shap_values = explainer(X_train)
	# visualize the first prediction's explanation
	shap.plots.beeswarm(shap_values)
	plt.savefig('beeswarm.png')
	plt.figure()
	shap.plots.waterfall(shap_values[20])
	plt.savefig('waterfall.png')
	plt.figure()
	shap.plots.force(shap_values[20])
	plt.savefig('force.png')
	plt.figure()


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


#################################
data_sets = [(X_train, y_train),(X_test, y_test)]
name=['']
#modelosDT(data_sets, name=name)
modelosRF(data_sets, name=name)
#plt.show()
