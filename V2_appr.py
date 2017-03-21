# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 12:00:08 2017

@author: T366159
"""


'''                    ###  PROJET PREDICTION RETARDS DE VOLS  ###                      '''

''' ############################# IMPORTS #########################################                                                         #'''
import pylab as P                                                             #
import os                                                                     #
                         
#from pandas.tools.plotting import scatter_matrix
#from sklearn.cluster import DBSCAN                                      
#from sklearn.cluster import KMeans                                          #
#import scipy.stats as stats
#from sklearn.cross_validation import train_test_split
#from statsmodels.graphics.mosaicplot import mosaic   
    
import math
import pandas as pd                                                           #
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score          
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression       
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
#from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  
#from sklearn.metrics import roc_curve
#from scipy.cluster.hierarchy import dendrogram, linkage
#from scipy.stats import skew, boxcox, normaltest  
#from pandas.core.config import get_option    
import pickle           #
#from V2_annexes import invboxcox
#import exploration as exp

import imp
imp.find_module('numpy')



clf=DecisionTreeClassifier(max_depth=3)
clf=clf.fit(X_train,Y_train)
export_graphviz(clf,out_file='tree.dot')

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
import pydotplus
import pydotplus as pydot
treeG=DecisionTreeClassifier(max_depth=treeOpt.best_params_['max_depth'])
treeG.fit(X_train,Y_train)
dot_data = StringIO() 
export_graphviz(clf, out_file=dot_data) 
graph=pydotplus.graph_from_dot_data(dot_data.getvalue()) 
graph.write_png("treeOpt.png")
'''
###############################################################################

print("Recupération des données")

'''
'''
with open('C:/Users/t366159/Desktop/Rendu/Projet_DS3/echantillons/cdg_treated/Y_train.csv', 'rb') as rdata0:
    y_train = pickle.load(rdata0)
    
with open('C:/Users/t366159/Desktop/Rendu/Projet_DS3/echantillons/cdg_treated/Y_test.csv', 'rb') as rdata1:
    y_test = pickle.load(rdata1)
    
with open('C:/Users/t366159/Desktop/Rendu/Projet_DS3/echantillons/cdg_treated/X_train.csv', 'rb') as rdata2:
    x_train = pickle.load(rdata2)
   
with open('C:/Users/t366159/Desktop/Rendu/Projet_DS3/echantillons/cdg_treated/X_test.csv', 'rb') as rdata3:
    x_test = pickle.load(rdata3)
          
with open('C:/Users/t366159/Desktop/Rendu/Projet_DS3/echantillons/ory_treated/Y_train.csv', 'rb') as data0:
    Y_train = pickle.load(data0)
    
with open('C:/Users/t366159/Desktop/Rendu/Projet_DS3/echantillons/ory_treated/Y_test.csv', 'rb') as data1:
    Y_test = pickle.load(data1)
    
with open('C:/Users/t366159/Desktop/Rendu/Projet_DS3/echantillons/ory_treated/X_train.csv', 'rb') as data2:
    X_train = pickle.load(data2)
   
with open('C:/Users/t366159/Desktop/Rendu/Projet_DS3/echantillons/ory_treated/X_test.csv', 'rb') as data3:
    X_test = pickle.load(data3)
'''


with open('C:/Users/t366159/Desktop/Rendu/Projet_DS3/echantillons/ory_t&n_nc/Y_train.csv', 'rb') as data0:
    Y_train = pickle.load(data0)
    
with open('C:/Users/t366159/Desktop/Rendu/Projet_DS3/echantillons/ory_t&n_nc/Y_test.csv', 'rb') as data1:
    Y_test = pickle.load(data1)
    
with open('C:/Users/t366159/Desktop/Rendu/Projet_DS3/echantillons/ory_t&n_nc/X_train.csv', 'rb') as data2:
    X_train = pickle.load(data2)
   
with open('C:/Users/t366159/Desktop/Rendu/Projet_DS3/echantillons/ory_t&n_nc/X_test.csv', 'rb') as data3:
    X_test = pickle.load(data3)

    
#ylambda=exp.appren()    
  
''' 
'''#################################################### RANDOM FOREST ##################################################### '''
            # oob 0.158849478542    #  0.160736   # score prev : 5,22   # de grande valeurs de max features répondent mieux aux objs
# CLASSIFICATION #
print("Random Forest - Classification")
forest = RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=30, min_samples_split=2, min_samples_leaf=1,
                                max_features=5, max_leaf_nodes=None,bootstrap=True, oob_score=True)
titan_rfOpt=forest.fit(X_train, Y_train)

#X_test.head()

# optimisation de max_features
#param=[{"max_features":list(5,12,18,25)}]
#titan_rf= GridSearchCV(RandomForestClassifier(n_estimators=5000),param,cv=5,n_jobs=-1)
#titan_rfOpt=titan_rf.fit(X_train, Y_train)


# paramètre optimal
#params=titan_rfOpt.best_params_
# Coeff d'importance des variables 
#importances = titan_rfOpt.feature_importances_

# erreur de prévision sur le test
print("Meilleur score = %f" %(1-titan_rfOpt.score(X_test,Y_test)))
#print("OOB SCORE :  " , 1-titan_rfOpt.oob_score_)


# prévision
z_chap = titan_rfOpt.predict(X_test)
# matrice de confusion
table=pd.crosstab(Y_test,z_chap)
print(table)
#plt.matshow(table)
#plt.title("Matrice de Confusion")
#plt.colorbar()
#plt.show() '''
'''
print("Random Forest - Classification")
#forest = RandomForestClassifier(n_estimators=5000, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=5, max_leaf_nodes=None,bootstrap=True, oob_score=True)
#titan_rfOpt=forest.fit(X_train, Y_train)

# optimisation de max_features
param=[{"max_features":[5,12,18,25]}]
titan_rf= GridSearchCV(RandomForestClassifier(n_estimators=5000),param,cv=5)
titan_rfOpt=titan_rf.fit(x_train, y_train)


# paramètre optimal
params=titan_rfOpt.best_params_
# Coeff d'importance des variables 
#importances = titan_rfOpt.feature_importances_

# erreur de prévision sur le test
print("Meilleur score = %f" %(1-titan_rfOpt.score(x_test,y_test)))
#print("OOB SCORE :  " , 1-titan_rfOpt.oob_score_)


# prévision
z_chap = titan_rfOpt.predict(x_test)
# matrice de confusion
table=pd.crosstab(y_test,z_chap)
print(table)

print("Recupération des données")
with open('C:/Users/t366159/Desktop/Rendu/Projet_DS3/echantillons/cdg_t&n/Y_train.csv', 'rb') as rdata0:
    y_train = pickle.load(rdata0)
    
with open('C:/Users/t366159/Desktop/Rendu/Projet_DS3/echantillons/cdg_t&n/Y_test.csv', 'rb') as rdata1:
    y_test = pickle.load(rdata1)
    
with open('C:/Users/t366159/Desktop/Rendu/Projet_DS3/echantillons/cdg_t&n/X_train.csv', 'rb') as rdata2:
    x_train = pickle.load(rdata2)
   
with open('C:/Users/t366159/Desktop/Rendu/Projet_DS3/echantillons/cdg_t&n/X_test.csv', 'rb') as rdata3:
    x_test = pickle.load(rdata3)

print("Random Forest - Classification")
#forest = RandomForestClassifier(n_estimators=5000, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=5, max_leaf_nodes=None,bootstrap=True, oob_score=True)
#titan_rfOpt=forest.fit(X_train, Y_train)

# optimisation de max_features
param=[{"max_features":[5,12,18,25]}]
titan_rf= GridSearchCV(RandomForestClassifier(n_estimators=500),param,cv=5)
titan_rfOpt=titan_rf.fit(x_train, y_train)


# paramètre optimal
params=titan_rfOpt.best_params_
# Coeff d'importance des variables 
#importances = titan_rfOpt.feature_importances_

# erreur de prévision sur le test
print("Meilleur score = %f" %(1-titan_rfOpt.score(x_test,y_test)))
#print("OOB SCORE :  " , 1-titan_rfOpt.oob_score_)


# prévision
z_chap = titan_rfOpt.predict(x_test)
# matrice de confusion
table=pd.crosstab(y_test,z_chap)
print(table)    
    

'''    

with open('C:/Users/t366159/Desktop/Rendu/Projet_DS3/echantillons/ory_t&n_nc/Yr_train.csv', 'rb') as data0:
    Y_train = pickle.load(data0)
    
with open('C:/Users/t366159/Desktop/Rendu/Projet_DS3/echantillons/ory_t&n_nc/Yr_test.csv', 'rb') as data1:
    Y_test = pickle.load(data1)
    
with open('C:/Users/t366159/Desktop/Rendu/Projet_DS3/echantillons/ory_t&n_nc/Xr_train.csv', 'rb') as data2:
    X_train = pickle.load(data2)
   
with open('C:/Users/t366159/Desktop/Rendu/Projet_DS3/echantillons/ory_t&n_nc/Xr_test.csv', 'rb') as data3:
    X_test = pickle.load(data3)
    
# REGRESSION #
#, max_depth='none'
print("Random Forest - Regression")
rforest=RandomForestRegressor(n_estimators=1000, criterion='mse', min_samples_split=2, 
                            min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', 
                            max_leaf_nodes=None, bootstrap=True, random_state=None, verbose=0, warm_start=False)
rtitan_rfOpt=rforest.fit(X_train,Y_train)

# ♣ optimisation de max_features
#param=[{"max_features":list(range(10,15))}]
#rparam=[{"max_features":list(range(8,24,4))}]
#rtitan_rf= GridSearchCV(RandomForestRegressor(n_estimators=5),param,cv=10,n_jobs=-1)
#rtitan_rfOpt=rtitan_rf.fit(x_train,y_train['Y'])

# paramètre optimal
#print("Meilleur score = %f, Meilleur paramètre = %s" % (1. - rtitan_rfOpt.best_score_,rtitan_rfOpt.best_params_))
# erreur de prévision sur le test
print("Erreur de prévision = {0}".format(1-rtitan_rfOpt.score(X_test,Y_test)))
#print("OOB SCORE :  " , 1-rtitan_rfOpt.oob_score_)


# Coeff d'importance des variables 
rimportances = rtitan_rfOpt.feature_importances_


# prévision
z_chap = rtitan_rfOpt.predict(X_test)
#z_chap=invboxcox(z_chap,ylambda)
Z_chap=(z_chap>0)

yb_test=(Y_test > 0)

# matrice de confusion
rtable=pd.crosstab(yb_test,Z_chap)
print(rtable)





'''
indices = np.argsort(rimportances)[::-1]
for f in range(X_train.shape[1]-310):
    print(X_train.columns[indices[f]] , rimportances[indices[f]])
''' 
    
#↨print(rimportances)
'''
plt.matshow(rtable)
plt.title("Matrice de Confusion")
plt.colorbar()
plt.show()


'''

'''#########################################################################################################
'''
'''  NEURAL NETWORK

#Meilleur score = 0.150792, Meilleur paramètre = {'hidden_layer_sizes': (8,)}

from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  
# L'algorithme de N.NET nécessite éventuellement une normalisation des variables explicatives 
scaler = StandardScaler()  
scaler.fit(X_train)  
Xnet_train = scaler.transform(X_train)  
# Meme transformation sur le test
Xnet_test = scaler.transform(X_test)


# définition des paramètres. Nombre de neurones, alpha règle la régularisation par défaut 10-5 
# Le nombre de neurones est optimisé mais ce peut être alpha avec un nombre grand de neurones
# Le nombre max d'itérations est fixé à 500.
# Optimisation du nombre de neurones
param_grid=[{"hidden_layer_sizes":list([(5,),(6,),(7,),(8,)])}]
nnet= GridSearchCV(MLPClassifier(max_iter=500),param_grid,cv=10,n_jobs=-1)
nnetOpt=nnet.fit(X_train, Y_train['Y'])
# paramètre optimal
print("Meilleur score = %f, Meilleur paramètre = %s" % (1. - nnetOpt.best_score_,nnetOpt.best_params_))

# Estimation de l'erreur de prévision sur le test
1-nnetOpt.score(X_test,Y_test)

# prévision de l'échantillon test
y_chap = nnetOpt.predict(X_test)
# matrice de confusion
table=pd.crosstab(y_chap,Y_test['domain'])
print(table)
'''

'''######################################################################################################## '''
'''   GRADIENT BOOSTING             # 0.14927633793335582

print("Gradient Boosting - Classification")
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor


parametres = {"max_depth":[2,3,4,5,6,10],"n_estimators":[100,500,1000]}
parametres = {"max_depth":[2,3,4,5,6,10]}   #,"n_estimators":[100,500,1000]
gradBoost = GradientBoostingClassifier(learning_rate=0.01)
gradBoostOpt = GridSearchCV(gradBoost,param_grid=parametres,scoring="accuracy") 
gradBoostOptFit = gradBoostOpt.fit(X_train,Y_train)

1-gradBoostOptFit.score(X_test,Y_test)




# prévision
z_chap = rtitan_rfOpt.predict(X_test)
#z_chap=invboxcox(z_chap,ylambda)
Z_chap=(z_chap>0)

yb_test=(Y_test > 0)

# matrice de confusion
rtable=pd.crosstab(yb_test,Z_chap)
print(rtable)


# prévision de l'échantillon test
y_chap = gradBoostOptFit.predict(X_test)
# matrice de confusion
table=pd.crosstab(y_chap,Y_test['domain'])
print(table)




print("Gradient Boosting - Regression")
'''






'''
method=["RF",forest]
probas_ = method[1][1].fit(X_train, Y_train['Y']).predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(Y_test['Y'], probas_[:,1])
plt.plot(fpr, tpr, lw=1,label="%s"%method[1][0])

plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.legend(loc="lower right")
plt.show()
'''
