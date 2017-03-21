# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 13:21:52 2017

@author: t366159
"""

'''                    ###  PROJET PREDICTION RETARDS DE VOLS  ###                      '''

''' ############################# IMPORTS #########################################                                                         #'''
import pylab as P                                                             #
import os                                                                     #
import pandas as pd                                                           #
import numpy as np                                                            #
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn.cluster import DBSCAN                                      
from sklearn.cluster import KMeans                                          #
#import scipy.stats as stats
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score          
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression       
from sklearn.tree import DecisionTreeClassifier
from statsmodels.graphics.mosaicplot import mosaic       
import math
#from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_curve
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import skew, boxcox, normaltest                 #
###############################################################################





''' ####################################### IMPORTATION DES DONNEES #################################### '''
 #D:/Users/oziyati/Desktop/MyDatab.csv
 pdata=pd.read_csv("C:/Users/t366159/Desktop/Rendu/input/MyDatab.csv", usecols=['MDL_AVI', 'WGT_AVB', 'FDR_OPS_FLI_CNY_COD', 'FDR_OPS_FLI_BRD' , 'FDR_OPS_FLI_OFF' , 'COD_CTN' , 'NUM_CCW', 'NUM_TCW', 'NUM_PAX_EXP_VER' ,'COD_ARV_DEP' , 'COD_EQP' , 'COD_CAT_AVI' , 'STT_PRK' , 'RGM_DOU_PRK' ,  'STT_VOL_PRI' , 'COD_DMQ_ITL','NBR_DEP_AER' , 'NBR_ARV_AER', 'DIS_TOT' , 'NBR_PAX' , 'Total_Minutes' , 'JDS' , 'LIB_CRT_MOI_ANG']).dropna()
 
 listeVarbis=['MDL' , 'WGT' , 'CNY' , 'DEP' , 'ARV' , 'CTN' , 'CCW' , 'TCW' , 'PAX_EXP' , 
'AD_COD' , 'EQPMT' , 'CAT_AVI' , 'PRK' , 'DOU' , 'PRTY' , 'DOMINT' , 'NBD' , 'NBA' , 'DIS' , 
'PAX' , 'RTD' , 'JDS', 'MTH']    

listeVarq=["WGT","CCW","TCW","PAX_EXP","NBD","NBA", "DIS","PAX","RTD"]
            
listeVarQ=['MDL' , 'CNY' , 'DEP' , 'ARV' , 'CTN' ,'AD_COD' , 'EQPMT' , 
                           'CAT_AVI' , 'PRK' , 'DOU' , 'PRTY' , 'DOMINT' , 'JDS', 'MTH']
#Changement noms des variables                           
pdata.columns=listeVarbis                    
                           
#Configuration du type de chaque variable (quali ou quanti)                            
pdata.JDS=pdata.JDS.astype('category')
pdata.MDL=pdata.MDL.astype('category')
pdata.CNY=pdata.CNY.astype('category')
pdata.DEP=pdata.DEP.astype('category')
pdata.ARV=pdata.ARV.astype('category')
pdata.CTN=pdata.CTN.astype('category')
pdata.AD_COD=pdata.AD_COD.astype('category')
pdata.EQPMT=pdata.EQPMT.astype('category')
pdata.CAT_AVI=pdata.CAT_AVI.astype('category')
pdata.PRK=pdata.PRK.astype('category')
pdata.DOU=pdata.DOU.astype('category')
pdata.PRTY=pdata.PRTY.astype('category')
pdata.DOMINT=pdata.DOMINT.astype('category')
pdata.MTH=pdata.MTH.astype('category')
pdata.CCW=pdata.CCW.astype('category')
pdata.TCW=pdata.TCW.astype('category')
pdata.PAX_EXP=pdata.PAX_EXP.astype(np.integer)
pdata.DIS=pdata.DIS.astype(np.integer)
pdata.PAX=pdata.PAX.astype(np.integer)
pdata.WGT=pdata.WGT.astype(np.integer)
pdata.RTD=pdata.RTD.astype(np.integer)

data.dtypes
'''################################################## NETTOYAGE ####################################### '''
pdata=pdata[pdata.WGT != '?']
pdata=pdata[pdata.RTD != '?']
pdata=pdata[pdata.DIS > 50]
pdata=pdata[pdata.TCW != 1]
#pdata=pdata[((pdata.CCW != 0) & (pdata.CCW != 1)]

#Temporaire puisque normalement j'aurais bientot les bonnes valeurs du retard
pdata=pdata[(pdata.RTD<300) & (pdata.RTD>=-5) ]

#Creation de var qualitative retard ou pas             
pdata["Y"]=pdata["RTD"].map(lambda x: x > 9)
pdata.Y.value_counts()
#W(pdata['RTD']>9).value_counts()
data=pdata 

#dispersion(data)

#Regroupement de modalité  (We can run "dispersion" before)
   
data['CCW']=data['CCW'].apply(categ_ccw)
data['EQPMT']=data['EQPMT'].apply(categ_eqpmt)
data['CAT_AVI']=data['CAT_AVI'].apply(categ_catavi)
data['DOU']=data['DOU'].apply(categ_dou)





'''############################################## DATA VIZ  ################################################'''

data=clean(listeVarbis, data)  #interactive program whose code is below (starting l.638), and which needs to be executed before this line


# USEFULL PLOTS 
print_distribution_ini(data, 'MDL', 320, 'JDS', 5)
plt.scatter(  data.RTD, data.WGT , lw=0, alpha=.08, color='k')
plt.scatter(  data.RTD, data.TCW , lw=0, alpha=.08, color='k')
plt.scatter(  data[data.DEP=='CDG'].RTD, data[data.DEP=='CDG'].NBD , lw=0, alpha=.08, color='k')   # combiné au nom de l'aeroport il devrait y avoir une mini relation
plt.scatter(  data[data.DEP=='CDG'].RTD, data[data.DEP=='CDG'].NBA , lw=0, alpha=.08, color='k')          #mêmes remarques 
plt.scatter(  data.RTD, data.DIS , lw=0, alpha=.08, color='k')
plt.scatter(  data.TCW, data.RTD , lw=0, alpha=.08, color='k')
scatter_quanti(data)

#determine if you have a linear correlation between multiple variables and the density of each 
scatter_matrix(pd.DataFrame(data[[1,9,16,17,18,19,20]]), alpha=0.2, figsize=(15, 15), diagonal='kde')
plt.show()






''' ###################################### Echantillonage ##############################################'''
# variables prédictives et cible
#pdata=data 

yr=edata[[20]]
yb=edata[[23]]
x=data[[1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22]]
x

x_train,x_test,y_train,y_test=train_test_split(x,yr,test_size=0.25,random_state=1871)
X_train,X_test,Y_train,Y_test=train_test_split(x,yb,test_size=0.25,random_state=11)


## Extraction des variables dummies 
listeVarq=['WGT',  'PAX_EXP', 'NBD', 'NBA', 'DIS', 'PAX']
listeVarQ=['MDL', 'CCW', 'TCW','CNY', 'DEP', 'ARV', 'CTN', 'AD_COD', 'EQPMT', 'CAT_AVI', 'PRK', 'DOU', 'PRTY', 'DOMINT', 'JDS', 'MTH']
Dum=pd.get_dummies(edata[listeVarQ])
#del ozoneDum["JOUR_0"]
Quant=edata[listeVarq]
x=pd.concat([Dum,Quant],axis=1)
x.head()








'''############################################ MACHINE LEARNING ################################################'''

''' REGRESSION LINEAIRE '''   #LASSO          # MSE=727.108226698  # R2=0.0205054211449
                                            # OPT : alpha=0.01   MSE= 673.115106425   R2 = 0.096275
                                            # also: alpha=0.001  MSE= 671.868747835   R2 = 0.099158
# grille de valeurs du paramètre alpha à optimiser
param=[{"alpha":[0.001, 0.005, 0.008, 0.01,0.015, 0.02, 0.05,0.1]}]
regLasso = GridSearchCV(linear_model.Lasso(), param,cv=5,n_jobs=-1)
regLassOpt=regLasso.fit(x_train, y_train)

prev=regLassOpt.predict(x_test)
# paramètre optimal
regLassOpt.best_params_["alpha"]
print("Meilleur R2 = %f, Meilleur paramètre = %s" % (regLassOpt.best_score_,regLassOpt.best_params_))
print("MSE=",mean_squared_error(prev,y_test))

# On applique maintenant ce param optimal sur notre apprentissage 
#regLasso = linear_model.Lasso(alpha=0.01) 
#regLasso.fit(x_train,y_train)
#prev=regLasso.predict(x_test)
#print("MSE=",mean_squared_error(y_test,prev))
#print("R2=",r2_score(y_test,prev))
#regLasso.fit(x_train,y_train).coef_
'''
# AFFICHAGE  
plt.plot(prev[:],y_test['RTD']-prev[:],"o")
plt.xlabel(u"Prédites")
plt.ylabel(u"Résidus")
plt.hlines(0,-20,85)
plt.show()

plt.plot(prev[:],y_test['RTD'],"o")
plt.xlabel(u"Prédites")
plt.ylabel(u"Observés")
e=np.linspace(0,100,101)
plt.plot(e,e,'k-')
#plt.dlines()
plt.show()

prediction = pd.DataFrame(prev).to_csv('prediction0.csv')
dprediction = pd.DataFrame(y_test).to_csv('test0.csv')
'''

'''#########################################################################################################

     REGRESSION LOGISTIQUE '''                 # Erreur sur ech de test 0.25882010954377166 (10) OPT

# Optimisation du paramètre de pénalisation
# grille de valeurs
param=[{"C":[7,9,10,11,12]}]
logit = GridSearchCV(LogisticRegression(penalty="l1"), param,cv=5,n_jobs=-1)
logitOpt=logit.fit(X_train, Y_train['Y'])  # GridSearchCV est lui même un estimateur

# paramètre optimal
logitOpt.best_params_["C"]
print("Meilleur score = %f, Meilleur paramètre = %s" % (1.-logitOpt.best_score_,logitOpt.best_params_))
# Coefficients
#LogisticRegression(penalty="l1",C=logitOpt.best_params_['C']).fit(X_train, Yb_train).coef_

logit = LogisticRegression(penalty="l1", C=10)
titan_logit=logit.fit(X_train, Y_train['Y'])
# Erreur sur l'écahntillon test
1-titan_logit.score(X_test, Y_test)

#:titan_logit.coef_

# Matrice de confusion 
y_chap = logitOpt.predict(X_test)
# matrice de confusion
table=pd.crosstab(y_chap,Y_test["domain"])
print(table)


    
'''#########################################################################################################
'''
'''  NEURAL NETWORK '''

#Meilleur score = 0.150792, Meilleur paramètre = {'hidden_layer_sizes': (8,)}

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


'''###########################################################################################################[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23]]
'''
''' ARBRES BINAIRES '''           # 0.32904439888620296 (defaut)       # 0.29561518925369479 (11)  => 0.15317768734126858
                                         #   0.29157614516079677 (13)  OPTIMAL       [[[ 0.23440225207306997 ]]] with rtd>20mins

'''
tree=DecisionTreeClassifier()
digit_tree=tree.fit(x_train, y_train) 
# Estimation de l'erreur de prévision
1-digit_tree.score(X_test,Y_test) 

param=[{"max_depth":list(range(10,20))}]
titan_tree= GridSearchCV(DecisionTreeClassifier(),param,cv=5,n_jobs=-1)
titan_opt=titan_tree.fit(X_train, Y_train["Y"])
# paramètre optimal
titan_opt.best_params_
'''

treee=DecisionTreeClassifier(criterion='gini', splitter='best', 
            max_depth=11, min_samples_split=2, min_samples_leaf=1, 
            min_weight_fraction_leaf=0.0, max_features=None, 
            random_state=None, max_leaf_nodes=None, class_weight=None, 
            presort=False)

titan_tree=treee.fit(X_train, Y_train)
# Estimation de l'erreur de prévision
# sur l'échantillon test
1-titan_tree.score(X_test,Y_test)

# prévision de l'échantillon test
z_chap = titan_tree.predict(X_test)
# matrice de confusion
table=pd.crosstab(Y_test['domain'],z_chap)
print(table)
plt.matshow(table)
plt.title("Matrice de Confusion")
plt.colorbar()
plt.show()

titan_tree.feature_importances_ 




'''#########################################################################################################
     RANDOM FOREST'''       # oob 0.158849478542    #  0.160736   # score prev : 5,22

from sklearn.ensemble import RandomForestClassifier 


# définition des paramètres
forest = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=None, min_samples_split=2, 
    min_samples_leaf=1, max_features='auto', max_leaf_nodes=None,bootstrap=True, oob_score=True)



# apprentissage
forest = forest.fit(X_train,Y_train['Y'])
print(1-forest.oob_score_)
# erreur de prévision sur le test
print("Meilleur score = %f" % (1.-forest.score(X_test,Y_test['Y'])))

# prévision
y_chap = forest.predict(X_test)
# matrice de confusion
table=pd.crosstab(y_chap,Y_test['Y'])
print(table)


importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]-22):
    print(x.columns[indices[f]], importances[indices[f]])
    
################

# optimisation de max_features
param=[{"max_features":list(range(6,15))}]
titan_rf= GridSearchCV(RandomForestClassifier(n_estimators=100),param,cv=5,n_jobs=-1)
titan_rfOpt=titan_rf.fit(X_train, Y_train['Y'])
# paramètre optimal
titan_rfOpt.best_params_

# erreur de prévision sur le test
1-titan_rfOpt.score(X_test,Y_test['Y'])

# prévision
z_chap = titan_rfOpt.predict(X_test)
# matrice de confusion
table=pd.crosstab(Y_test['Y'],z_chap)
print(table)
plt.matshow(table)
plt.title("Matrice de Confusion")
plt.colorbar()
plt.show()


# Coeff d'importance des variables 
rf= RandomForestClassifier(n_estimators=100,max_features=2)
rfFit=rf.fit(X_train, Yb_train)
# Importance décroissante des variables
importances = rfFit.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print(dfC.columns[indices[f]], importances[indices[f]])

# Graphe des importances
plt.figure()
plt.title("Importances des variables")
plt.bar(range(X_train.shape[1]), importances[indices])
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()

method=["RF",forest]
probas_ = method[1][1].fit(X_train, Y_train['Y']).predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(Y_test['Y'], probas_[:,1])
plt.plot(fpr, tpr, lw=1,label="%s"%method[1][0])

plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.legend(loc="lower right")
plt.show()

'''########################################################################################################
   GRADIENT BOOSTING '''             # 0.14927633793335582


from sklearn.ensemble import GradientBoostingClassifier


parametres = {"max_depth":[2,3,4,5,6,10],"n_estimators":[100,500,1000]}
gradBoost = GradientBoostingClassifier(learning_rate=0.01)
gradBoostOpt = GridSearchCV(gradBoost,param_grid=parametres,scoring="accuracy") 
gradBoostOptFit = gradBoostOpt.fit(X_train,Y_train['domain'])

1-gradBoostOptFit.score(X_test,Y_test['domain'])

# prévision de l'échantillon test
y_chap = gradBoostOptFit.predict(X_test)
# matrice de confusion
table=pd.crosstab(y_chap,Y_test['domain'])
print(table)


'''########################################################################################################
    K-Nearest Neighbors '''         # Param optim : 14: 0.15296349560906952, 18, 28 : 0.15046969186989378:5,64 

    
from sklearn.neighbors import KNeighborsClassifier

Xnet_train=X_train[listeVarq]
Xnet_test=X_test[listeVarq]

param=[{"n_neighbors":list(range(10,30))}]
knn=GridSearchCV(KNeighborsClassifier(),param,cv=5,n_jobs=4)
knnOpt=knn.fit(Xnet_train, Y_train['Y'])
# paramètre optimal
knnOpt.best_params_["n_neighbors"]    

nkk=knnOpt.best_params_["n_neighbors"]
n_proc=-1
knn = KNeighborsClassifier(n_neighbors=2, weights='uniform', 
        algorithm='auto', leaf_size=30, p=2, metric='minkowski', 
        metric_params=None, n_jobs=n_proc)
knnFit=knn.fit(Xnet_train, Y_train['Y']) 
# Estimation de l'erreur de prévision
# sur l'échantillon test
1-knnFit.score(Xnet_test,Y_test['domain'])

# prévision de l'échantillon test
y_chap = knnFit.predict(Xnet_test)
# matrice de confusion
table=pd.crosstab(y_chap,Y_test['domain'])
print(table)


''' ########################################################################################################
    LDA/QDA '''
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

qda =QuadraticDiscriminantAnalysis()
qda=qda.fit(Xnet_train, Y_train['Y'])
1-qda.score(Xnet_test,Y_test['domain'])

# prévision de l'échantillon test
y_chap = qda.predict(Xnet_test)
# matrice de confusion
table=pd.crosstab(y_chap,Y_test['domain'])
print(table)


'''########################################################################################################
      COURBES ROC '''

from sklearn.metrics import roc_curve
listMethod=[["K-nn",knnFit],["QDA", qda],["RF",forest],["NN",nnetOpt],["Tree",titan_tree],["Logit",titan_logit],["GB",gradBoostOptFit]]
#listMethod=[["NN",nnetOpt],["Tree",titan_tree],["Logit",titan_logit]]

for method in enumerate(listMethod):
    ki,kk=item
    if ki==0 | ki==1:
        
        probas_ = method[1][1].fit(Xnet_train, Y_train['Y']).predict_proba(Xnet_test)
        fpr, tpr, thresholds = roc_curve(Y_test['domain'], probas_[:,1])
        plt.plot(fpr, tpr, lw=1,label="%s"%method[1][0])

    else:        
        probas_ = method[1][1].fit(X_train, Y_train['Y']).predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(Y_test['domain'], probas_[:,1])
        plt.plot(fpr, tpr, lw=1,label="%s"%method[1][0])

plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.legend(loc="best")
plt.show()


'''
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    '''






''' ############################################### ANNEXES #################################################
'''

def scatter_quanti(data): 
    for i in listeVarbis: 
        print(i)
        X=data[i]
        if (X.dtypes.name in ['float64', 'int64'] and i!='RTD') : 
            plt.scatter(  X, data['RTD'], lw=0, alpha=.08, color='k' , label='bonjouuuuuur' ) 
            plt.show() 
            ppp=input()       

def categ_ccw(x):
    if x in [0.0 , 1.0]: #'clear sky' or x=='Sky is Clear':
        x='A'
    elif x in [2.0,3.0,4.0,5.0]: #=='broken clouds' or x=='few clouds' or x=='mist' or x=='light intensity drizzle' or x=='drizzle':
        x='B'
    elif x in [6.0,7.0,8.0,9.0]:  #=='light rain' or x=='scattered clouds' or x=='thunderstorm with light rain' or x=='thunderstorm' or x=='moderate rain' or x=='light intensity drizzle rain':
        x='C'
    elif x in [10.0 , 11.0 , 12.0, 13.0]:  #=='heavy rain' or x=='heavy intensity rain' or x=='very heavy rain' or x=='overcast clouds' or x=='thunderstorm with heavy rain'or x=='thunderstorm with rain':
        x='D'
    elif x in [14.0,15.0,16.0,17.0,18.0,19.0,20.0,22.0,23.0,24.0,25.0]:  #=='heavy rain' or x=='heavy intensity rain' or x=='very heavy rain' or x=='overcast clouds' or x=='thunderstorm with heavy rain'or x=='thunderstorm with rain':
        x='E'
    elif x in [21.0]:  #=='heavy rain' or x=='heavy intensity rain' or x=='very heavy rain' or x=='overcast clouds' or x=='thunderstorm with heavy rain'or x=='thunderstorm with rain':
        x='F'
    return x  
    
    
def categ_eqpmt(x):
    if  (x=='AB'or x=='AH' or x=='CA' or x=='CB'or x=='EC' or x=='MD' or x=='NC' or x=='NZ'):
        x='A'
    elif x in ['CC', 'CD', 'DA','DB','EB','DF','MA','NO']:
        x='B'
    elif x in ['AV']: 
        x='D'
    else:  
        x='C'
    return x  

def categ_catavi(x):
    if x in ['0','M','U']:
        x='A'
    elif x in ['A','B','Q','S']:
        x='B'
    else:  
        x='C'
    return x  

def categ_dou(x):
    if x=='M':
        x='I'
    return x  
    
    
def print_details(name_dtf,variable):
   print("Description of {0}".format(variable))
   print()
   print(name_dtf[variable].describe())
   print()
   print(name_dtf[variable].value_counts())
   print();
  

def dispersion(data):
    for i in listeVarbis: 
        print(i)
        X=data[i]
        if X.dtypes.name in ['category']:
            Y= data.groupby(i).mean().dropna()    
            print(Y['RTD'])
            label=Y['RTD'].index 
            
            plt.xticks( np.arange(2*len(label)) , label , rotation='vertical')
            #plt.yticks( np.arange(60) , [-50 , 0 , 50 ,100 ,150 , 200])
            plt.plot(Y.RTD.values, 'o-', color='r', lw=2, label='Retard Average' , alpha=.4)
            plt.show()
            ans=input()
            
        elif X.dtypes.name in ['float64', 'int64'] : 
            print("next variable")
                
            plt.xticks( np.arange(25) , rotation='vertical')
            plt.hist(data['CCW'], bins=np.arange(25))
                
            stats.probplot(data['CCW'], dist="norm", plot=P) 
            plt.show()
            ans=input()
            
 
def print_distribution(name_dtf,variable1 : str, value1, variable2 : str, value2) :
    #défine an error block in the case valux not in set(variablex)
    #get the indices of the two cases we want ot study
    answer1000=0
    dic1=create_index(name_dtf[variable1])
    dic2=create_index(name_dtf[variable2])
    #get the indices of the intersection of the two cases (so we can have the relevant set)
    #print(max(dic1.get(value1)),max(dic2.get(value2))
    
    indices=set(dic1[value1]).intersection(dic2[value2])
    #convert to list type in order to select automatically the right part of the dataframe
    indices=list(indices)
    #print (max(indices))
    #get the variable we want to study within the case set before by the arguments
    #define error in the case what the user enter is not a variable 
    #while answer1000!=1:
        #print("Give variable to print in case where {0}={1} and {2}={3}".format(variable1, value1, variable2, value2))
    #target=RTD   #input()
        #while target not in name_dtf.columns:
         #   target=input()
        
    print("Here is the histogram of {1} established on {0} values".format(len(indices),RTD))
    PLT=pd.DataFrame({RTD : name_dtf.iloc[indices][RTD]}, columns=[RTD])
    try:
            print(PLT.columns)
            #plt.subplots(4,2,figsize=(8,8))
            PLT[target].hist(alpha=0.5, bins=96)
            #plt.hist(PLT[target],alpha=0.5,bins=96, by=name_dtf.iloc[indices]['CARD_TYP_CD'])
            plt.xlabel('Time in seconds')
            plt.title("{0}={1} and {2}={3}".format(variable1, value1, variable2, value2))
            plt.legend()
            plt.show()
    except NameError:
            print("You probably did not compile the packages dude")
        #print("Do you want to print another variable for this case ?")
        #answer1000=1   #input()
        #if 'y' not in answer:
         #   break 
         


        
def clean(feature_names, data_to_clean):     # USES PRINT_DETAILS 
    print("Do you want to interact with this program?")
    ans=input()
    if 'y' in ans:
        interact=True
    else:
        interact=False
        
    print("Do you want to see output?")
    ans1=input()
    if 'y' in ans1:
        show=True
    else:
        show=False
                
    minimum=1000
    for i in range(len(feature_names)):
        print( "Etape N°: ",i+1," / " , len(feature_names))
#        try:
        if ((data_to_clean[feature_names[i]].dtypes == 'int64') | (data_to_clean[feature_names[i]].dtypes == 'int32') | (data_to_clean[feature_names[i]].dtypes == 'float64')):
            if (show):
                #if (interact): 
                    #print("c'est quantitatif!") 
                data_to_clean[feature_names[i]].plot(kind="box")
                plt.show()
                data_to_clean[feature_names[i]].plot(kind="hist")
                plt.show()
                print(data_to_clean[feature_names[i]].describe())
                stats.probplot(data_to_clean[feature_names[i]], dist="norm", plot=P)
                P.show()   
                
            if abs(skew(data_to_clean[feature_names[i]]))>1:
                o=skew(data_to_clean[feature_names[i]]) 
                print('Normality of the features is ' , o , " (mind that the best is 0)" )
                gg=min(data_to_clean[feature_names[i]])
    
                if gg>0:
                    gg=0
                else:
                    gg=abs(gg)+1                        
               
                if abs(skew(boxcox(data_to_clean[feature_names[i]]+gg)[0])) < abs(skew(np.sqrt(data_to_clean[feature_names[i]]+gg))):
                    data_to_clean[feature_names[i]]=boxcox(data_to_clean[feature_names[i]]+gg)[0]
                else:
                    data_to_clean[feature_names[i]]=np.sqrt(data_to_clean[feature_names[i]]+gg)
                print("variable ", feature_names[i],  " processed")
                data_to_clean[feature_names[i]].plot(kind="hist")
                plt.show()
                stats.probplot(data_to_clean[feature_names[i]], dist="norm", plot=P)
                P.show()
  
        else: 
            #
            kdata=data_to_clean
            X=kdata[feature_names[i]].value_counts()
            if (interact): 
                print("c'est qualitatif!")
                print_details(data_to_clean,feature_names[i])
            listeb=[]  

            for j in range(len(X)):

                if (X.values[j] < minimum):
                    listeb=[X.index[u] for u in range(j,len(X))]

                    
                    for f in listeb:
                        kdata=kdata[kdata[feature_names[i]]!=f] 
                    break
                
            if '?' in X.index:
                kdata=kdata[kdata[feature_names[i]]!='?']
                if interact:
                    print("no more interrogations!")
            if (interact) :
                print("you're okay getting rid of data under represented?")
                anssss=input()
                if 'y' in anssss: 
                    data_to_clean=kdata  
                    print("done")
            else:
                data_to_clean=kdata
                print("done")

                
            
            if show: 
                print(data_to_clean[feature_names[i]].value_counts())
                #print("Les variables vous semblent-elles liées?")
                #data.boxplot('RTD',feature_names[i])
            kdata=[]
            
            
#       except TypeError: 
        if (interact) & (i!=len(feature_names)-1):
            print("Do you want stop this?")
            answer=input()
            if 'y' in answer:
                return data_to_clean 
                break
            
        elif (i==len(feature_names)-1): 
            print("l'interface de préparation de données est terminée")
            return data_to_clean
            
            
            
            
            
''' Significativité décroissante des variables 
WGT 0.225774321911
NBD 0.139862167995
NBA 0.139042169094
PAX 0.0536657338299
PAX_EXP 0.0509014416516
DIS 0.0392911424055
JDS_4 0.0118978527244
JDS_1 0.011858309706
JDS_7 0.0109879129861
JDS_5 0.0109395223407
JDS_2 0.0108989988991
JDS_3 0.0108471046948
JDS_6 0.00960553425984
MTH_DECEMBER 0.00859115178476
MTH_OCTOBER 0.00854886769061
MTH_MAY 0.00817973985752
MTH_JUNE 0.00789448681216
MTH_JANUARY 0.00782143752897
MTH_APRIL 0.00765464896059
MTH_JULY 0.00759677484843
MTH_SEPTEMBER 0.00752592616757
MTH_AUGUST 0.00752261822287
MTH_NOVEMBER 0.00742093061358
MTH_FEBRUARY 0.00740931764498
MTH_MARCH 0.00737286123472
DOU_I 0.00470601973778
DOU_D 0.00469729251112
MDL_320 0.00458566993667
MDL_321 0.00447900874183
EQPMT_A 0.00441705954649
CTN_AFR 0.00381966655151
CTN_EUR 0.00381304151954
MDL_318 0.00376185487729
PRK_C 0.00374367664923
MDL_319 0.00357795220533
ARV_TLS 0.00356695012985
'''