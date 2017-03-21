# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 11:13:11 2017

@author: T366159
"""

'''                    ###  PROJET PREDICTION RETARDS DE VOLS  ###                      '''

''' ############################# IMPORTS #########################################                                                         #'''
import pylab as P                                                             #
import os           
                                                         #
import V2_annexes as A                                              #
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn.cluster import DBSCAN                                      
from sklearn.cluster import KMeans                                          #
#import scipy.stats as stats
from sklearn.cross_validation import train_test_split

from statsmodels.graphics.mosaicplot import mosaic   
    
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
#from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import roc_curve
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import skew, boxcox, normaltest, probplot 
from pandas.core.config import get_option    
import pickle   
import random  
import datetime      #
###############################################################################

listeVarbis=['usls', 'MDL' , 'dat', 'WGT' , 'CNY' , 'DEP' , 'ARV' , 'matric', 'CTN' , 'CCW' , 'TCW' , 'hour', 'PAX_OFF' ,
             'PAX_EBQ', 'BAG_EBQ' , 'LRG_CTC', 'STN', 'TER', 'LEG', 'PAX_CORR', 'SEJ_PRG', 'EQPMT' , 'CAT_AVI' , 'PRK' , 
             'DOU' , 'PRTY' , 'DOMINT', 'TYP_CR' , 'NBD' , 'NBA' , 'DIS' , 'TYP_CRR', 'REG_ACC_SRC' , 'RTD' , 'JDS', 'MTH']    

listeVar=[ 'MDL' ,  'WGT'  , 'DEP' , 'ARV' ,  'CTN' , 'CCW' , 'TCW' , 'hour', 'PAX_OFF' ,
             'PAX_EBQ', 'BAG_EBQ' , 'LRG_CTC', 'STN', 'TER',  'SEJ_PRG', 'EQPMT' , 'CAT_AVI' , 'PRK' , 
             'DOU' , 'PRTY' , 'DOMINT' , 'NBD' , 'NBA' , 'DIS' , 'TYP_CRR', 'TYP_CR' , 'RTD' , 'JDS', 'MTH']    

listeVarq=["WGT","PAX_EBQ", "PAX_OFF", 'SEJ_PRG', "BAG_EBQ","NBD","NBA", "DIS", "RTD"]
            
listeVarQ=['MDL' ,'LRG_CTC', 'STN', 'TER', "CCW","TCW" , 'DEP' , 'ARV' , 'CTN' , 'EQPMT' , 
           'CAT_AVI' , 'PRK' , 'DOU' , 'PRTY' , 'DOMINT' , 'JDS', 'MTH', 'TYP_CRR', 'TYP_CR' , 'hour']

''' ####################################### IMPORTATION DES DONNEES #################################### '''
             
pdata=pd.read_csv("C:/Users/t366159/Desktop/Rendu/input/data_0.txt", sep=";")

#Changement noms des variables                           
pdata.columns=listeVarbis   

#Configuration du type de chaque variable (quali ou quanti)               
pdata=interr(pdata, listeVar, listeVarq)   
                    
pdata.hour=pdata.hour.map(lambda x : 'aprem' if '17:00:00'>x>='14:00:00' else x)
pdata.hour=pdata.hour.map(lambda x : 'midi' if '14:00:00'>x>='12:00:00' else x)
pdata.hour=pdata.hour.map(lambda x : 'matinée' if '12:00:00'>x>='09:00:00' else x)
pdata.hour=pdata.hour.map(lambda x : 'matin' if '09:00:00'>x>='04:00:00' else x)
pdata.hour=pdata.hour.map(lambda x : 'nuit' if '04:00:00'>x>='00:00:00' else x)
pdata.hour=pdata.hour.map(lambda x : 'nuit' if '21:00:00'<=x<='23:59:59' else x)
pdata.hour=pdata.hour.map(lambda x : 'soiree' if '17:00:00'<=x<'21:00:00' else x)
#pdata.hour.value_counts()
pdata.hour=pdata.hour.astype('category')


for i in range(24):
    try: 
        pdata.CCW[pdata.CCW=='%s' %(i)]=i
    except(ValueError):
        print("error")
#pdata.CCW.value_counts()


                  
for i in range(1,4):
    pdata.TCW[pdata.TCW=='%s' %(i)]=i              
pdata.TCW.value_counts()
              
pdata.STN[pdata.STN=='2']=2  

    #regularization(pdata, ['TCW','CCW', 'STN' ]) 





'''################################################## NETTOYAGE ####################################### '''
print("NETTOYAGE") 
pdata=pdata[pdata.DIS > 50]
pdata=pdata[pdata.TCW != 1]
pdata=pdata[(pdata.CCW != 0)]
pdata=pdata[(pdata.CCW != 1)]
#pdata=pdata[((pdata.ARV != 'ORY') & (pdata.ARV!='CDG'))]
#pdata=pdata[pdata.DEP!=pdata.ARV]    
pdata=pdata[pdata.PAX_EBQ!=0]    # a prendre en compte????
pdata=pdata[pdata.BAG_EBQ!=0]


# vols considérés long courrier mais avec -1000 km 

 
# if bag_ebq = 0 and pax_ebq is > pax_off/3 then there is an error somewhere (and if the flight goes to another continent). remplace bag_ebq value by its mean in the very same leg
# VOIR AUSSI :pdata.PAX_OFF - pdata.PAX_EBQ
         
#Temporaire puisque normalement j'aurais bientot les bonnes valeurs du retard
#pdata=pdata[(pdata.RTD<300) & (pdata.RTD>=-5) ]
#pdata.RTD.hist()
             
#Creation de var qualitative retard ou pas             
pdata["Y"]=pdata["RTD"].map(lambda x: x > 0)

#pdata.Y.value_counts()

data=pdata.dropna()
data=data[data.RTD<60]  



#A.dispersion(data)


#dispersion(data)
#Regroupement de modalité  (We can run "dispersion" before)   
data['CCW']=data['CCW'].apply(A.categ_ccw)
data['EQPMT']=data['EQPMT'].apply(A.categ_eqpmt)
data['CAT_AVI']=data['CAT_AVI'].apply(A.categ_catavi)
data['DOU']=data['DOU'].apply(A.categ_dou)


#data['Y']=data['RTD'].apply(categ_rtd)

print("Cleaning")


############# CLEANING ###############
#MDL != E90 , ER4 , F50 , CRJ  
data=data[((data.MDL!='E90') &(data.MDL!='ER4') &(data.MDL!='F50') & (data.MDL!='CRJ'))]

#CCW : suppr modalités 17 18 & 19  (revoir pour 7 car mean:43 pr cnt 219)
data=data[(data.CCW!=17) & (data.CCW!=18) & (data.CCW!=19) & (data.CCW!=7)]

# STN : supr S, 1 & 3 ca ss repr 
data=data[(data.STN==2) | (data.STN=='W')]

# TER : WGEF only    -- SAME F*cking problem
#data=data[((data.TER=='S')  and (data.TER!='A') and (data.TER!=' '))]

# EQPMT : se va : - QA ; OE ; NI ; NG ;  -  NF ;  - NE ; MY ; MM ; - MF         
data=data[(data.EQPMT!='OE') &    (data.EQPMT!='NI') &    (data.EQPMT!='NG') &    (data.EQPMT!='MY') &    (data.EQPMT!='MM')]

#CAT_AVI : I & N BAD. 
data=data[(data.CAT_AVI!='I')&(data.CAT_AVI!='N')]

#ARV :           
'''ABJ     42.026101  ABJ    613
BGF     45.735178  BGF    253
BKO     30.528771  BKO    643
CUN     30.569343  CUN    137
DRS    140.000000  DRS      1
EBU    -11.000000  EBU      1
FNA     36.034884  FNA    344
GRQ     -2.000000  GRQ      1
GRX     -6.000000  GRX      1
LYN     75.000000  LYN      1
MXP     -1.000000  MXP      1
NSI     41.448276  NSI    261
OUA     30.411458  OUA    576
PAD     -5.000000  PAD      1
PSA     79.000000  PSA      1
SAW     43.000000  SAW      1

data=data[ (data.ARV != 'DRS') ]  
#data[(data['WGT']==1532) & (data['NBA']==219) & (data['NBD']==216)]

ss=data[data.ARV.map(lambda x: "of" if x.startswith("LYN") else x)=="of"]
ss
        hh=ss.map(lambda x: "of" if x.startswith("LY") else x)

hh.value_counts()




data[data.ARV == 'LYN'] 
'''




# Que pasa con los "Ficititious points AF" QZW/Y/X & ctn= zzz
#data[(data.CTN=='zzz')]
#data.CTN.value_counts()     


#DROP AIRPORT with only 1 flight history... not relevant 
a=data['ARV'].apply(lambda i:  len(data[data['ARV']==i].values))
a[a==1].index
#data=data.drop([109619, 138136, 211495, 211512, 211522, 211612, 250577, 353587,  353589, 353597, 353600, 353610, 353617, 353626, 353627, 353637, 353656])
data=data.drop(a[a==1].index) 

#PAX_CORR
#data.PAX_CORR.hist()

data["Vide"]=(data.PAX_OFF-data.PAX_EBQ)/data.PAX_OFF


############################################## DATA VIZ  ################################################
#CDG
#data_CDG=data[data.DEP.map(lambda x: "goal" if x.startswith("CDG") else x)=="goal"]

#ORY
data_ORY=data[data.DEP.map(lambda x: "goal" if x.startswith("ORY") else x)=="goal"]          

 
              
'''
data.TYP_CRR.value_counts()  

data_1=data[data.TYP_CRR=='1']
data_1.DIS.hist()

data_3=data[data.TYP_CRR=='3']
data_3.DIS.hist()

data_4=data[data.TYP_CRR=='4']
data_4.DIS.hist()

data_6=data[data.TYP_CRR=='6']
data_6.DIS.hist()

data_7=data[data.TYP_CRR=='7']
data_7.DIS.hist()'''


          
'''
dataq=data[data['DEP']=='ORY']
indexx=data['DEP'].apply(lambda i : str(i)=='ORY')


)=='ORY'

indexx.value_counts()


print("DATA VIZ")
A.mean_cat(data,'ARV', True)

# USEFULL PLOTS 
#print_distribution(data, 'hour', 'matin', 'JDS', 5)
#plt.scatter(  pdata.RTD, pdata.PAX_OFF - pdata.PAX_EBQ , lw=0, alpha=.08, color='k')
plt.scatter(  data.RTD, data.Vide , lw=0, alpha=.08, color='k')
plt.scatter(  data.RTD, data.BAG_EBQ , lw=0, alpha=.08, color='k')
plt.scatter(  data.RTD, data.SEJ_PRG, lw=0, alpha=.08, color='k')
plt.scatter(  dataq.RTD, dataq.NBA , lw=0, alpha=.08, color='k')
plt.scatter(  dataq.RTD, dataq.NBD, lw=0, alpha=.08, color='k')
plt.scatter(  data.RTD, data.DIS , lw=0, alpha=.08, color='k')
plt.scatter(  data.TCW, data.RTD , lw=0, alpha=.08, color='k')

A.scatter_quanti(data)

#determine if you have a linear correlation between multiple variables and the density of each 
scatter_matrix(pd.DataFrame(data_ory_n[[1,9,10,11,16,26,24,25,28,32]]), alpha=0.2, figsize=(15, 15), diagonal='kde')
plt.show()

#'''

''' ###################################### Echantillonage ##############################################'''
#data=A.normalization(listeVar, data)

#data_cdg_n=A.normalization(listeVar, data_CDG)
data_ory_n=A.normalization(listeVar, data_ORY)


# variables prédictives et cible
#pdata=data 
print("Echantillonage")
'''
y_ory=data_ORY['Y']
#y_cdg=data_CDG['Y']

## Extraction des variables dummies 
listeVarq=["WGT","PAX_EBQ", "PAX_OFF", 'PAX_CORR', 'SEJ_PRG', "BAG_EBQ","NBD","NBA", "DIS" , "Vide"]
#Dum_cdg=pd.get_dummies(data_CDG[listeVarQ])
Dum_ory=pd.get_dummies(data_ORY[listeVarQ])

#Quant_cdg=data_CDG[listeVarq]
Quant_ory=data_ORY[listeVarq]

#x_cdg=pd.concat([Dum_cdg,Quant_cdg],axis=1)
x_ory=pd.concat([Dum_ory,Quant_ory],axis=1)


#x_train,x_test,y_train,y_test=train_test_split(x_cdg,y_cdg,test_size=0.30,random_state=1871)
X_train,X_test,Y_train,Y_test=train_test_split(x_ory,y_ory,test_size=0.30,random_state=11)

'''
#var normalisées

y_oryn=data_ory_n['Y']
#y_cdgn=data_cdg_n['Y']

#Dum_cdgn=pd.get_dummies(data_cdg_n[listeVarQ])
Dum_oryn=pd.get_dummies(data_ory_n[listeVarQ])


listeVarq=["WGT","PAX_EBQ", "PAX_OFF", 'SEJ_PRG', "BAG_EBQ","NBD","NBA", "DIS", "Vide"]
#Quant_cdgn=data_cdg_n[listeVarq]
Quant_oryn=data_ory_n[listeVarq]

#x_cdgn=pd.concat([Dum_cdgn,Quant_cdgn],axis=1)
x_oryn=pd.concat([Dum_oryn,Quant_oryn],axis=1)


#x_train,x_test,y_train,y_test=train_test_split(x_cdgn,y_cdgn,test_size=0.30,random_state=71)
X_train,X_test,Y_train,Y_test=train_test_split(x_oryn,y_oryn,test_size=0.30,random_state=42)

Xr_train,Xr_test,Yr_train,Yr_test=train_test_split(x_oryn,data_ory_n['RTD'],test_size=0.30,random_state=19)


''' ###################################### EXPORT DATASETS INTO DIRECTORY ##############################################
'''

print("Export Data")

with open('C:/Users/t366159/Desktop/Rendu/Projet_DS3/echantillons/X_test.csv', 'wb') as output0:
    pickle.dump(X_test, output0, protocol=2)
with open('C:/Users/t366159/Desktop/Rendu/Projet_DS3/echantillons/X_train.csv', 'wb') as output1:
    pickle.dump(X_train, output1, protocol=2)    
with open('C:/Users/t366159/Desktop/Rendu/Projet_DS3/echantillons/Y_test.csv', 'wb') as output2:
    pickle.dump(Y_test, output2, protocol=2)
with open('C:/Users/t366159/Desktop/Rendu/Projet_DS3/echantillons/Y_train.csv', 'wb') as output3:
    pickle.dump(Y_train, output3, protocol=2)   
    
with open('C:/Users/t366159/Desktop/Rendu/Projet_DS3/echantillons/Xr_test.csv', 'wb') as output0:
    pickle.dump(Xr_test, output0, protocol=2)
with open('C:/Users/t366159/Desktop/Rendu/Projet_DS3/echantillons/Xr_train.csv', 'wb') as output1:
    pickle.dump(Xr_train, output1, protocol=2)    
with open('C:/Users/t366159/Desktop/Rendu/Projet_DS3/echantillons/Yr_test.csv', 'wb') as output2:
    pickle.dump(Yr_test, output2, protocol=2)
with open('C:/Users/t366159/Desktop/Rendu/Projet_DS3/echantillons/Yr_train.csv', 'wb') as output3:
    pickle.dump(Yr_train, output3, protocol=2)       
    
    
'''with open('C:/Users/t366159/Desktop/Rendu/Projet_DS3/echantillons/Xr_test.csv', 'wb') as routput0:
    pickle.dump(x_test, routput0, protocol=2)
with open('C:/Users/t366159/Desktop/Rendu/Projet_DS3/echantillons/Xr_train.csv', 'wb') as routput1:
    pickle.dump(x_train, routput1, protocol=2)    
with open('C:/Users/t366159/Desktop/Rendu/Projet_DS3/echantillons/Yr_test.csv', 'wb') as routput2:
    pickle.dump(y_test, routput2, protocol=2)
with open('C:/Users/t366159/Desktop/Rendu/Projet_DS3/echantillons/Yr_train.csv', 'wb') as routput3:
    pickle.dump(y_train, routput3, protocol=2)       
    
print(ylambda)    

def appren():
    return ylambda 
  
'''
