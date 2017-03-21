# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 12:38:50 2017

@author: T366159
"""




'''                    ###  PROJET PREDICTION RETARDS DE VOLS  ###                      '''

''' ############################# IMPORTS #########################################                                                         #'''
import pylab as P                                                             #
import matplotlib.pyplot as plt
import pandas as pd                                                           #
import numpy as np  
from scipy.stats import skew, boxcox , probplot     #
###############################################################################



''' ############################################### ANNEXES #################################################
'''

def scatter_quanti(udata): 
    for i in udata.columns: 
        print(i)
        X=udata[i]
        if (X.dtypes.name in ['float64', 'int64' , 'int32'] and i!='RTD') : 
            plt.scatter(  X, udata['RTD'], lw=0, alpha=.08, color='k' , label='bonjouuuuuur' ) 
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

def categ_rtd(x):
    if x<=-3: # in ['0','M','U']:
        Y='A' 
    elif x in [-2, -1, 0, 1, 2]:
        Y='B'
    elif ((x>2) and (x<16)):  
        Y='C'
    else:
        Y='D'
    return Y      
    
    
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
  
def interr (data, liste, listeq):
    a=[]
    for i in data.columns: # listeVarbis :
        if i in liste: 
            print(i)
            try: 
                a=data[data[i] != '?']
                #print(pdata[pdata[i] != '?'])
            except TypeError:
                print('error')
            try: 
                a=a[a[i] != ' ']
                #print(pdata[pdata[i] != '?'])
            except TypeError:
                print('errorbis')    
            
            data=a
            if i in listeq: 
                data[i]=data[i].astype(np.integer)
            elif i=='hour': 
                print('none')
            else:
                data[i]=data[i].astype('category')
        else:
            del data[i]
    return (data)

def regularization(data,liste):
    for i in liste: 
        print(i)
        C=data[i].value_counts().index.unique()
        C=set(C)
        print (C)
        for j in C:
            if (isinstance( j, int )):
                print(j)
                #data.loc[data[i]==j,i]='%s' %(j) 
                u="'%s'" %(j)
                indexs=[data[i]==u]
               #u = urllib.quote("'" + j + "'")              
                print(u)
                for k in indexs:
                    data[k][i]= j
                    

           
'''   
               U=data[data[i]==j]
               U[i]='%s' %(j)
               data[data[i]==j]=U'''
    
def mean_cat(data,var, val=False):
    b=data.groupby([var]).mean()
    if val:
        print("give me some limit value!")
        value=input()
        #valueb=np.percentile(data.RTD,25)
    else:
        value=np.percentile(data.RTD,75)
        #valueb=np.percentile(data.RTD,25)
    a=b[(b.RTD>float(value))]               #  '''| (b.RTD<float(valueb))'''
    a["mod"]=a.index
    a["cnt"]=a['mod'].apply(lambda i:  len(data[data[var]==i].values))
    print (a[['RTD','mod','cnt']])
    plt.scatter(a.RTD,a.cnt)
    return a[['RTD','mod','cnt']]
    
def dispersion(data):
    for i in data.columns: 
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
         
              
        elif X.dtypes.name in ['float64', 'int64', 'int32'] : 
            print("next variable")
                
            #plt.xticks( np.arange(25) , rotation='vertical')
            plt.hist(X)#, bins=np.arange(25)
            plt.show()
                
            probplot(X, dist="norm", plot=P) 
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
    ylambda=0 
    print("Do you want to interact with this program?")
    ans='n' #   input()
    if 'y' in ans:
        interact=True
    else:
        interact=False
        
    print("Do you want to see output?")
    ans1='n'  #input()
    if 'y' in ans1:
        show=True
    else:
        show=False
                
    minimum=10
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
                probplot(data_to_clean[feature_names[i]], dist="norm", plot=P)
                P.show()   
            print(abs(skew(data_to_clean[feature_names[i]])))    
            if abs(skew(data_to_clean[feature_names[i]]))>1:
                o=skew(data_to_clean[feature_names[i]]) 
                print('Normality of the features is ' , o , " (mind that the best is 0)" )
                gg=min(data_to_clean[feature_names[i]])
    
                if gg>0:
                    gg=0
                else:
                    gg=abs(gg)+1                        
               
                if abs(skew(boxcox(data_to_clean[feature_names[i]]+gg)[0])) < abs(skew(np.sqrt(data_to_clean[feature_names[i]]+gg))):
                    data_to_clean[feature_names[i]] , lambdaa=boxcox(data_to_clean[feature_names[i]]+gg)
                    print("lambda =" , lambdaa, " for " , feature_names[i])
                    if feature_names[i]=='RTD':
                       # global ylambda
                        ylambda=lambdaa

                else:
                    data_to_clean[feature_names[i]]=np.sqrt(data_to_clean[feature_names[i]]+gg)
                    print("c'est sqrt!")
                print("variable ", feature_names[i],  " processed")
                data_to_clean[feature_names[i]].plot(kind="hist")
                plt.show()
                probplot(data_to_clean[feature_names[i]], dist="norm", plot=P)
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
            return (data_to_clean,ylambda)
            
def invboxcox(y,ld):
   if ld == 0:
      return(np.exp(y))
   else:
      return(np.exp(np.log(ld*y+1)/ld))
      
      
      
def normalization (feature_names, data_to_clean): 
     for i in range(len(feature_names)):
        print( "Etape N°: ",i+1," / " , len(feature_names))
        if (((data_to_clean[feature_names[i]].dtypes == 'int64') | (data_to_clean[feature_names[i]].dtypes == 'int32') | (data_to_clean[feature_names[i]].dtypes == 'float64')) & (feature_names[i] !='RTD')) :
            print(abs(skew(data_to_clean[feature_names[i]])))    
            
            if abs(skew(data_to_clean[feature_names[i]]))>1:
                o=skew(data_to_clean[feature_names[i]]) 
                print('Normality of the features is ' , o , " (mind that the best is 0)" )
                gg=min(data_to_clean[feature_names[i]])
    
                if gg>0:
                    gg=0
                else:
                    gg=abs(gg)+1                        
                print("gg = " , gg)
                if abs(skew(boxcox(data_to_clean[feature_names[i]]+gg)[0])) < abs(skew(np.sqrt(data_to_clean[feature_names[i]]+gg))):
                    data_to_clean[feature_names[i]] , lambdaa=boxcox(data_to_clean[feature_names[i]]+gg)
                    print("lambda =" , lambdaa, " for " , feature_names[i])

                else:
                    data_to_clean[feature_names[i]]=np.sqrt(data_to_clean[feature_names[i]]+gg)
                    print("c'est sqrt!")
                print("variable ", feature_names[i],  " processed")
                data_to_clean[feature_names[i]].plot(kind="hist")
                plt.show()
                probplot(data_to_clean[feature_names[i]], dist="norm", plot=P)
                P.show()      
     return data_to_clean 
     
'''     
def scatter_quanti(data): 
    for i in listeVarbis: 
        print(i)
        X=data[i]
        if (X.dtypes.name in ['float64', 'int64'] and i!='RTD') : 
            plt.scatter(  X, data['RTD'], lw=0, alpha=.08, color='k' , label='bonjouuuuuur' ) 
            plt.show() 
            ppp=input()       

 '''           
