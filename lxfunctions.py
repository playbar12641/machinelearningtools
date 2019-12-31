# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd                       # DataFrames for tabular data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 

#import seaborn as sns
def principle_component_analysis(df, p):
    # this function perform principle component analysis on the input data frame
    # df is the input dataframe 
    # p is the number of priciple components
    #1. standardized the input data
    scaler = StandardScaler().fit(df)  # scaler object for the original data
    df_st = scaler.transform(df)   # transform the data
     
    n_features = len(df_st[0]) # columns number is the number of features
    #array_features = str(np.arange(n_features)+1)
        
    #print(n_features)
     # set hyper parameters of principle component
    pca = PCA(n_components=n_features).fit(df_st) # extensiate the object    

    # calculate priciple component scores
    pc_score = pca.transform(df_st) # calculate principle components scores with the created method
    
    # generate name for the component score: PC#1, PC#2.....PC#n
    pc_feature = ['PC#' + x for x in (np.arange(n_features)+1).astype(str)]
    df_pc_score = pd.DataFrame(data=pc_score, columns=pc_feature)  # convert PC score array to data frame
    
    # reconstruct data by reverse transform the component score and loading
    data_it = scaler.inverse_transform(pca.transform(df_st)[:,:p] @ pca.components_[:p,:])
    df_rc = pd.DataFrame(data=data_it, columns=df.columns)  # dimensionality reduced dataframe
    
    
#    fh = 1 # figure height
#    sns.set(style='ticks')
#    h = sns.pairplot(df_rc, plot_kws={'alpha':0.6},height=fh,aspect=1)
#    ts1 = 'reconstruction with ' + str(p) + ' principle component(s)'
#    h.fig.suptitle(ts1, y=1)

    
    return {'df_rc': df_rc, 'df_pcs': df_pc_score,'pca_result': pca ,'scaler': scaler}


#import os    # set working directory, run executables
#import copy
##import lxfunctions as lxf
#path=os.path.expanduser("~\\Box Sync\\2019\\PGE 383 Subsurface Maching Learning\\Homework\\HW3")  
## get the relative path to the user directory
#os.chdir(path) # set the working directory
#
#df = pd.read_csv('unconv_MV_v5.csv')   # load our data table 
#df_backup = copy.deepcopy(df)          # make a backup copy of the input data
#df = df.drop(columns = ['Well','VR','TOC','Prod'])    # drop the well number,
#df.describe().transpose()      