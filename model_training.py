#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 11:11:20 2018

@author: liuze
"""
import sys
from sklearn.externals import joblib
from feature_selection import *
from sklearn import preprocessing
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from numpy import  *
from sklearn.decomposition import PCA

def exit_with_help(argv):
	print("""\
Usage: python model_training.py training_positive_dataset training_negative_dataset model_file scale_file pca_file
This script was used to implement the m6ABRP tool.
Outputs:
     1--a model file, model.pkl, which can be directly used for prediction.
     2--a normalized file, normalization.pkl, which can be used to normalized the input data.
     3--a pca model file, pca.pkl, which can be used to generate the principal components.
""".format(argv[0]))
	exit(1)

def process_options(argv):
    argc=len(argv)
    if argc!=6:
        exit_with_help(argv)
    posCV=open(argv[1],'r')
    negCV=open(argv[2],'r')
    model=argv[3]
    scale=argv[4]
    pcafile=argv[5]
    return posCV, negCV, model, scale, pcafile
        
def main(argv=sys.argv):
    pos_train_file, neg_train_file, model_file, scale_file, pca_file=process_options(argv)
    feature_matrix=[]
    label_vector=[]
    #extract features of positive instances from fasta file
    thermo=zeros((16,125),dtype=float)
    thermofile=open('Thermo.txt','r')
    Thermo_row=0
    for line in thermofile:
        thermo_list=line.strip('\n').split()
        thermo[Thermo_row :]=thermo_list[0:125]
        Thermo_row+=1
    thermofile.close()
   
    
    for line in pos_train_file:
        feature_vector=[]
        sequence_infor=line.split()     
        sequence=sequence_infor[0]
        feature_vector.extend(kmer(sequence)[1]+ksnpf(sequence))
        thermo_result=np.dot(kmer(sequence)[0],thermo)
        #feature_vector.extend(thermo_result)
        label_vector.append('1')
        feature_matrix.append(feature_vector)
    pos_train_file.close()
    
    for line in neg_train_file:
        feature_vector=[]
        sequence_infor=line.split()
        sequence=sequence_infor[0]
        feature_vector.extend(kmer(sequence)[1]+ksnpf(sequence))
        thermo_result=np.dot(kmer(sequence)[0],thermo)
        #feature_vector.extend(thermo_result)
        label_vector.append('-1')
        feature_matrix.append(feature_vector)
    feature_array = np.array(feature_matrix,dtype=np.float32)
    min_max_scaler = preprocessing.MinMaxScaler(copy=True, feature_range=(-1, 1))
    feature_scaled= min_max_scaler.fit_transform(feature_array)
    neg_train_file.close()

    X=feature_scaled
    y=label_vector
    
    indices_path='./indices.txt'
    indices_file=open(indices_path,'r')
    indices=[]
    for line in indices_file:
        indices.append(int(line.split()[0]))
    indices_file.close()
    
    ranked_feature=list(indices)
    ks=X[:,ranked_feature[:215]]
    pca=PCA(n_components=58)
    pca.fit(ks)
    x_pca=pca.transform(ks)
    print("Original shape:{}".format(str(ks.shape)))
    print("Reduced shape:{}".format(str(x_pca.shape)))    
    
    
######SVMClassifier
    clf = SVC(C=6.75,gamma=0.02,probability=True)
    #clf = RandomForestClassifier(n_estimators=1000, max_depth=None,min_samples_split=2, random_state=1)
    #clf= GradientBoostingClassifier(n_estimators=1000, learning_rate=1.0,max_depth=1, random_state=0)    
    #clf = AdaBoostClassifier(n_estimators=1000)
    #clf = ExtraTreesClassifier(n_estimators=1000, max_depth=None,min_samples_split=2, random_state=1)    
    clf.fit(x_pca,y)
    joblib.dump(clf,model_file)
    joblib.dump(min_max_scaler,scale_file)
    joblib.dump(pca,pca_file)

if __name__=='__main__':
    main(sys.argv)