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
from numpy import  *
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA

def exit_with_help(argv):
	print("""\
Usage: python model_indepedent_testing.py test_positive_dataset test_negative_dataset model_file scale_file pca_file
Evaluate the performance of m6ABRP on the indepedent testing dataset.
The model_file, scale_file and pca_file generated in the training process must be involved.
""".format(argv[0]))
	exit(1)

def process_options(argv):
    argc=len(argv)
    if argc!=6:
        exit_with_help(argv)
    posCV=open(argv[1],'r')
    negCV=open(argv[2],'r')
    model_file=joblib.load(argv[3])
    scale_file=joblib.load(argv[4])
    pca_file=joblib.load(argv[5])
    return posCV, negCV, model_file, scale_file, pca_file
        
def main(argv=sys.argv):
    pos_test_file, neg_test_file, model_file, scale_file, pca_file=process_options(argv)
    feature_matrix=[]
    label_vector=[]
    thermo=zeros((16,125),dtype=float)
    thermofile=open('Thermo.txt','r')
    Thermo_row=0
    for line in thermofile:
        thermo_list=line.strip('\n').split()
        thermo[Thermo_row :]=thermo_list[0:125]
        Thermo_row+=1
    thermofile.close()
    
    for line in pos_test_file:
        feature_vector=[]
        sequence_infor=line.split()     
        sequence=sequence_infor[0]
        feature_vector.extend(kmer(sequence)[1]+ksnpf(sequence))
        thermo_result=np.dot(kmer(sequence)[0],thermo)
        #feature_vector.extend(thermo_result)
        label_vector.append('1')
        feature_matrix.append(feature_vector)
    pos_test_file.close()
    
    for line in neg_test_file:
        feature_vector=[]
        sequence_infor=line.split()
        sequence=sequence_infor[0]
        feature_vector.extend(kmer(sequence)[1]+ksnpf(sequence))
        thermo_result=np.dot(kmer(sequence)[0],thermo)
        #feature_vector.extend(thermo_result)
        label_vector.append('-1')
        feature_matrix.append(feature_vector)
    feature_array = np.array(feature_matrix,dtype=np.float32)
    #min_max_scaler = preprocessing.MinMaxScaler(copy=True, feature_range=(-1, 1))
    feature_scaled= scale_file.transform(feature_array)
    neg_test_file.close()

    X_test=feature_scaled
    y_test=label_vector
    
    indices_path='./indices.txt'
    indices_file=open(indices_path,'r')
    indices=[]
    for line in indices_file:
        indices.append(int(line.split()[0]))
    indices_file.close()
    
    ranked_feature=list(indices)
    ks=X_test[:,ranked_feature[:215]]
    x_pca=pca_file.transform(ks)
    print("Original shape:{}".format(str(ks.shape)))
    print("Reduced shape:{}".format(str(x_pca.shape)))      
    
######extraTreesClassifier
#    clf = ExtraTreesClassifier(n_estimators=1000, max_depth=None,min_samples_split=2, random_state=1)
#    clf.fit(X,y)
#    joblib.dump(clf,model_file)
#    joblib.dump(min_max_scaler,scale_file)
    predict_y_test = model_file.predict(x_pca)
    prob_predict_y_test = model_file.predict_proba(x_pca)
    TP=0
    TN=0
    FP=0
    FN=0 
    for i in range(0,len(y_test)):
        if int(y_test[i])==1 and int(predict_y_test[i])==1:
            TP=TP+1
        elif int(y_test[i])==1 and int(predict_y_test[i])==-1:
            FN=FN+1
        elif int(y_test[i])==-1 and int(predict_y_test[i])==-1:
            TN=TN+1
        elif int(y_test[i])==-1 and int(predict_y_test[i])==1:
            FP=FP+1
    Sn=float(TP)/(TP+FN)
    Sp=float(TN)/(TN+FP)
    ACC=float((TP+TN))/(TP+TN+FP+FN)
    print('m6ABRP Accuracy:%s'%ACC)
    print('m6ABRP Sensitive:%s'%Sn)
    print('m6ABRP Specificity:%s'%Sp)
    predictions_test = prob_predict_y_test[:, 1]
    y_validation=np.array(y_test,dtype=int)
    fpr, tpr, thresholds =metrics.roc_curve(y_validation, predictions_test,pos_label=1)
    roc_auc = auc(fpr, tpr)
        
    F1=metrics.f1_score(y_validation, map(int,predict_y_test))
    MCC=metrics.matthews_corrcoef(y_validation,map(int,predict_y_test))
    print('m6ABRP AUC:%s'%roc_auc)
    print('m6ABRP F1:%s'%F1)
    print('m6ABRP MCC:%s'%MCC)
    np.savetxt("m6ABRP_score.txt",predictions_test,fmt='%s',delimiter='\n')
    np.savetxt("test_label.txt",y_validation,fmt='%s',delimiter='\n')  

if __name__=='__main__':
    main(sys.argv)