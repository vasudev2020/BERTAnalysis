#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 17:09:53 2021

@author: vasu
"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
import math
import warnings
from tqdm import tqdm

from BERT import BERT
bert = BERT()

def SplitToFolds(Data, num_folds):
    skf = StratifiedKFold(n_splits=num_folds)
    
    Folds = []
    for t in Data['Dominant_Topic'].unique():   Folds.append([])
    
    for t in Data['Dominant_Topic'].unique():
        seg = Data.loc[Data['Dominant_Topic'] == t]
        
        for train_index, test_index in skf.split(seg, seg['Labels']):
            Folds[int(t)].append(seg.iloc[test_index])
            
    return Folds


 
def Evaluate(y_test, predictions, pred_proba):
        matrix = confusion_matrix(y_test, predictions)
        tp = int(matrix[1][1])
        fn = int(matrix[1][0])
        fp = int(matrix[0][1])
        tn = int(matrix[0][0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            acc = accuracy_score(y_test, predictions)
            prec = precision_score(y_test, predictions, average='binary')
            rec = recall_score(y_test, predictions, average='binary')
            f1 = f1_score(y_test, predictions, average='binary')
            auct = roc_auc_score(y_test, predictions)
            pred_proba = [proba[1] for proba in pred_proba]
            auc = roc_auc_score(y_test,pred_proba)
            neg_prec = tn / (tn+fn+1e-16)
            neg_rec = tn / (tn+fp+1e-16)
            neg_f1 = 2*neg_prec*neg_rec / (neg_prec+neg_rec+1e-16)
            mcc =  float(tp*tn - fp*fn) / (math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))+1e-7)

        return acc, prec, rec, f1, neg_f1, auc, auct, mcc   

def perTopicIdiomDetection(Folds):
    model = MLPClassifier()
    Acc,F1,Auc=[],[],[]
    for tr_topic in range(len(Folds)):
        Acc.append([])
        F1.append([])
        Auc.append([])
        for ts_topic in range(len(Folds)):
            Acc[tr_topic].append(0)
            F1[tr_topic].append(0)
            Auc[tr_topic].append(0)

    for tr_topic in tqdm(range(len(Folds))):
        ts_topic = tr_topic

        for test_fold in range(5):
            trainX = [bert.getMean(sent) for fold in range(5) for sent in Folds[tr_topic][fold]['Sent'] if fold!=test_fold]
            trainY = [label for fold in range(5) for label in Folds[tr_topic][fold]['Labels'] if fold!=test_fold]
            model.fit(trainX,trainY)

            for ts_topic in range(len(Folds)):
                testX = [bert.getMean(sent) for sent in Folds[ts_topic][test_fold]['Sent']]
                testY = [label for label in Folds[ts_topic][test_fold]['Labels']]
                pred = model.predict(testX)
                pred_proba = model.predict_proba(testX)
        
                acc,_,_,f1,_,auc,_,_ = Evaluate(testY,pred,pred_proba)
                Acc[tr_topic][ts_topic]+=acc
                F1[tr_topic][ts_topic]+=f1
                Auc[tr_topic][ts_topic]+=auc
     
    print('Accuracies:')           
    for tr_topic in range(len(Folds)):
        print(' '.join([str(round(score/5,4)) for score in Acc[tr_topic]]))
    print('F1-scores:')           
    for tr_topic in range(len(Folds)):    
        print(' '.join([str(round(score/5,4)) for score in F1[tr_topic]]))
    print('Auc scores:')           
    for tr_topic in range(len(Folds)):
        print(' '.join([str(round(score/5,4)) for score in Auc[tr_topic]]))
                  
