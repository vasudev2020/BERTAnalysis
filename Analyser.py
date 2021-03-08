#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 17:09:53 2021

@author: vasu
"""
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,roc_auc_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from statistics import mean
from scipy import stats
#import math
#import warnings
from tqdm import tqdm

from BERT import BERT


class Analyser:   
    def __init__(self,num_folds):
        self.num_folds = num_folds
        self.bert = BERT()
    
         
    def CreateFolds(self, Data):
        skf = StratifiedKFold(n_splits=self.num_folds)
        
        self.Folds = []
        assert sorted(Data['Dominant_Topic'].unique())==list(range(len(Data['Dominant_Topic'].unique())))
        for t in Data['Dominant_Topic'].unique():   self.Folds.append([])
        
        for t in Data['Dominant_Topic'].unique():
            seg = Data.loc[Data['Dominant_Topic'] == t]
            
            for train_index, test_index in skf.split(seg, seg['Labels']):
                self.Folds[int(t)].append(seg.iloc[test_index])
                
    def perTopicAnalysis(self,Data):
        model = MLPClassifier()
        self.CreateFolds(Data)
        
        #TODO: print warning if labels<5
        
        Acc,F1,Auc=[],[],[]
        for tr_topic in range(len(self.Folds)):
            Acc.append([])
            F1.append([])
            Auc.append([])
            for ts_topic in range(len(self.Folds)):
                Acc[tr_topic].append(0)
                F1[tr_topic].append(0)
                Auc[tr_topic].append(0)
    
        for tr_topic in tqdm(range(len(self.Folds))):
            ts_topic = tr_topic
    
            for test_fold in range(5):
                trainX = [self.bert.getMean(sent) for fold in range(5) for sent in self.Folds[tr_topic][fold]['Sent'] if fold!=test_fold]
                trainY = [label for fold in range(5) for label in self.Folds[tr_topic][fold]['Labels'] if fold!=test_fold]
                model.fit(trainX,trainY)
    
                for ts_topic in range(len(self.Folds)):
                    testX = [self.bert.getMean(sent) for sent in self.Folds[ts_topic][test_fold]['Sent']]
                    testY = [label for label in self.Folds[ts_topic][test_fold]['Labels']]
                    
                    if sum(testY)==0 or sum(testY)==len(testY): 
                        print(ts_topic,test_fold, sum(testY))
                        print(Data.groupby(['Dominant_Topic','Labels']).size()[ts_topic])
                    
                    pred = model.predict(testX)
                    pred_proba = [proba[1] for proba in model.predict_proba(testX)]
            
                    Acc[tr_topic][ts_topic] += accuracy_score(testY, pred)
                    F1[tr_topic][ts_topic] += f1_score(testY, pred, average='binary')
                    Auc[tr_topic][ts_topic] += roc_auc_score(testY,pred_proba)
         
        '''
        print('Accuracies:')           
        for tr_topic in range(len(self.Folds)):
            print(' '.join([str(round(score/5,4)) for score in Acc[tr_topic]]))
        print(' '.join([str(round(Acc[topic][topic]/5,4)) for topic in range(len(self.Folds))]))

        print('F1-scores:')           
        for tr_topic in range(len(self.Folds)):    
            print(' '.join([str(round(score/5,4)) for score in F1[tr_topic]]))
        print(' '.join([str(round(F1[topic][topic]/5,4)) for topic in range(len(self.Folds))]))

        
        print('Auc scores:')           
        for tr_topic in range(len(self.Folds)):
            print(' '.join([str(round(score/5,4)) for score in Auc[tr_topic]]))
        print(' '.join([str(round(Auc[topic][topic]/5,4)) for topic in range(len(self.Folds))]))
        
        diff = mean([(Auc[t][t]/5 - ((sum(Auc[t])-Auc[t][t]) / (5*(len(self.Folds)-1)))) for t in range(len(self.Folds))])
        print('diff:',diff)

        '''
        seen_score = [Auc[t][t]/5 for t in range(len(self.Folds))]
        mean_unseen_score = [(sum(Auc[t]) - Auc[t][t]) / (5 * (len(self.Folds)-1)) for t in range(len(self.Folds))]
        #
        print('mean seen score:',round(mean(seen_score),4))
        print('mean unseen score:',round(mean(mean_unseen_score),4))
        print('difference:',round(mean([a-b for a,b in zip(seen_score,mean_unseen_score)]),4))
        #print(stats.ttest_ind(seen_score,mean_unseen_score))
        ttest = list(stats.ttest_rel(seen_score,mean_unseen_score))
        print('T-Test Results:',round(ttest[0],4),round(ttest[1],4))
        
        