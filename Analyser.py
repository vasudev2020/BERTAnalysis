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
import warnings
from warnings import filterwarnings
from sklearn.exceptions import ConvergenceWarning

import random

#from tqdm import tqdm
#from BERT import BERT,Glove

from Embedding import Embedding


class Analyser:   
    def __init__(self,num_folds,embtype='BERT',layer=11):
        self.num_folds = num_folds
        #self.bert = BERT()
        self.emb = Embedding(embtype,layer)
        #if embtype=='BERT': self.emb = BERT(layer)
        #if embtype=='Glove':    self.emb = Glove()
        self.Embs = {}
         
    def CreateFolds(self, Data):
        skf = StratifiedKFold(n_splits=self.num_folds)
        
        self.Folds = []
        assert sorted(Data['Dominant_Topic'].unique())==list(range(len(Data['Dominant_Topic'].unique())))
        for t in Data['Dominant_Topic'].unique():   self.Folds.append([])
        
        for t in Data['Dominant_Topic'].unique():
            seg = Data.loc[Data['Dominant_Topic'] == t]
            
            for train_index, test_index in skf.split(seg, seg['Labels']):
                self.Folds[int(t)].append(seg.iloc[test_index])
                
        
    def getEmb(self,sent):
        if sent not in self.Embs:   self.Embs[sent] = self.emb.getMean(sent)
        return self.Embs[sent]
                
    def perTopicAnalysis(self,Data):
        model = MLPClassifier()
        self.CreateFolds(Data)
                
        k = len(Data['Dominant_Topic'].unique())*5
        skf = StratifiedKFold(n_splits=k)
        mbs = 0.0
        fold_index_full = [test_index for _, test_index in skf.split(Data, Data['Labels'])]
        for _ in range(10):
            fold_index = random.sample(fold_index_full,self.num_folds)
            train_index = [x for f in fold_index[:self.num_folds-1] for x in f]
            test_index = fold_index[self.num_folds-1]
            base_trainX = [self.getEmb(sent) for sent in Data.iloc[train_index]['Sent']]
            base_trainY = [label for label in Data.iloc[train_index]['Labels']]
            base_testX = [self.getEmb(sent) for sent in Data.iloc[test_index]['Sent']]
            base_testY = [label for label in Data.iloc[test_index]['Labels']]
    
            #print(len(base_trainX),len(base_testX))
            #fs = int(mean([len(tg[fold]) for tg in self.Folds for fold in range(self.num_folds)]))
            #print(fs)
            with warnings.catch_warnings():
                filterwarnings("ignore", category=ConvergenceWarning)
                model.fit(base_trainX,base_trainY)
                pred = model.predict(base_testX)
                pred_proba = [proba[1] for proba in model.predict_proba(base_testX)]
                #base_Acc = accuracy_score(base_testY, pred)
                #base_F1 = f1_score(base_testY, pred, average='binary')
                base_Auc = roc_auc_score(base_testY,pred_proba)
            mbs += base_Auc
        mbs = mbs / 10

        
        '''
        fold_size = int(mean([len(tg[fold]) for tg in self.Folds for fold in range(self.num_folds)]))
        #print('Number of topics',len(Data['Dominant_Topic'].unique()))
        #print(fold_size,fold_size*self.num_folds)
        #base = Data['Sent'].sample(n=fold_size*self.num_folds)
        
        base = Data.sample(n=fold_size*self.num_folds)
        #print(len(base))
        skf = StratifiedKFold(n_splits=self.num_folds)
        base_Acc,base_F1,base_Auc = 0.0,0.0,0.0
        for train_index, test_index in skf.split(base, base['Labels']):
            #base_trainX = base.iloc[train_index]
            base_trainX = [self.getEmb(sent) for sent in base.iloc[train_index]['Sent']]
            base_trainY = [label for label in base.iloc[train_index]['Labels']]
            base_testX = [self.getEmb(sent) for sent in base.iloc[test_index]['Sent']]
            base_testY = [label for label in base.iloc[test_index]['Labels']]
            with warnings.catch_warnings():
                filterwarnings("ignore", category=ConvergenceWarning)
                model.fit(base_trainX,base_trainY)
                pred = model.predict(base_testX)
                pred_proba = [proba[1] for proba in model.predict_proba(base_testX)]
                base_Acc += accuracy_score(base_testY, pred)
                base_F1 += f1_score(base_testY, pred, average='binary')
                base_Auc += roc_auc_score(base_testY,pred_proba)
        mbs = base_Auc/self.num_folds
        '''
        Acc,F1,Auc=[],[],[]
        for tr_topic in range(len(self.Folds)):
            Acc.append([])
            F1.append([])
            Auc.append([])
            for ts_topic in range(len(self.Folds)):
                Acc[tr_topic].append(0)
                F1[tr_topic].append(0)
                Auc[tr_topic].append(0)
    
        for tr_topic in range(len(self.Folds)):
            ts_topic = tr_topic
    
            for test_fold in range(self.num_folds):
                #trainX = [self.bert.getMean(sent) for fold in range(5) for sent in self.Folds[tr_topic][fold]['Sent'] if fold!=test_fold]
                trainX = [self.getEmb(sent) for fold in range(self.num_folds) for sent in self.Folds[tr_topic][fold]['Sent'] if fold!=test_fold]
                trainY = [label for fold in range(self.num_folds) for label in self.Folds[tr_topic][fold]['Labels'] if fold!=test_fold]
                with warnings.catch_warnings():
                    filterwarnings("ignore", category=ConvergenceWarning)
                    model.fit(trainX,trainY)
    
                for ts_topic in range(len(self.Folds)):
                    #testX = [self.bert.getMean(sent) for sent in self.Folds[ts_topic][test_fold]['Sent']]
                    testX = [self.getEmb(sent) for sent in self.Folds[ts_topic][test_fold]['Sent']]
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
        seen_score = [Auc[t][t]/self.num_folds for t in range(len(self.Folds))]
        mean_unseen_score = [(sum(Auc[t]) - Auc[t][t]) / (self.num_folds * (len(self.Folds)-1)) for t in range(len(self.Folds))]
        #
        mss = mean(seen_score)
        mus = mean(mean_unseen_score)
        diff = mean([a-b for a,b in zip(seen_score,mean_unseen_score)])
        #print('mean seen score:',round(mean(seen_score),4))
        #print('mean unseen score:',round(mean(mean_unseen_score),4))
        #print('difference:',round(mean([a-b for a,b in zip(seen_score,mean_unseen_score)]),4))
        ##print(stats.ttest_ind(seen_score,mean_unseen_score))
        ttest = list(stats.ttest_rel(seen_score,mean_unseen_score))
        #print('T-Test Results:',round(ttest[0],4),round(ttest[1],4))
        
        return ([mss,mus,diff,mbs]+ttest)
        
        