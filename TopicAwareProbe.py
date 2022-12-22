#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 09:21:15 2022

@author: vasu
"""
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from scipy import stats
from statistics import mean

from collections import defaultdict
from tqdm import tqdm

import warnings
from warnings import filterwarnings
from sklearn.exceptions import ConvergenceWarning

import argparse, time, pickle
from datetime import timedelta

from Embedding import Embedding
from TopicModel import TopicModel

class TopicAwareProbe:
    def __init__(self,emb_types, mstart=5, mstop=51, mstep=5, train0labels=False, num_folds=5, allembs=False):
        self.TM = TopicModel()
        self.emb_types = emb_types
        self.mstart = mstart
        self.mstop = mstop
        self.mstep = mstep
        self.train0labels = train0labels
        
        self.num_folds = num_folds
        self.emb = Embedding(allembs)        
        self.Embs = {}
        
    def getTail(self,Topics,Labels):
        ''' Get the index of a tail topic if any. Input is a list of topics and number of labels'''
        
        for i in range(len(Topics)):
            if len(Topics[i].groupby(['Labels']).size())!=Labels:    return i
            if min([Topics[i].groupby(['Labels']).size()[l] for l in range(Labels)])<5:   return i
        return None
            
    def tailReduction(self,Topics,Labels):
        '''Iteratively reduce all tail topics by meging with other topics.
            If there are two tail topics merge it. 
            If there is only one tail topic merge it with last topic'''
            
        DomTopics = list(Topics['Dominant_Topic'].unique())
        Topics = [Topics.loc[(Topics['Dominant_Topic']==dt)].reset_index() for dt in DomTopics]
        #print('Number of initial topics:',len(self.Topics))  
        
        while True:
            tail_index_1 = self.getTail(Topics,Labels)
            if tail_index_1 is None:   break
            tail_1 = Topics[tail_index_1]
            del Topics[tail_index_1]
            
            tail_index_2 = self.getTail(Topics,Labels)
            if tail_index_2 is None:   
                Topics[-1] = pd.concat([Topics[-1],tail_1],ignore_index=True)
                break
            tail_2 = Topics[tail_index_2]
            del Topics[tail_index_2]

            Topics.append(pd.concat([tail_1,tail_2],ignore_index=True))
        
        #print('Number of topics after tail reduction:',len(self.Topics))
        
        for i in range(len(Topics)):
            Topics[i]=Topics[i].rename(columns={'Dominant_Topic':'Old_Topic'})
            Topics[i].insert(0,'Dominant_Topic',[i]*len(Topics[i].index))
            
        return pd.concat(Topics,ignore_index=True)
    
    def entropyAnalysis(self,data,labels,expressions):
        LabelEntropy = defaultdict(list)
        #ExpressionEntropy = defaultdict(list)

        for mp in tqdm(range(self.mstart,self.mstop,self.mstep)): 
            if self.train0labels:    self.TM.train([d for d,l in zip(data,labels) if l==0],mp)
            else:   self.TM.train(data,mp)
            
            Topics = self.TM.topicModel(data,expressions,labels)
            Topics = Topics.loc[Topics['Dominant_Topic']!=-1]
            
            max_entropy = stats.entropy([1]*len(Topics['Dominant_Topic'].unique()))
            #print('Maximum entropy', max_entropy)
            
            tg = Topics.groupby(['Labels','Dominant_Topic']).size()
            for l in Topics['Labels'].unique():
                LabelEntropy[l].append(stats.entropy(tg[l])/max_entropy)
            #LabelEntropy = [stats.entropy(tg[l])/max_entropy for l in Topics['Labels'].unique()]
            '''
            tg = Topics.groupby(['Expressions','Dominant_Topic']).size()
            for e in Topics['Expressions'].unique():
                Ze = len(Topics.loc[Topics['Expressions']==e])
                Zt = len(Topics['Dominant_Topic'].unique())
                
                Z = stats.entropy([1]*min(Zt,Ze))
                h = stats.entropy(tg[e])/Z
                assert h<=1.0 and h>=0
                ExpressionEntropy[e].append(h)
            '''
                
            #TODO: keywords of topics
             
        #print(LabelEntropy)
        
        for l in LabelEntropy:  
            print('Label:',l,':',mean(LabelEntropy[l]),min(LabelEntropy[l]),max(LabelEntropy[l]))
            #print(LabelEntropy[l])
        #for e in ExpressionEntropy:  print('Expression:',e,':',mean(ExpressionEntropy[e]))
    
    def probe(self,data,labels,expressions):
        '''Do a topic aware probing and print all the results. Input: 
                data: a list of sentences
                labels: a list of labels
                expressions: a list of expressions/type of input''' 
                
        n_class = len(set(labels))
        self.multiclass = False if n_class==2 else True
        
        Scores = {}
        for mp in tqdm(range(self.mstart,self.mstop,self.mstep)):  
            if self.train0labels:    self.TM.train([d for d,l in zip(data,labels) if l==0],mp)
            else:   self.TM.train(data,mp)
            Topics = self.TM.topicModel(data,expressions,labels)
            Topics = Topics.loc[Topics['Dominant_Topic']!=-1]
            
            Labels = len(Topics['Labels'].unique())
        
            '''
            gld = [Topics.groupby(['Labels']).size()[l] for l in range(Labels)]
            gld = [float(e)/sum(gld) for e in gld]
            print('global label distribution:',gld)
            '''

            Topics = self.tailReduction(Topics,Labels)
                        
            for embtype in self.emb_types:
                seen_score,unseen_score = self.getSeenUnseenScores(Topics,embtype)
                if embtype+'_seen' not in Scores: Scores[embtype+'_seen']=[]
                Scores[embtype+'_seen'].append(seen_score)
                if embtype+'_unseen' not in Scores: Scores[embtype+'_unseen']=[]
                Scores[embtype+'_unseen'].append(unseen_score)
                
        print('{0:>8} {1:>8} {2:>8} {3:>8} {4:>8}'.format('Emb','Seen','Unseen','Diff','p-value'))
        for embtype in self.emb_types:
            mean_seen = mean(Scores[embtype+'_seen'])
            mean_unseen = mean(Scores[embtype+'_unseen'])
            diff = [a-b for a,b in zip(Scores[embtype+'_seen'],Scores[embtype+'_unseen'])]
            mean_diff = mean(diff)
            t,p = list(stats.ttest_rel(diff,[0]*len(diff),alternative='greater'))
            #print(embtype,mean_seen,mean_unseen,mean_diff,p)
            print('{0:>8} {1:>8} {2:>8} {3:>8} {4:>8}'.format(embtype,round(mean_seen,4),round(mean_unseen,4),round(mean_diff,4),p))
        
        
    def getSeenUnseenScores(self,Data,embtype):
        '''Get the  seen score and the unseen score
        Input: topic wise partitioned data and the embedding type'''
        
        model = MLPClassifier()
        self.createFolds(Data)
        topic_wise_seen_scores,topic_wise_unseen_scores = [],[]
        for tr_topic in range(len(self.Folds)):
            seen_crossval, unseen_crossval = [],[]
            for test_fold in range(self.num_folds):
                trainX = [self.getEmb(sent,embtype) for fold in range(self.num_folds) for sent in self.Folds[tr_topic][fold]['Sent'] if fold!=test_fold]
                trainY = [label for fold in range(self.num_folds) for label in self.Folds[tr_topic][fold]['Labels'] if fold!=test_fold]
                with warnings.catch_warnings():
                    filterwarnings("ignore", category=ConvergenceWarning)
                    model.fit(trainX,trainY)
                    
                seenX = [self.getEmb(sent,embtype) for sent in self.Folds[tr_topic][test_fold]['Sent']]
                seenY = [label for label in self.Folds[tr_topic][test_fold]['Labels']]
                unseenX = [self.getEmb(sent,embtype) for topic in range(len(self.Folds)) for sent in self.Folds[topic][test_fold]['Sent'] if topic!=tr_topic]
                unseenY = [label for topic in range(len(self.Folds)) for label in self.Folds[topic][test_fold]['Labels'] if topic!=tr_topic]
                
                if self.multiclass:
                    seen_crossval.append(roc_auc_score(seenY,model.predict_proba(seenX),multi_class='ovr'))
                    unseen_crossval.append(roc_auc_score(unseenY,model.predict_proba(unseenX),multi_class='ovr'))

                else:
                    seen_crossval.append(roc_auc_score(seenY,[proba[1] for proba in model.predict_proba(seenX)]))
                    unseen_crossval.append(roc_auc_score(unseenY,[proba[1] for proba in model.predict_proba(unseenX)]))
            topic_wise_seen_scores.append(mean(seen_crossval))
            topic_wise_unseen_scores.append(mean(unseen_crossval))
            
        return [mean(topic_wise_seen_scores),mean(topic_wise_unseen_scores)]

    def createFolds(self, Data):
        '''Split each of the part in the partitioned data into different folds for the cross validation
            This will create a table 'Folds' with size (no. topics, no_folds), as a list of lists. '''
            
        skf = StratifiedKFold(n_splits=self.num_folds)
        
        self.Folds = []
        assert sorted(Data['Dominant_Topic'].unique())==list(range(len(Data['Dominant_Topic'].unique())))
        for t in Data['Dominant_Topic'].unique():   self.Folds.append([])
        
        for t in Data['Dominant_Topic'].unique():
            seg = Data.loc[Data['Dominant_Topic'] == t]
            
            for train_index, test_index in skf.split(seg, seg['Labels']):
                self.Folds[int(t)].append(seg.iloc[test_index])
                        
    def getEmb(self,sent,embtype):
        ''' This will return the 'embtype' embedding of input sentence 'sent' '''
        
        if sent not in self.Embs:   self.Embs[sent] = self.emb.getMean(sent)
        return self.Embs[sent][embtype]
        
                      
#def BERTTopicAnalysis(task, mstart, mstop, mstep, size, alllayers, train0labels=False):
    
def stratified_sample_df(df, col, n_samples):
    n = min(n_samples, df[col].value_counts().min())
    df_ = df.groupby(col).apply(lambda x: x.sample(n))
    df_.index = df_.index.droplevel(0)
    return df_

def LoadVNC(size):
    dataset = pickle.load(open("../Data/ID/vnics_dataset_full_ratio-split.pkl", "rb"), encoding='latin1')          
    data = [p['sent'].replace(' &apos;','\'') for p in dataset["train_sample"]+dataset["test_sample"]]
    labels = [p['lab_int'] for p in dataset["train_sample"]+dataset["test_sample"]]
    expressions = [p['verb']+' '+p['noun'] for p in dataset["train_sample"]+dataset["test_sample"]]
    print('Loaded VNC dataset with')
    print('No. of samples:',len(data)) 
    print('No. of idioms:',sum(labels))
    print('No. of literals:',len(labels)-sum(labels))
    print('No. of idiomatic expressions:',len(set(expressions)))
    return data,labels,expressions 

def LoadProbingTask(task,size):
    df = pd.read_csv('../Data/ProbingTasks/'+task+'.txt', sep='\t')
    df.columns = ['exps','labels','data']
    L = list(df.labels.unique())

    df = stratified_sample_df(df,'labels',len(df) if size is None else int(size/len(L)))

    exps = list(df.iloc[:,0])
    labels = list(df.iloc[:,1])
    data = list(df.iloc[:,2])
        
    labels = [L.index(l) for l in labels]
    
    print('Loaded',task,'with')
    print('No. samples:', len(data))
    print('Labels:', L)
    print(df.labels.value_counts())

    return list(data),list(labels),list(exps)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='idiom', help='task name')
    parser.add_argument('--mstart', type=int, default=5, help='starting number of topics')
    parser.add_argument('--mstop', type=int, default=51, help='stoping number of topics')
    parser.add_argument('--mstep', type=int, default=5, help='stepping number of topics increment')
    parser.add_argument('--size', type=int, default=None, help='maximum data size limit. None for no limit')

    parser.add_argument('--alllayers', action='store_true', help='Analyse all 12 layers of BERT')
    parser.add_argument('--train0labels', action='store_true', help='Train topic model using 0 (literal in case of idiom task) labels')
    parser.add_argument('--entropyanalyse', action='store_true')
    
    args=parser.parse_args() 
    t0 = time.time()
    if args.task=='idiom':   data,labels,expressions = LoadVNC(args.size)
    else:   data,labels,expressions = LoadProbingTask(args.task,args.size)
    
    #print(len(data))
    embs = ['Glove','Rand']
    embs += (['BERT'+str(l) for l in range(12)] if args.alllayers else ['BERT11'])
    
    ProbeModel = TopicAwareProbe(embs, args.mstart, args.mstop, args.mstep, args.train0labels, 5, args.alllayers)
    ProbeModel.probe(data,labels,expressions)
    t = timedelta(seconds=time.time()-t0)
    #print(time.time()-t0)
    print(t)

    
