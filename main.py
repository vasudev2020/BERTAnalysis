#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 15:46:11 2021

@author: vasu
"""
import pandas as pd
import argparse
from collections import defaultdict
from statistics import mean
from scipy import stats

import time

import json, pickle
from tqdm import tqdm


from TopicModel import TopicModel

from Analyser import Analyser
from Merger import Merger

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
    print('Number of samples',len(data)) 
    print(sum(labels),len(labels)-sum(labels))
    print(len(set(expressions)))
    return data,labels,expressions 

def LoadProbingTask(task,size):
    df = pd.read_csv('../Data/ProbingTasks/'+task+'.txt', sep='\t')
    df.columns = ['exps','labels','data']
    L = list(df.labels.unique())
    print(L)

    df = stratified_sample_df(df,'labels',len(df) if size is None else int(size/len(L)))
    print(df.labels.value_counts())

    exps = list(df.iloc[:,0])
    labels = list(df.iloc[:,1])
    data = list(df.iloc[:,2])
        
    labels = [L.index(l) for l in labels]

    return list(data),list(labels),list(exps)

def Print(Table,name,task):
    with open(task+'-'+name+'.json','w') as fp:
        json.dump(Table,fp)
    _,seen,unseen,diff = zip(*Table)
    t,p = list(stats.ttest_rel(diff,[0]*len(diff),alternative='greater'))
    print(task,name,mean(seen),mean(unseen),mean(diff),t,p)       
 
def PrintKeywords(size):
    data,labels,expressions = LoadVNC(size)
    
    TM = TopicModel()
    TM.train(data,25)
    Topics = TM.topicModel(data,expressions,labels)
    Topics = Topics.loc[Topics['Dominant_Topic']!=-1]
    
    Samples = {}
    for i,r in Topics.iterrows(): 
        topic = str(int(r['Dominant_Topic']))
        if topic not in Samples:    Samples[topic]=[r['Keywords'],set(r['Expressions']),1,0 if r['Labels']==0 else 1]
        else:   
            Samples[topic][1].add(r['Expressions'])
            Samples[topic][2]+=1
            if r['Labels']==1: Samples[topic][3]+=1
        #print(int(r['Dominant_Topic']),r['Keywords'],r['Labels'])
    for t in Samples.keys():
        print(t,Samples[t][0],Samples[t][2],min(Samples[t][3],(Samples[t][2]-Samples[t][3])))
           
def TMAnalyse(task, mstart, mstop, mstep, size, train0labels=False):
        
    if task=='idiom':   data,labels,expressions = LoadVNC(size)
    else:   data,labels,expressions = LoadProbingTask(task,size)
    
    TM = TopicModel()
    LabelEntropy = defaultdict(list)
    #ExpressionEntropy = defaultdict(list)

    for mp in tqdm(range(mstart,mstop,mstep)): 
        if train0labels:    TM.train([d for d,l in zip(data,labels) if l==0],mp)
            
        else:   TM.train(data,mp)
        
        Topics = TM.topicModel(data,expressions,labels)
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
    for l in LabelEntropy:  print('Label:',l,':',mean(LabelEntropy[l]),min(LabelEntropy[l]),max(LabelEntropy[l]))
    #for e in ExpressionEntropy:  print('Expression:',e,':',mean(ExpressionEntropy[e]))
             
def BERTTopicAnalysis(task, mstart, mstop, mstep, size, alllayers, train0labels=False):
    if task=='idiom':   data,labels,expressions = LoadVNC(size)
    else:   data,labels,expressions = LoadProbingTask(task,size)
    
    print(len(data))
    embs = ['Glove','Rand']
    embs += (['BERT'+str(l) for l in range(12)] if alllayers else ['BERT11'])
    
    merger = Merger(lc=100000,sc=1,tc=100)
    TM = TopicModel()
    n_class = len(set(labels))
    A = Analyser(5,False if n_class==2 else True, alllayers)
    
    Tables = [[] for _ in embs]
    TopicScores = [[] for _ in embs]
    
    for mp in range(mstart,mstop,mstep):  
        if train0labels:    TM.train([d for d,l in zip(data,labels) if l==0],mp)
        else:   TM.train(data,mp)
        Topics = TM.topicModel(data,expressions,labels)
        Topics = Topics.loc[Topics['Dominant_Topic']!=-1]
        
        MergedTopics = merger.Merge(Topics,m=None) 
        
        for i,embtype in enumerate(embs):
            scores,mbs,ci = A.perTopicAnalysis(MergedTopics,embtype)
            Tables[i]+=scores
            TopicScores[i].append([mbs,ci])
    for T,embtype in zip(Tables,embs):    Print(T,embtype,task) 
    print(TopicScores)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='idiom', help='task name')
    parser.add_argument('--mstart', type=int, default=5, help='starting number of topics')
    parser.add_argument('--mstop', type=int, default=51, help='stoping number of topics')
    parser.add_argument('--mstep', type=int, default=5, help='stepping number of topics increment')
    parser.add_argument('--size', type=int, default=None, help='maximum data size limit. None for no limit')
    parser.add_argument('--alllayers', action='store_true')
    parser.add_argument('--train0labels', action='store_true')
    parser.add_argument('--entropyanalyse', action='store_true')
    
    args=parser.parse_args() 
    t0 = time.time()
    if args.entropyanalyse:
        TMAnalyse(args.task,args.mstart,args.mstop,args.mstep,args.size,args.train0labels)
    else:
        BERTTopicAnalysis(args.task,args.mstart,args.mstop,args.mstep,args.size,args.alllayers,args.train0labels)
    #TMAnalyse('bshift',10,51,5,10000)
    print(time.time()-t0)
    
    #
    #PrintKeywords(None)
