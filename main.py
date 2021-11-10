#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 15:46:11 2021

@author: vasu
"""
import pandas as pd
import pickle
import argparse
from collections import defaultdict
from statistics import mean
from scipy import stats

import time

import json


from TopicModel import TopicModel

from Analyser import Analyser
from Merger import Merger

def LoadVNC(size):
    dataset = pickle.load(open("../Data/ID/vnics_dataset_full_ratio-split.pkl", "rb"), encoding='latin1')          
    data = [p['sent'].replace(' &apos;','\'') for p in dataset["train_sample"]+dataset["test_sample"]]
    labels = [p['lab_int'] for p in dataset["train_sample"]+dataset["test_sample"]]
    expressions = [p['verb']+' '+p['noun'] for p in dataset["train_sample"]+dataset["test_sample"]]
    print('Number of samples',len(data)) 
    print(sum(labels),len(labels)-sum(labels))
    print(len(set(expressions)))
    return data,labels,expressions 

def LoadFullVNC(size):        
    with open("../Data/ID/cook_dataset.tok", "r") as fp:
        dataset = fp.readlines()
    dataset = [d.strip().split('||') for d in dataset]
    dataset = [[d[0],d[1]+' '+d[2],d[3]] for d in dataset if d[0]!='Q']
    size = len(dataset) if size is None else size
    dataset = dataset[:size]
    print('Number of samples',len(dataset))
    labels,exps, data = zip(*dataset)
       
    labels = [0 if l == 'L' else 1 for l in labels]
    print(sum(labels),len(labels)-sum(labels))
    print(len(set(exps)))    
    return list(data),list(labels),list(exps)
    

def LoadBShift(size):
    with open('../Data/bigram_shift.txt','r') as fp:
        dataset = fp.readlines()
    size = len(dataset) if size is None else size
    dataset=dataset[:size]
    print('Number of samples',len(dataset))
    dataset = [d.strip().split('\t') for d in dataset]
    exps,labels,data = zip(*dataset)
    
    labels = [0 if l == 'O' else 1 for l in labels]
    print(sum(labels),len(labels)-sum(labels))
    return list(data),list(labels),list(exps)

def LoadJokes(size):
    with open('../Data/reddit_jokes.json','r') as fp:
        dataset = json.load(fp)
    size = len(dataset) if size is None else size
    dataset = dataset[:size]
    print('Number of samples',len(dataset))
    data = [d['title']+d['body'] for d in dataset]
    labels = [d['score'] for d in dataset]
    labels = [1 if lb > 0 else 0 for lb in labels]
    exps = [d['title'] for d in dataset]
    return data,labels,exps

def LoadSOMO(size):    
    with open('../Data/odd_man_out.txt','r') as fp:
        dataset = fp.readlines()
    size = len(dataset) if size is None else size
    dataset=dataset[:size]
    print('Number of samples',len(dataset))
    dataset = [d.strip().split('\t') for d in dataset]
    exps,labels,data = zip(*dataset)
    
    labels = [0 if l == 'O' else 1 for l in labels]
    return list(data),list(labels),list(exps)
    return data,labels,exps
def LoadSentLen(size):
    df = pd.read_csv('../Data/Sentlen/sentence_length.txt', sep='\t')
    df.columns = ['exps','labels','data']

    df = stratified_sample_df(df,'labels',len(df) if size is None else int(size/6))
    exps = list(df.iloc[:,0])
    labels = list(df.iloc[:,1])
    data = list(df.iloc[:,2])
    
    labels = [0 if l in [0,1,2] else 1 for l in labels]

    print('0:',sum(labels))
    print('1:',len(labels)-sum(labels))
    return list(data),list(labels),list(exps)
 
def stratified_sample_df(df, col, n_samples):
    n = min(n_samples, df[col].value_counts().min())
    df_ = df.groupby(col).apply(lambda x: x.sample(n))
    df_.index = df_.index.droplevel(0)
    return df_

def Print(Table,name,task):
    with open(task+'-'+name+'.json','w') as fp:
        json.dump(Table,fp)
    _,seen,unseen,diff = zip(*Table)
    #t1,p1 = list(stats.ttest_rel(seen,unseen))
    #t2,p2 = list(stats.ttest_rel(diff,[0]*len(diff)))
    t,p = list(stats.ttest_rel(diff,[0]*len(diff),alternative='greater'))
    #print(task,name,mean(seen),mean(unseen),mean(diff),t1,p1,t2,p2)       
    print(task,name,mean(seen),mean(unseen),mean(diff),t,p)       
    
def BERTTopicAnalysis(task,mstart,mstop,mstep,size,alllayers,train0labels=False):
    if task=='idiom':   data,labels,expressions = LoadVNC(size)
    if task=='fullidiom':   data,labels,expressions = LoadFullVNC(size)
    if task=='bshift':   data,labels,expressions = LoadBShift(size)
    if task=='somo':  data,labels,expressions = LoadSOMO(size)
    if task=='sentlen':  data,labels,expressions = LoadSentLen(size)
    
    embs = ['Glove','Rand']
    embs += (['BERT'+str(l) for l in range(12)] if alllayers else ['BERT11'])
    
    merger = Merger(lc=100000,sc=1,tc=100)
    TM = TopicModel()
    A = Analyser(5,alllayers)
    
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
    
def TMAnlyse_old(task,mstart,mstop,mstep,size,train0labels=False):
    if task=='idiom':   data,labels,expressions = LoadVNC(size)
    if task=='fullidiom':   data,labels,expressions = LoadFullVNC(size)
    if task=='bshift':   data,labels,expressions = LoadBShift(size)
    if task=='somo':  data,labels,expressions = LoadSOMO(size)
    if task=='sentlen':  data,labels,expressions = LoadSentLen(size)

    TM = TopicModel()
    H_exp = pd.Series()
    H_label = pd.Series()
    for mp in range(mstart,mstop,mstep):  
        if train0labels:    TM.train([d for d,l in zip(data,labels) if l==0],mp)
        else:   TM.train(data,mp)
        Topics = TM.topicModel(data,expressions,labels)
        Topics = Topics.loc[Topics['Dominant_Topic']!=-1]
        h_exp = Topics.groupby('Dominant_Topic')['Expressions'].apply(lambda x : stats.entropy(x.value_counts(), base=2)).reset_index()
        h_label = Topics.groupby('Dominant_Topic')['Labels'].apply(lambda x : stats.entropy(x.value_counts(), base=2)).reset_index()
        H_exp = pd.concat([H_exp,h_exp],axis=0,ignore_index=True)
        H_label = pd.concat([H_label,h_label],axis=0,ignore_index=True)

    print('Mean expression entropy',H_exp.mean()['Expressions'])
    print('Mean label entropy',H_label.mean()['Labels'])
 
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
        
    
    
def TMAnalyse(task,mstart,mstop,mstep,size,train0labels=False):
        
    if task=='idiom':   data,labels,expressions = LoadVNC(size)
    if task=='fullidiom':   data,labels,expressions = LoadFullVNC(size)
    if task=='bshift':   data,labels,expressions = LoadBShift(size)
    if task=='somo':  data,labels,expressions = LoadSOMO(size)
    if task=='sentlen':  data,labels,expressions = LoadSentLen(size)

    
    TM = TopicModel()
    LabelEntropy = defaultdict(list)
    ExpressionEntropy = defaultdict(list)

    for mp in range(mstart,mstop,mstep): 
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
        
        tg = Topics.groupby(['Expressions','Dominant_Topic']).size()
        for e in Topics['Expressions'].unique():
            Ze = len(Topics.loc[Topics['Expressions']==e])
            Zt = len(Topics['Dominant_Topic'].unique())
            
            Z = stats.entropy([1]*min(Zt,Ze))
            h = stats.entropy(tg[e])/Z
            assert h<=1.0 and h>=0
            ExpressionEntropy[e].append(h)
            
        #TODO: keywords of topics
         
        
            
    #print(LabelEntropy)
    for l in LabelEntropy:  print('Label:',l,':',mean(LabelEntropy[l]))
    for e in ExpressionEntropy:  print('Expression:',e,':',mean(ExpressionEntropy[e]))
        
     
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='idiom', help='task name')
    parser.add_argument('--mstart', type=int, default=5, help='starting number of topics')
    parser.add_argument('--mstop', type=int, default=51, help='stoping number of topics')
    parser.add_argument('--mstep', type=int, default=5, help='stepping number of topics increment')
    parser.add_argument('--size', type=int, default=None, help='maximum data size limit. None for no limit')
    parser.add_argument('--alllayers', action='store_true')
    parser.add_argument('--train0labels', action='store_true')
    
    
     
    args=parser.parse_args() 
    t0 = time.time()
    BERTTopicAnalysis(args.task,args.mstart,args.mstop,args.mstep,args.size,args.alllayers,args.train0labels)
    #LoadSentLen(6000)
    print(time.time()-t0)
    
    #ComputeCoherance(args.mstart,args.mstop,args.mstep)
    #LoadBShift(10000)
    #LoadSOMO(None)
    #LoadVNC(None)
    #LoadFullVNC(None)
    
    #TMAnalyse('bshift',10,51,5,10000)
    
    #PrintKeywords(None)
