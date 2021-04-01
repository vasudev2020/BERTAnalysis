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
    return data,labels,expressions 

def LoadBShift(size):
    with open('../Data/bigram_shift.txt','r') as fp:
        dataset = fp.readlines()
    size = len(dataset) if size is None else size
    dataset=dataset[:size]
    print('Number of samples',len(dataset))
    dataset = [d.strip().split('\t') for d in dataset]
    exps,labels,data = zip(*dataset)
    
    labels = [0 if l == 'O' else 1 for l in labels]
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

def Print(Table,name,task):
    with open(task+'-'+name+'.json','w') as fp:
        json.dump(Table,fp)
    _,seen,unseen,diff = zip(*Table)
    t1,p1 = list(stats.ttest_rel(seen,unseen))
    t2,p2 = list(stats.ttest_rel(diff,[0]*len(diff)))
    
    print(task,name,mean(seen),mean(unseen),mean(diff),t1,p1,t2,p2)       
    
def BERTTopicAnalysis(task,mstart,mstop,mstep,size,alllayers,glove,randemb):
    if task=='idiom':   data,labels,expressions = LoadVNC(size)
    #data,labels,expressions = LoadBShift(size)
    #data,labels,expressions = LoadJokes(size)
    
    merger = Merger(lc=100000,sc=1,tc=100)
    TM = TopicModel()
    if alllayers:   Analysers = [Analyser(5,'BERT',i) for i in range(12)]
    else: Analysers = [Analyser(5,'BERT',11),Analyser(5,'BERT',8),Analyser(5,'BERT',0)]
    if glove:   Analysers.append(Analyser(5,'Glove'))
    if randemb: Analysers.append(Analyser(5,'Rand'))
    
    Tables = [[] for _ in Analysers]
    TopicScores = [[] for _ in Analysers]
    
    for mp in range(mstart,mstop,mstep):       
        TM.train(data,mp)
        Topics = TM.topicModel(data,expressions,labels)
        Topics = Topics.loc[Topics['Dominant_Topic']!=-1]
        
        MergedTopics = merger.Merge(Topics,m=None) 
        
        for i,A in enumerate(Analysers):
            scores,mbs,ci = A.perTopicAnalysis(MergedTopics)
            Tables[i]+=scores
            TopicScores[i].append([mbs,ci])
    for T,A in zip(Tables,Analysers):    Print(T,A.getName(),task) 
    print(TopicScores)
        
     
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='idiom', help='task name')
    parser.add_argument('--mstart', type=int, default=5, help='starting number of topics')
    parser.add_argument('--mstop', type=int, default=51, help='stoping number of topics')
    parser.add_argument('--mstep', type=int, default=5, help='stepping number of topics increment')
    parser.add_argument('--size', type=int, default=None, help='maximum data size limit. None for no limit')
    parser.add_argument('--alllayers', action='store_true')
    parser.add_argument('--glove', action='store_true')
    parser.add_argument('--randemb', action='store_true')
    
     
    args=parser.parse_args() 
    
    BERTTopicAnalysis(args.task,args.mstart,args.mstop,args.mstep,args.size,args.alllayers,args.glove,args.randemb)
    #ComputeCoherance(args.mstart,args.mstop,args.mstep)
    #LoadBShift()
