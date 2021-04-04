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
    

def Print(Table,name,task):
    with open(task+'-'+name+'.json','w') as fp:
        json.dump(Table,fp)
    _,seen,unseen,diff = zip(*Table)
    t1,p1 = list(stats.ttest_rel(seen,unseen))
    t2,p2 = list(stats.ttest_rel(diff,[0]*len(diff)))
    
    print(task,name,mean(seen),mean(unseen),mean(diff),t1,p1,t2,p2)       
    
def BERTTopicAnalysis(task,mstart,mstop,mstep,size,alllayers):
    if task=='idiom':   data,labels,expressions = LoadVNC(size)
    if task=='fullidiom':   data,labels,expressions = LoadFullVNC(size)
    if task=='bshift':   data,labels,expressions = LoadBShift(size)
    if task=='somo':  data,labels,expressions = LoadSOMO(size)
    
    embs = ['Glove','Rand']
    embs += (['BERT'+str(l) for l in range(12)] if alllayers else ['BERT11'])
    
    merger = Merger(lc=100000,sc=1,tc=100)
    TM = TopicModel()
    A = Analyser(5,alllayers)
    
    Tables = [[] for _ in embs]
    TopicScores = [[] for _ in embs]
    
    for mp in range(mstart,mstop,mstep):       
        TM.train(data,mp)
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
    
     
    args=parser.parse_args() 
    
    BERTTopicAnalysis(args.task,args.mstart,args.mstop,args.mstep,args.size,args.alllayers)
    #ComputeCoherance(args.mstart,args.mstop,args.mstep)
    #LoadSOMO(None)
    #LoadVNC(None)
    #LoadFullVNC(None)
