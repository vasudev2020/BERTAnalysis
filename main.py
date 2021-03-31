#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 14:19:49 2021

@author: vasu
"""


import pickle
import pandas as pd
import argparse

from bertopic import BERTopic

#from TopicModel import TopicModel
from Analyser import Analyser
from Merger import Merger
#import os
from collections import defaultdict
import statistics

def shortview(Data):
    data = Data.groupby(['Dominant_Topic','Labels','Old_Topic']).size()
    
    for dt in Data['Dominant_Topic'].unique():
        data = Data.loc[Data['Dominant_Topic']==dt]
        literal = data.groupby('Labels').size()[0]
        idiom = data.groupby('Labels').size()[1]
        size = idiom+literal
        no_topics = data['Old_Topic'].unique()
        print(dt,size,idiom/size,no_topics)
        
def BERTopicModel(data,expressions,labels,topic_model):
    topics,_ = topic_model.fit_transform(data)
    
    sent = pd.Series(data,name='Sent')
    exp = pd.Series(expressions,name='Expressions')
    label = pd.Series(labels,name='Labels')
    kw = pd.Series([','.join([k for k,_ in topic_model.get_topic(t)]) for t in topics],name='Keywords')
    dt = pd.Series(topics,name='Dominant_Topic')
    
    Topics = pd.concat([dt,kw,sent,label,exp],axis=1)
    
    '''Remove outliers'''
    Topics = Topics.loc[Topics['Dominant_Topic']!=-1]
    
    return Topics

def BERTTopicAnalysis(min_topic_size,m):
    dataset = pickle.load(open("../Data/ID/vnics_dataset_full_ratio-split.pkl", "rb"), encoding='latin1')          
    data = [p['sent'].replace(' &apos;','\'') for p in dataset["train_sample"]+dataset["test_sample"]]
    labels = [p['lab_int'] for p in dataset["train_sample"]+dataset["test_sample"]]
    expressions = [p['verb']+' '+p['noun'] for p in dataset["train_sample"]+dataset["test_sample"]]
    print('Number of samples',len(data))
    
    merger = Merger(lc=100000,sc=1,tc=100)
    topic_model = BERTopic(min_topic_size=min_topic_size)
        
    BERT11Analyser = Analyser(5,'BERT',11)
    BERT10Analyser = Analyser(5,'BERT',10)
    BERT9Analyser = Analyser(5,'BERT',9)
    BERT8Analyser = Analyser(5,'BERT',8)
    BERT7Analyser = Analyser(5,'BERT',7)
    BERT6Analyser = Analyser(5,'BERT',6)
    BERT5Analyser = Analyser(5,'BERT',5)
    BERT4Analyser = Analyser(5,'BERT',4)
    BERT3Analyser = Analyser(5,'BERT',3)
    BERT2Analyser = Analyser(5,'BERT',2)
    BERT1Analyser = Analyser(5,'BERT',1)
    BERT0Analyser = Analyser(5,'BERT',0)
    #GloveAnalyser = Analyser(5,'Glove')
    #RandAnalyser = Analyser(5,'Rand')

    TableBERT11 = defaultdict(list)
    TableBERT10 = defaultdict(list)
    TableBERT9 = defaultdict(list)
    TableBERT8 = defaultdict(list)
    TableBERT7 = defaultdict(list)
    TableBERT6 = defaultdict(list)
    TableBERT5 = defaultdict(list)
    TableBERT4 = defaultdict(list)
    TableBERT3 = defaultdict(list)
    TableBERT2 = defaultdict(list)
    TableBERT1 = defaultdict(list)
    TableBERT0 = defaultdict(list)
    #TableGlove = defaultdict(list)
    #TableRand = defaultdict(list)
    for _ in range(10):
        Topics = BERTopicModel(data,expressions,labels,topic_model)
        MergedTopics = merger.Merge(Topics,m)          
       
        for i,v in enumerate(BERT11Analyser.perTopicAnalysis(MergedTopics)): 
            TableBERT11[i].append(v)
        
        for i,v in enumerate(BERT10Analyser.perTopicAnalysis(MergedTopics)): 
            TableBERT10[i].append(v)
            
        for i,v in enumerate(BERT9Analyser.perTopicAnalysis(MergedTopics)): 
            TableBERT9[i].append(v)
           
        for i,v in enumerate(BERT8Analyser.perTopicAnalysis(MergedTopics)): 
            TableBERT8[i].append(v)
        
        for i,v in enumerate(BERT7Analyser.perTopicAnalysis(MergedTopics)): 
            TableBERT7[i].append(v)
            
        for i,v in enumerate(BERT6Analyser.perTopicAnalysis(MergedTopics)): 
            TableBERT6[i].append(v)
           
        for i,v in enumerate(BERT5Analyser.perTopicAnalysis(MergedTopics)): 
            TableBERT5[i].append(v)
            
        for i,v in enumerate(BERT4Analyser.perTopicAnalysis(MergedTopics)): 
            TableBERT4[i].append(v)
            
        for i,v in enumerate(BERT3Analyser.perTopicAnalysis(MergedTopics)): 
            TableBERT3[i].append(v)            
            
        for i,v in enumerate(BERT2Analyser.perTopicAnalysis(MergedTopics)): 
            TableBERT2[i].append(v)
            
        for i,v in enumerate(BERT1Analyser.perTopicAnalysis(MergedTopics)): 
            TableBERT1[i].append(v)
            
        for i,v in enumerate(BERT0Analyser.perTopicAnalysis(MergedTopics)): 
            TableBERT0[i].append(v)
          
        '''
        for i,v in enumerate(GloveAnalyser.perTopicAnalysis(MergedTopics)): 
            TableGlove[i].append(v)
                
        for i,v in enumerate(RandAnalyser.perTopicAnalysis(MergedTopics)): 
            TableRand[i].append(v)
        '''
                    
    print('BERT11:',' '.join([str(statistics.mean(TableBERT11[v])) for v in TableBERT11]))
    print('BERT10:',' '.join([str(statistics.mean(TableBERT10[v])) for v in TableBERT10]))
    print('BERT9:',' '.join([str(statistics.mean(TableBERT9[v])) for v in TableBERT9]))
    print('BERT8:',' '.join([str(statistics.mean(TableBERT8[v])) for v in TableBERT8]))
    print('BERT7:',' '.join([str(statistics.mean(TableBERT7[v])) for v in TableBERT7]))
    print('BERT6:',' '.join([str(statistics.mean(TableBERT6[v])) for v in TableBERT6]))
    print('BERT5:',' '.join([str(statistics.mean(TableBERT5[v])) for v in TableBERT5]))
    print('BERT4:',' '.join([str(statistics.mean(TableBERT4[v])) for v in TableBERT4]))
    print('BERT3:',' '.join([str(statistics.mean(TableBERT3[v])) for v in TableBERT3]))
    print('BERT2:',' '.join([str(statistics.mean(TableBERT2[v])) for v in TableBERT2]))
    print('BERT1:',' '.join([str(statistics.mean(TableBERT1[v])) for v in TableBERT1]))
    print('BERT0:',' '.join([str(statistics.mean(TableBERT0[v])) for v in TableBERT0]))
    #print(' '.join([str(statistics.mean(TableGlove[v])) for v in TableGlove]))
    #print(' '.join([str(statistics.mean(TableRand[v])) for v in TableRand]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, default=None, help='number of topics')
    parser.add_argument('--min_topic_size', type=int, default=5, help='minimum size of topic')
     
    args=parser.parse_args() 
    
    BERTTopicAnalysis(args.min_topic_size,args.m)


#TM = TopicModel()


