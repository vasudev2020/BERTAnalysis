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

from TopicModel import TopicModel
from Analyser import Analyser
from Merger import Merger
import os

    
def view(Data):        
    Record = {}
    
    for i,row in Data.iterrows():
        topic = row['Dominant_Topic']
        oldtopic = row['Old_Topic'] if 'Old_Topic' in row else topic

        if topic not in Record: 
            Record[topic]={'Exps':[],'Labels':[],'Keywords':[],'OldTopics':[]}
        if row['Expressions'] not in Record[topic]['Exps']:  Record[topic]['Exps'].append(row['Expressions'])
        Record[topic]['Labels'].append(row['Labels'])
        if row['Keywords'] not in Record[topic]['Keywords']:  Record[topic]['Keywords'].append(row['Keywords'])
        if oldtopic not in Record[topic]['OldTopics']:  Record[topic]['OldTopics'].append(oldtopic)
        
    for topic in sorted(Record.keys()):
        id_count = sum(Record[topic]['Labels'])
        lt_count = len(Record[topic]['Labels'])-id_count
        print('Topic:',topic,lt_count,id_count,id_count/len(Record[topic]['Labels']))
        print('Old Topics:',Record[topic]['OldTopics'])
        print('Expressions:',', '.join(Record[topic]['Exps']))
        for k in Record[topic]['Keywords']:   print('Keywords:', k)
        print('')

def shortview(Data):
    data = Data.groupby(['Dominant_Topic','Labels','Old_Topic']).size()
    
    #print(data)
    for dt in Data['Dominant_Topic'].unique():
        data = Data.loc[Data['Dominant_Topic']==dt]
        literal = data.groupby('Labels').size()[0]
        idiom = data.groupby('Labels').size()[1]
        size = idiom+literal
        no_topics = data['Old_Topic'].unique()
        print(dt,size,idiom/size,no_topics)

        
    #size, ratio, number of topics
    
def perLabelTopicModeling(no_topics):
    dataset = pickle.load(open("../Data/ID/vnics_dataset_full_ratio-split.pkl", "rb"), encoding='latin1')       

    p_data = [p['sent'].replace(' &apos;','\'') for p in dataset["train_sample"] if p['lab_int']==1]
    n_data = [p['sent'].replace(' &apos;','\'') for p in dataset["train_sample"] if p['lab_int']==0]
    p_exps = [p['verb']+' '+p['noun'] for p in dataset["train_sample"] if p['lab_int']==1]
    n_exps = [p['verb']+' '+p['noun'] for p in dataset["train_sample"] if p['lab_int']==0]
    
    TM = TopicModel()
    
    TM.train(p_data,no_topics)
    p_topics = TM.topicModel(p_data,p_exps,[1]*len(p_data))
    
    TM.train(n_data,no_topics)
    n_topics = TM.topicModel(n_data,n_exps,[0]*len(n_data))
    
    return p_topics,n_topics

def perExpressionTopicModeling(no_topics):
    Data,Labels = {},{}
    dataset = pickle.load(open("../Data/ID/vnics_dataset_full_ratio-split.pkl", "rb"), encoding='latin1')       
    data = [p['sent'].replace(' &apos;','\'') for p in dataset["train_sample"]+dataset["test_sample"]]
    expressions = [p['verb']+' '+p['noun'] for p in dataset["train_sample"]+dataset["test_sample"]]   
    labels = [p['lab_int'] for p in dataset["train_sample"]+dataset["test_sample"]]
    
    for sent,exp,label in zip(data, expressions, labels):
        if exp not in Data: 
            Data[exp]=[]
            Labels[exp]=[]
        Data[exp].append(sent)
        Labels[exp].append(label)
     
    TM = TopicModel()

    for exp in Data:
        print(exp)
        TM.train(Data[exp],4)
        topics = TM.topicModel(Data[exp],[exp]*len(Data[exp]),Labels[exp])
        view(topics)
    
    
def trainTopicModel(n,num_topics,use_vnic):
    files = os.listdir('../Data/BNC/ParsedTexts')
    data = []
    #TODO: list of para or sent?
    if n==-1:   n = len(files)
    for f in files[:n]: data = data+open('../Data/BNC/ParsedTexts/'+f,'r').readlines()
        
    if use_vnic:
        '''read data from VNIC'''
        dataset = pickle.load(open("../Data/ID/vnics_dataset_full_ratio-split.pkl", "rb"), encoding='latin1')       
        data = data + [p['sent'].replace(' &apos;','\'') for p in dataset["train_sample"]+dataset["test_sample"]]
    print('Training data size:', len(data))
    TM = TopicModel()
    TM.train(data,num_topics)
    TM.save()
    print('Trainig finished')
    
def predictTopic(merge):
    '''read data as a list of sentences'''
    dataset = pickle.load(open("../Data/ID/vnics_dataset_full_ratio-split.pkl", "rb"), encoding='latin1')       
    
    data = [p['sent'].replace(' &apos;','\'') for p in dataset["train_sample"]+dataset["test_sample"]]
    labels = [p['lab_int'] for p in dataset["train_sample"]+dataset["test_sample"]]
    expressions = [p['verb']+' '+p['noun'] for p in dataset["train_sample"]+dataset["test_sample"]]
    
    TM = TopicModel()
    TM.load()
    topics = TM.topicModel(data,expressions,labels)
    if merge:   topics = merge(topics)

    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):    print(topics)
    view(topics)
    return topics
        
def idiomTest(merge):  
    '''read data as a list of sentences'''
    dataset = pickle.load(open("../Data/ID/vnics_dataset_full_ratio-split.pkl", "rb"), encoding='latin1')       
    
    data = [p['sent'].replace(' &apos;','\'') for p in dataset["train_sample"]+dataset["test_sample"]]
    labels = [p['lab_int'] for p in dataset["train_sample"]+dataset["test_sample"]]
    expressions = [p['verb']+' '+p['noun'] for p in dataset["train_sample"]+dataset["test_sample"]]
    
    TM = TopicModel()
    TM.load()
    topics = TM.topicModel(data,expressions,labels)
    
    if merge:   
        M = Merger(m=12)
        topics = M.Merge(topics)
        
    A = Analyser(5)
    A.perTopicAnalysis(topics)        
    
    #compute_coherence_values(start=2, limit=100, step=1)
    
def BERTTopicAnalysis(min_topic_size):
    dataset = pickle.load(open("../Data/ID/vnics_dataset_full_ratio-split.pkl", "rb"), encoding='latin1')       
    
    data = [p['sent'].replace(' &apos;','\'') for p in dataset["train_sample"]+dataset["test_sample"]]
    labels = [p['lab_int'] for p in dataset["train_sample"]+dataset["test_sample"]]
    expressions = [p['verb']+' '+p['noun'] for p in dataset["train_sample"]+dataset["test_sample"]]
    
    #topic_model = BERTopic(min_topic_size=8)
    topic_model = BERTopic(min_topic_size=min_topic_size)
    #topic_model = BERTopic()
    topics,_ = topic_model.fit_transform(data)
    
    sent = pd.Series(data,name='Sent')
    exp = pd.Series(expressions,name='Expressions')
    label = pd.Series(labels,name='Labels')
    kw = pd.Series([','.join([k for k,_ in topic_model.get_topic(t)]) for t in topics],name='Keywords')
    dt = pd.Series(topics,name='Dominant_Topic')
    
    Topics = pd.concat([dt,kw,sent,label,exp],axis=1)
    print('Number of samples',len(Topics.index))
    
    '''Remove outliers'''
    Topics = Topics.loc[Topics['Dominant_Topic']!=-1]

    print('Number of Topics:',len(Topics['Dominant_Topic'].unique()))
    print('Number of samples after outlier removal',len(Topics.index))
    
    lc = 100000
    sc = 1
    tc = 100
    #tc = 1000/(len(Topics['Dominant_Topic'].unique())-m+1)
    
    M = Merger(lc=lc,sc=sc,tc=tc)
    A = Analyser(5)
    for m in range(5,21,5):
        print('Number of topic groups after merging)',m)

        MergedTopics = M.Merge(Topics,m)
        A.perTopicAnalysis(MergedTopics) 
        print('Number of sentences after merging:',len(MergedTopics.index))    
        #shortview(MergedTopics)
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    #parser.add_argument('--comp_op', type=str, default='add', help='name of the compositional operator (add/linear/holo)')
    #parser.add_argument('--lr', type=float, default=0.25, help='Learning rate')
    parser.add_argument('--n', type=int, default=1, help='number of BNC files for training')
    parser.add_argument('--num_topics', type=int, default=10, help='number of topics')

    parser.add_argument('--train', action='store_true',help='Train topic model')
    parser.add_argument('--test', action='store_true',help='Test topic model')
    parser.add_argument('--idiomtest', action='store_true',help='Test topic model')

    parser.add_argument('--use_vnic', action='store_true',help='Use VNIC data along with BNC for training')
    parser.add_argument('--bertopic', action='store_true',help='Test topic model')
    parser.add_argument('--merge', action='store_true',help='Test topic model')
    parser.add_argument('--m', type=int, default=10, help='number of topics')
    parser.add_argument('--min_topic_size', type=int, default=5, help='minimum size of topic')

 
    args=parser.parse_args() 
    if args.train:  trainTopicModel(args.n, args.num_topics, args.use_vnic)
    if args.test:   predictTopic(args.merge)
    if args.idiomtest:  idiomTest(args.merge)
    if args.bertopic:   BERTTopicAnalysis(args.min_topic_size)


#TM = TopicModel()
#TM.train(['What is your name? my name is Vasu.What is topic modelling?I am fine.How do you do?','This is a sample sentence.'],2)


