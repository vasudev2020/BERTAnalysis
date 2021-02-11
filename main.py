#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 14:19:49 2021

@author: vasu
"""


import pickle

from TopicModel import TopicModel
import os

import argparse


def view(Data):    
    #print(Data)
    #for i in Data['Dominant_Topic'].unique():
    #    Data[i]
    #print(Data.groupby(['Dominant_Topic']).mean())
    #print(Data.groupby(['Dominant_Topic']).size())
    #print(Data.groupby(['Dominant_Topic','Labels']).size())    
    
    #Topics = {}
    Exps = {}
    Labels = {}
    Keywords = {}
    for i,row in Data.iterrows():
        exp = row['Expressions']
        topic = row['Dominant_Topic']
        label = row['Labels']
        kw = row['Keywords']
        #print(exp)
        if topic not in Exps:   Exps[topic]=[]
        if exp not in Exps[topic]:  Exps[topic].append(exp)
        if topic not in Labels:   Labels[topic]=[]
        Labels[topic].append(label)
        if topic not in Keywords:   Keywords[topic]=kw
        else:   assert Keywords[topic]==kw
        
        #if topic not in Topics: Topics[topic]={}
        #if exp not in Topics[topic]:    Topics[topic][exp]=0
    #Topics[topic][exp]+=1
    #print(row['Document_No'],row['Dominant_Topic'],row['Keywords'],row['Text'])
    
    for topic in Exps:
        ratio = round(sum(Labels[topic])/len(Labels[topic]),4)
        ratio = (1-ratio) if ratio<0.5 else ratio
        print('Topic:',int(topic))
        print('Labels:',sum(Labels[topic]),len(Labels[topic]),ratio)
        print('Expressions:',', '.join(Exps[topic]))
        print('Keywords:', Keywords[topic])

    '''
    for topic in Topics:
        print(str(int(topic))+' - '+','.join([exp+':'+str(Topics[topic][exp]) for exp in Topics[topic]]))
    topic_count = [sum([Topics[topic][exp] for exp in Topics[topic]]) for topic in Topics]
    print(topic_count)
    '''
    
    '''
    for topic in Topics:
        print(str(int(topic))+' - '+','.join([exp+'('+str(Topics[topic][exp][0])+','+str(Topics[topic][exp][1])+')' for exp in Topics[topic]]))
        
    topic_count = [sum([Topics[topic][exp][1]+Topics[topic][exp][1] for exp in Topics[topic]]) for topic in Topics]
    print(topic_count)
    '''
    
    #union = lambda x: reduce(set.union, x)
    #print(dom_topics.groupby(['Dominant_Topic']).aggregate({'Expression':union,'Label':'mean'}))
    
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
    
def trainTopicModel(n,use_vnic):
    files = os.listdir('../Data/BNC/ParsedTexts')
    data = []
    #TODO: list of para or sent?
    for f in files[:n]: data = data+open('../Data/BNC/ParsedTexts/'+f,'r').readlines()
        
    if use_vnic:
        '''read data from VNIC'''
        dataset = pickle.load(open("../Data/ID/vnics_dataset_full_ratio-split.pkl", "rb"), encoding='latin1')       
        data = data + [p['sent'].replace(' &apos;','\'') for p in dataset["train_sample"]+dataset["test_sample"]]
    print('Training data size:', len(data))
    TM = TopicModel()
    TM.train(data,40)
    TM.save()
    print('Trainig finished')
    
def predictTopic():
    '''read data as a list of sentences'''
    dataset = pickle.load(open("../Data/ID/vnics_dataset_full_ratio-split.pkl", "rb"), encoding='latin1')       
    
    data = [p['sent'].replace(' &apos;','\'') for p in dataset["train_sample"]+dataset["test_sample"]]
    labels = [p['lab_int'] for p in dataset["train_sample"]+dataset["test_sample"]]
    expressions = [p['verb']+' '+p['noun'] for p in dataset["train_sample"]+dataset["test_sample"]]
    
    TM = TopicModel()
    TM.load()
    topics = TM.topicModel(data,expressions,labels)
    view(topics)
    return topics
        
def main():    
    
    '''
    Data,Labels = {},{}
    for sent,exp,label in zip(data,expressions,labels):
        if exp not in Data: 
            Data[exp]=[]
            Labels[exp]=[]
        Data[exp].append(sent)
        Labels[exp].append(label)
        
    for exp in Data:
        print(exp)
        topics = topicModel(Data[exp],[exp]*len(Data[exp]),Labels[exp],4)
        view(topics)
    
    '''
    
    #compute_coherence_values(start=2, limit=100, step=1)
        
    
    #Folds = SplitToFolds(topics,5)
    #perTopicIdiomDetection(Folds)
    
    #print(len(Folds),len(Folds[0]))
    #for i in range(10): print(len(Folds[i][0]),len(Folds[i][1]),len(Folds[i][2]),len(Folds[i][3]),len(Folds[i][4]))
    #print(Folds[0][0])

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    #parser.add_argument('--comp_op', type=str, default='add', help='name of the compositional operator (add/linear/holo)')
    #parser.add_argument('--lr', type=float, default=0.25, help='Learning rate')
    parser.add_argument('--n', type=int, default=10, help='number of BNC files for training')
    parser.add_argument('--train', action='store_true',help='Train topic model')
    parser.add_argument('--test', action='store_true',help='Test topic model')
    parser.add_argument('--use_vnic', action='store_true',help='Use VNIC data along with BNC for training')

 
    args=parser.parse_args() 
    if args.train:
        trainTopicModel(args.n,args.use_vnic)
    if args.test:
        predictTopic()


#TM = TopicModel()
#TM.train(['What is your name? my name is Vasu.What is topic modelling?I am fine.How do you do?','This is a sample sentence.'],2)


