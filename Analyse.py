#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 15:46:11 2021

@author: vasu
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import pandas as pd
import pickle
import argparse
from collections import defaultdict
import statistics

from TopicModel import TopicModel
from bertopic import BERTopic

from Analyser import Analyser
from Merger import Merger

count_vectorizer = CountVectorizer(stop_words='english')

def LSI(data,exps,labels,n_topics):

    TM = TopicModel()
    TM.train(data,n_topics)
    Topics = TM.topicModel(data,exps,labels)
    '''
    count_data = count_vectorizer.fit_transform(data)
    
    lda = LDA(n_components=n_topics)
    lda.fit(count_data)
    
    out = lda.transform(count_data) #array (n_samples,n_components)
    out = out.argmax(-1)            #array n_samples
    
    sent = pd.Series(data,name='Sent')
    exp = pd.Series(exps,name='Expressions')
    label = pd.Series(labels,name='Labels')
    dt = pd.Series(out,name='Dominant_Topic')
    
    Topics = pd.concat([dt,sent,label,exp],axis=1)
    '''
    return Topics

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

    
def BERTTopicAnalysis():
    dataset = pickle.load(open("../Data/ID/vnics_dataset_full_ratio-split.pkl", "rb"), encoding='latin1')          
    data = [p['sent'].replace(' &apos;','\'') for p in dataset["train_sample"]+dataset["test_sample"]]
    labels = [p['lab_int'] for p in dataset["train_sample"]+dataset["test_sample"]]
    expressions = [p['verb']+' '+p['noun'] for p in dataset["train_sample"]+dataset["test_sample"]]
    print('Number of samples',len(data))
    
    merger = Merger(lc=100000,sc=1,tc=100)
        
    #BERT11Analyser = Analyser(5,'BERT',11)
    #BERT8Analyser = Analyser(5,'BERT',8)
    #BERT0Analyser = Analyser(5,'BERT',0)
    #GloveAnalyser = Analyser(5,'Glove')

    #TableBERT11,TableBERT8,TableBERT0,TableGlove = {},{},{},{}
    #defaultdict(list)
    TableBERT11 = defaultdict(lambda: defaultdict(list))
    TableBERT10 = defaultdict(lambda: defaultdict(list))
    TableBERT9 = defaultdict(lambda: defaultdict(list))
    TableBERT8 = defaultdict(lambda: defaultdict(list))
    TableBERT7 = defaultdict(lambda: defaultdict(list))
    TableBERT6 = defaultdict(lambda: defaultdict(list))
    TableBERT5 = defaultdict(lambda: defaultdict(list))
    TableBERT4 = defaultdict(lambda: defaultdict(list))
    TableBERT3 = defaultdict(lambda: defaultdict(list))
    TableBERT2 = defaultdict(lambda: defaultdict(list))
    TableBERT1 = defaultdict(lambda: defaultdict(list))
    TableBERT0 = defaultdict(lambda: defaultdict(list))
    TableGlove = defaultdict(lambda: defaultdict(list))
    TableRand = defaultdict(lambda: defaultdict(list))

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
    GloveAnalyser = Analyser(5,'Glove')
    RandAnalyser = Analyser(5,'Rand')
    

    #for mp in range(10,51,5):
    for mp in range(5,11,5):
        #topic_model = BERTopic(min_topic_size=mp)
        #Topics = BERTopicModel(data,expressions,labels,topic_model)
        Topics = LSI(data,expressions,labels,mp)

        MergedTopics = merger.Merge(Topics,m=None) 
        
        m = len(Topics['Dominant_Topic'].unique())
        for i,v in enumerate(BERT11Analyser.perTopicAnalysis(MergedTopics)): TableBERT11[m][i].append(v)
        for i,v in enumerate(BERT10Analyser.perTopicAnalysis(MergedTopics)): TableBERT10[m][i].append(v)
        for i,v in enumerate(BERT9Analyser.perTopicAnalysis(MergedTopics)): TableBERT9[m][i].append(v)
        for i,v in enumerate(BERT8Analyser.perTopicAnalysis(MergedTopics)): TableBERT8[m][i].append(v)
        for i,v in enumerate(BERT7Analyser.perTopicAnalysis(MergedTopics)): TableBERT7[m][i].append(v)
        for i,v in enumerate(BERT6Analyser.perTopicAnalysis(MergedTopics)): TableBERT6[m][i].append(v)
        for i,v in enumerate(BERT5Analyser.perTopicAnalysis(MergedTopics)): TableBERT5[m][i].append(v)
        for i,v in enumerate(BERT4Analyser.perTopicAnalysis(MergedTopics)): TableBERT4[m][i].append(v)
        for i,v in enumerate(BERT3Analyser.perTopicAnalysis(MergedTopics)): TableBERT3[m][i].append(v)
        for i,v in enumerate(BERT2Analyser.perTopicAnalysis(MergedTopics)): TableBERT2[m][i].append(v)
        for i,v in enumerate(BERT1Analyser.perTopicAnalysis(MergedTopics)): TableBERT1[m][i].append(v)
        for i,v in enumerate(BERT0Analyser.perTopicAnalysis(MergedTopics)): TableBERT0[m][i].append(v)
        for i,v in enumerate(GloveAnalyser.perTopicAnalysis(MergedTopics)): TableGlove[m][i].append(v)
        for i,v in enumerate(RandAnalyser.perTopicAnalysis(MergedTopics)): TableRand[m][i].append(v)

        #TableBERT8[len(Topics['Dominant_Topic'].unique())]=BERT8Analyser.perTopicAnalysis(MergedTopics)
        #TableBERT0[len(Topics['Dominant_Topic'].unique())]=BERT0Analyser.perTopicAnalysis(MergedTopics)
        #TableGlove[len(Topics['Dominant_Topic'].unique())]=GloveAnalyser.perTopicAnalysis(MergedTopics)
     
    print('BERT11')
    for m in TableBERT11:
        print(m,' '.join([str(statistics.mean(TableBERT11[m][v])) for v in TableBERT11[m]]))
        
    print('BERT10')
    for m in TableBERT10:
        print(m,' '.join([str(statistics.mean(TableBERT10[m][v])) for v in TableBERT10[m]]))
        
    print('BERT9')
    for m in TableBERT9:
        print(m,' '.join([str(statistics.mean(TableBERT9[m][v])) for v in TableBERT9[m]]))
    
    print('BERT8')
    for m in TableBERT8:
        print(m,' '.join([str(statistics.mean(TableBERT8[m][v])) for v in TableBERT8[m]]))
        
    print('BERT7')
    for m in TableBERT7:
        print(m,' '.join([str(statistics.mean(TableBERT7[m][v])) for v in TableBERT7[m]]))
        
    print('BERT6')
    for m in TableBERT6:
        print(m,' '.join([str(statistics.mean(TableBERT6[m][v])) for v in TableBERT6[m]]))
        
    print('BERT5')
    for m in TableBERT5:
        print(m,' '.join([str(statistics.mean(TableBERT5[m][v])) for v in TableBERT5[m]]))
        
    print('BERT4')
    for m in TableBERT4:
        print(m,' '.join([str(statistics.mean(TableBERT4[m][v])) for v in TableBERT4[m]]))
        
    print('BERT3')
    for m in TableBERT3:
        print(m,' '.join([str(statistics.mean(TableBERT3[m][v])) for v in TableBERT3[m]]))
        
    print('BERT2')
    for m in TableBERT2:
        print(m,' '.join([str(statistics.mean(TableBERT2[m][v])) for v in TableBERT2[m]]))
        
    print('BERT1')
    for m in TableBERT1:
        print(m,' '.join([str(statistics.mean(TableBERT1[m][v])) for v in TableBERT1[m]]))
        
    print('BERT0')
    for m in TableBERT0:
        print(m,' '.join([str(statistics.mean(TableBERT0[m][v])) for v in TableBERT0[m]]))
        
    print('Glove')
    for m in TableGlove:
        print(m,' '.join([str(statistics.mean(TableGlove[m][v])) for v in TableGlove[m]]))
        
    print('Rand')
    for m in TableRand:
        print(m,' '.join([str(statistics.mean(TableRand[m][v])) for v in TableRand[m]]))
    '''
    print('BERT11')
    for m in TableBERT11:
        print(m,' '.join([str(round(v,4)) for v in TableBERT11[m]]))
        
    print('BERT8')
    for m in TableBERT8:
        print(m,' '.join([str(round(v,4)) for v in TableBERT8[m]]))
        
    print('BERT0')
    for m in TableBERT0:
        print(m,' '.join([str(round(v,4)) for v in TableBERT0[m]]))
        
    print('Glove')
    for m in TableGlove:
        print(m,' '.join([str(round(v,4)) for v in TableGlove[m]]))
    '''
            
     

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, default=None, help='number of topics')
    parser.add_argument('--min_topic_size', type=int, default=5, help='minimum size of topic')
     
    args=parser.parse_args() 
    
    BERTTopicAnalysis()

#BERTTopicAnalysis()
#TM = TopicModel()


'''

sents = ['cat dog and mouse','mouse cat and cow','cash money property']
exps = ['exp1','exp2','exp3']
labels = [1,0,1]

T = TopicModel(sents,exps,labels,2)

print(T)
'''