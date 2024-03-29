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
    def __init__(self,emb_types, mstart=5, mstop=51, mstep=5, train0labels=False, num_folds=5, allembs=False, roberta=False):
        self.TM = TopicModel()
        self.emb_types = emb_types
        self.mstart = mstart
        self.mstop = mstop
        self.mstep = mstep
        self.train0labels = train0labels
        self.roberta = roberta
        
        self.num_folds = num_folds
        self.allembs = allembs
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
    
    def getTopicCounts(self, data, labels, expressions):
        for mp in range(self.mstart,self.mstop,self.mstep): 
            with warnings.catch_warnings():
                filterwarnings("ignore", category=FutureWarning)
                if self.train0labels:    self.TM.train([d for d,l in zip(data,labels) if l==0],mp)
                else:   self.TM.train(data,mp)
                
                Topics = self.TM.topicModel(data,expressions,labels)
                Topics = Topics.loc[Topics['Dominant_Topic']!=-1]
                init_topics = len(Topics['Dominant_Topic'].unique())            
                Labels = len(Topics['Labels'].unique())
                
                DomTopics = list(Topics['Dominant_Topic'].unique())
                TopicGroup = [Topics.loc[(Topics['Dominant_Topic']==dt)].reset_index() for dt in DomTopics]
                
                tails = [1 if len(g.groupby(['Labels']).size())!=Labels or min([g.groupby(['Labels']).size()[l] for l in range(Labels)])<5 else 0 for g in TopicGroup]
                
    
                Topics = self.tailReduction(Topics,Labels)
                topics_after_red = len(Topics['Dominant_Topic'].unique()) 
                print(mp, init_topics,len(tails)-sum(tails),sum(tails),topics_after_red)

        
    
    def printEntropy(self,Entropies):
        print('{0:<18} {1:<18} {2:<18} {3:<18}'.format('Category','Mean entropy', 'Min entropy', 'Max entropy'))
        for e in Entropies:
            print('{0:<18} {1:<18} {2:<18} {3:<18}'.format(e,round(mean(Entropies[e]),4), round(min(Entropies[e]),4), round(max(Entropies[e]),4)))
        avg_mean = round(mean([mean(Entropies[e]) for e in Entropies]),4)
        avg_min = round(mean([min(Entropies[e]) for e in Entropies]),4)
        avg_max = round(mean([max(Entropies[e]) for e in Entropies]),4)
        print('{0:<18} {1:<18} {2:<18} {3:<18}'.format('Average',avg_mean,avg_min,avg_max))
            
    
    def entropyAnalysis(self,data,labels,expressions):
        LabelEntropy = defaultdict(list)
        ExpressionEntropy = defaultdict(list)

        LiteralExpressionEntropy = defaultdict(list)
        IdiomaticExpressionEntropy = defaultdict(list)

        for mp in tqdm(range(self.mstart,self.mstop,self.mstep)): 
            with warnings.catch_warnings():
                filterwarnings("ignore", category=FutureWarning)
                
                if self.train0labels:    self.TM.train([d for d,l in zip(data,labels) if l==0],mp)
                else:   self.TM.train(data,mp)
            
            Topics = self.TM.topicModel(data,expressions,labels)
            Topics = Topics.loc[Topics['Dominant_Topic']!=-1]
            
            max_entropy = stats.entropy([1]*len(Topics['Dominant_Topic'].unique()))
            #print('Maximum entropy', max_entropy)
            
            tg = Topics.groupby(['Labels','Dominant_Topic']).size()
            for l in Topics['Labels'].unique():
                LabelEntropy[l].append(stats.entropy(tg[l])/max_entropy)
            
            tg = Topics.groupby(['Expressions','Dominant_Topic']).size()
            for e in Topics['Expressions'].unique():
                max_exp_entropy = stats.entropy([1]*len(Topics.loc[Topics['Expressions']==e]))  
                ExpressionEntropy[e].append(stats.entropy(tg[e])/min(max_entropy,max_exp_entropy))
            
            tg = Topics.groupby(['Expressions','Labels','Dominant_Topic']).size()
            for e in Topics['Expressions'].unique():
                max_lit_exp_entropy = stats.entropy([1]*len(Topics.loc[(Topics['Expressions']==e) & (Topics['Labels']==0)]))  
                max_idiom_exp_entropy = stats.entropy([1]*len(Topics.loc[(Topics['Expressions']==e) & (Topics['Labels']==1)]))  

                if 0 in tg[e] and max_lit_exp_entropy>0:
                    LiteralExpressionEntropy[e].append(stats.entropy(tg[e][0])/min(max_entropy,max_lit_exp_entropy))
                if 1 in tg[e] and max_idiom_exp_entropy>0:
                    IdiomaticExpressionEntropy[e].append(stats.entropy(tg[e][1])/min(max_entropy,max_idiom_exp_entropy))

        print('Label distributional entropies')
        self.printEntropy(LabelEntropy)

        print('Expression distributional entropies')
        self.printEntropy(ExpressionEntropy)
        
        print('Literal Expression distributional entropies')
        self.printEntropy(LiteralExpressionEntropy)
        
        print('Idiomatic Expression distributional entropies')
        self.printEntropy(IdiomaticExpressionEntropy)
        
        '''
        print('{0:<18} {1:<18} {2:<18} {3:<18}'.format('Expression','Mean literal-entropy', 'Min literal-entropy', 'Max literal-entropy'))
        for e in LiteralExpressionEntropy:  
            print('{0:<18} {1:<18} {2:<18} {3:<18}'.format(e,round(mean(LiteralExpressionEntropy[e]),4), round(min(LiteralExpressionEntropy[e]),4), round(max(LiteralExpressionEntropy[e]),4)))
        avg_mean = round(mean([mean(LiteralExpressionEntropy[e]) for e in LiteralExpressionEntropy]),4)
        avg_min = round(mean([min(LiteralExpressionEntropy[e]) for e in LiteralExpressionEntropy]),4)
        avg_max = round(mean([max(LiteralExpressionEntropy[e]) for e in LiteralExpressionEntropy]),4)
        print('{0:<18} {1:<18} {2:<18} {3:<18}'.format('Average',avg_mean,avg_min,avg_max))
            
        print()
        print('{0:<18} {1:<18} {2:<18} {3:<18}'.format('Expression','Mean idiom-entropy', 'Min idiom-entropy', 'Max idiom-entropy'))
        for e in IdiomaticExpressionEntropy:  
            print('{0:<18} {1:<18} {2:<18} {3:<18}'.format(e,round(mean(IdiomaticExpressionEntropy[e]),4), round(min(IdiomaticExpressionEntropy[e]),4), round(max(IdiomaticExpressionEntropy[e]),4)))
        avg_mean = round(mean([mean(IdiomaticExpressionEntropy[e]) for e in IdiomaticExpressionEntropy]),4)
        avg_min = round(mean([min(IdiomaticExpressionEntropy[e]) for e in IdiomaticExpressionEntropy]),4)
        avg_max = round(mean([max(IdiomaticExpressionEntropy[e]) for e in IdiomaticExpressionEntropy]),4)
        print('{0:<18} {1:<18} {2:<18} {3:<18}'.format('Average',avg_mean,avg_min,avg_max))
        
        print()

        print('{0:<18} {1:<18} {2:<18} {3:<18}'.format('Label','Mean entropy', 'Min entropy', 'Max entropy'))
        for l in LabelEntropy:  
            print('{0:<18} {1:<18} {2:<18} {3:<18}'.format(l,round(mean(LabelEntropy[l]),4), round(min(LabelEntropy[l]),4), round(max(LabelEntropy[l]),4)))
            #print('Label:',l,':',mean(LabelEntropy[l]),min(LabelEntropy[l]),max(LabelEntropy[l]))
            
        print()
        print('{0:<18} {1:<18} {2:<18} {3:<18}'.format('Expression','Mean entropy', 'Min entropy', 'Max entropy'))
        for e in ExpressionEntropy:  
            print('{0:<18} {1:<18} {2:<18} {3:<18}'.format(e,round(mean(ExpressionEntropy[e]),4), round(min(ExpressionEntropy[e]),4), round(max(ExpressionEntropy[e]),4)))
            #print('Expression:',e,':',mean(ExpressionEntropy[e]),min(ExpressionEntropy[e]),max(ExpressionEntropy[e]))
        avg_mean = round(mean([mean(ExpressionEntropy[e]) for e in ExpressionEntropy]),4)
        avg_min = round(mean([min(ExpressionEntropy[e]) for e in ExpressionEntropy]),4)
        avg_max = round(mean([max(ExpressionEntropy[e]) for e in ExpressionEntropy]),4)
        print('{0:<18} {1:<18} {2:<18} {3:<18}'.format('Average',avg_mean,avg_min,avg_max))
        
        '''
    def getSeenUnseenScores(self,Data,embtype):
        model = MLPClassifier()
        self.createFolds(Data)

        Auc = [[0 for ts in range(len(self.Folds))] for tr in range(len(self.Folds))]
    
        for tr_topic in range(len(self.Folds)):
            ts_topic = tr_topic
    
            for test_fold in range(self.num_folds):
                trainX = [self.getEmb(sent,embtype) for fold in range(self.num_folds) for sent in self.Folds[tr_topic][fold]['Sent'] if fold!=test_fold]
                trainY = [label for fold in range(self.num_folds) for label in self.Folds[tr_topic][fold]['Labels'] if fold!=test_fold]
                with warnings.catch_warnings():
                    filterwarnings("ignore", category=ConvergenceWarning)
                    model.fit(trainX,trainY)
    
                for ts_topic in range(len(self.Folds)):
                    testX = [self.getEmb(sent,embtype) for sent in self.Folds[ts_topic][test_fold]['Sent']]
                    testY = [label for label in self.Folds[ts_topic][test_fold]['Labels']]
                    
                    if sum(testY)==0 or sum(testY)==len(testY): 
                        print(ts_topic,test_fold, sum(testY))
                        print(Data.groupby(['Dominant_Topic','Labels']).size()[ts_topic])
                    
                    if self.multiclass:
                        Auc[tr_topic][ts_topic] += roc_auc_score(testY,model.predict_proba(testX),multi_class='ovr')
                    else:
                        pred_proba = [proba[1] for proba in model.predict_proba(testX)]
                        Auc[tr_topic][ts_topic] += roc_auc_score(testY,pred_proba)
                             
        seen_score = [Auc[t][t]/self.num_folds for t in range(len(self.Folds))]
        unseen_score = [(sum(Auc[t]) - Auc[t][t]) / (self.num_folds * (len(self.Folds)-1)) for t in range(len(self.Folds))]
        
        #ttest = list(stats.ttest_rel(seen_score,unseen_score))    
        #scores = [[str(len(self.Folds))+'_'+str(i),seen,unseen,seen-unseen] for i, (seen,unseen) in enumerate(zip(seen_score,unseen_score))]
        
        return seen_score, unseen_score
    
    '''For each topic model calculate single seen score and single unseen score and then 
        take an average across different topic model. 
        To calculate the unseen score, for each trained probing model, 
            merge test fold from all k-1 unseen topics and evaluate it together.'''

    def probe(self,data,labels,expressions):
        '''Do a topic aware probing and print all the results. Input: 
                data: a list of sentences
                labels: a list of labels
                expressions: a list of expressions/type of input''' 
                
        n_class = len(set(labels))
        self.multiclass = False if n_class==2 else True
        
        Scores = {}
        E = [[emb+'_seen', emb+'_unseen'] for emb in self.emb_types]
        print(',,'+','.join([ee for e in E for ee in e]))
        #for mp in tqdm(range(self.mstart,self.mstop,self.mstep)):  
        for mp in range(self.mstart,self.mstop,self.mstep):  
            with warnings.catch_warnings():
                filterwarnings("ignore", category=FutureWarning)
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
                        
            SS = []
            for embtype in self.emb_types:
                #seen_score,unseen_score = self.getSeenUnseenScores_new(Topics,embtype)
                seen_score,unseen_score = self.getSeenUnseenScores(Topics,embtype)
                
                SS.append(zip(seen_score,unseen_score))
                #for ss,us in zip(seen_score,unseen_score):  print(embtype+','+str(mp)+','+str(ss)+','+str(us))
                
                if embtype+'_seen' not in Scores: Scores[embtype+'_seen']=[]
                Scores[embtype+'_seen'] += seen_score
                if embtype+'_unseen' not in Scores: Scores[embtype+'_unseen']=[]
                Scores[embtype+'_unseen'] += unseen_score
                
            for i,S in enumerate(zip(*SS)):
                S = [str(ss) for s in S for ss in s]
                print(str(mp)+','+str(i)+','+','.join(S))
                
        #print('{0:<8} {1:<8} {2:<8} {3:<8} {4:<8}'.format('Emb','Seen','Unseen','Diff','p-value'))
        print('{0:<8} {1:<8} {2:<8} {3:<8}'.format('Emb','Seen','Unseen','Diff'))
        for embtype in self.emb_types:
            mean_seen = mean(Scores[embtype+'_seen'])
            mean_unseen = mean(Scores[embtype+'_unseen'])
            diff = [a-b for a,b in zip(Scores[embtype+'_seen'],Scores[embtype+'_unseen'])]
            mean_diff = mean(diff)
            t,p = list(stats.ttest_rel(diff,[0]*len(diff),alternative='greater'))
            #print(embtype,mean_seen,mean_unseen,mean_diff,p)
            #print('{0:<8} {1:<8} {2:<8} {3:<8} {4:<8}'.format(embtype,round(mean_seen,4),round(mean_unseen,4),round(mean_diff,4),p))
            print('{0:<8} {1:<8} {2:<8} {3:<8}'.format(embtype,round(mean_seen,4),round(mean_unseen,4),round(mean_diff,4)))
        
        #TODO: remove p-value print
        
    def getSeenUnseenScores_alt(self,Data,embtype):
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
            
        return [[mean(topic_wise_seen_scores)],[mean(topic_wise_unseen_scores)]]

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
        if not hasattr(self,'emb'):   self.emb = Embedding(self.allembs,self.roberta) 
        if sent not in self.Embs:   self.Embs[sent] = self.emb.getMean(sent)
        return self.Embs[sent][embtype]
        
                      
#def BERTTopicAnalysis(task, mstart, mstop, mstep, size, alllayers, train0labels=False):
    
def stratified_sample_df(df, col, n_samples):
    n = min(n_samples, df[col].value_counts().min())
    df_ = df.groupby(col).apply(lambda x: x.sample(n))
    df_.index = df_.index.droplevel(0)
    return df_

def LoadVNC(size):
    dataset = pickle.load(open("./data/vnics_dataset_full_ratio-split.pkl", "rb"), encoding='latin1')          
    data = [p['sent'].replace(' &apos;','\'') for p in dataset["train_sample"]+dataset["test_sample"]]
    labels = [p['lab_int'] for p in dataset["train_sample"]+dataset["test_sample"]]
    expressions = [p['verb']+' '+p['noun'] for p in dataset["train_sample"]+dataset["test_sample"]]
    print('Loaded VNC dataset with')
    print('No. of samples:',len(data)) 
    print('No. of idioms:',sum(labels))
    print('No. of literals:',len(labels)-sum(labels))
    print('No. of idiomatic expressions:',len(set(expressions)))
    return data,labels,expressions 

def LoadFullVNC(size):        
    with open("./data/cook_dataset.tok", "r") as fp:
        dataset = fp.readlines()
    dataset = [d.strip().split('||') for d in dataset]
    dataset = [[d[0],d[1]+' '+d[2],d[3]] for d in dataset if d[0]!='Q']
    size = len(dataset) if size is None else size
    dataset = dataset[:size]
    labels,exps, data = zip(*dataset)
       
    labels = [0 if l == 'L' else 1 for l in labels]
    
    print('Loaded VNC dataset with')
    print('No. of samples:',len(data)) 
    print('No. of idioms:',sum(labels))
    print('No. of literals:',len(labels)-sum(labels))
    print('No. of idiomatic expressions:',len(set(exps)))
    
    return list(data),list(labels),list(exps)


def LoadProbingTask(task,size):
    df = pd.read_csv('./data/'+task+'.txt', sep='\t')
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
    parser.add_argument('--task', type=str, default='fullidiom', help='task name')
    parser.add_argument('--mstart', type=int, default=5, help='starting number of topics')
    parser.add_argument('--mstop', type=int, default=51, help='stoping number of topics')
    parser.add_argument('--mstep', type=int, default=5, help='stepping number of topics increment')
    parser.add_argument('--size', type=int, default=None, help='maximum data size limit. None for no limit')

    parser.add_argument('--alllayers', action='store_true', help='Analyse all 12 layers of BERT')
    parser.add_argument('--roberta', action='store_true', help='Analyse RoBERTa instead of BERT')

    parser.add_argument('--train0labels', action='store_true', help='Train topic model using 0 (literal in case of idiom task) labels')
    parser.add_argument('--entropyanalysis', action='store_true', help='Do the entropy analysis')
    parser.add_argument('--probe', action='store_true', help='Do the topic aware probing')
    parser.add_argument('--topiccount', action='store_true', help='Do the topic aware probing')
    
    args=parser.parse_args() 
    t0 = time.time()
    #if args.task=='idiom':   data,labels,expressions = LoadVNC(args.size)
    if args.task=='fullidiom':   data,labels,expressions = LoadFullVNC(args.size)
    else:   data,labels,expressions = LoadProbingTask(args.task,args.size)
    
    #print(len(data))
    embs = ['Glove','Rand']
    if args.roberta:
        embs += (['RoBERTa'+str(l) for l in range(12)] if args.alllayers else ['RoBERTa11'])
    else:
        embs += (['BERT'+str(l) for l in range(12)] if args.alllayers else ['BERT11'])
    
    ProbeModel = TopicAwareProbe(embs, args.mstart, args.mstop, args.mstep, args.train0labels, 5, args.alllayers, args.roberta)
    
    if args.entropyanalysis:    ProbeModel.entropyAnalysis(data,labels,expressions)
    if args.probe:  ProbeModel.probe(data,labels,expressions)
    if args.topiccount:  ProbeModel.getTopicCounts(data,labels,expressions)

    t = timedelta(seconds=time.time()-t0)
    #print(time.time()-t0)
    print(t)

    
