#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 11:03:58 2021

@author: vasu
"""
import pandas as pd

class Merger:
    def __init__(self,lc,sc,tc):
        self.lc = lc
        self.sc = sc
        self.tc = tc
        
    '''Calculate the score of the merged topic ith and jth topics'''
    def VarScore(self,i,j):
        ldi = [self.Topics[i].groupby(['Labels']).size()[l] for l in range(self.Labels)]
        ldi = [float(e)/sum(ldi) for e in ldi]
        ldi_var = sum([(a-b)*(a-b) for a,b in zip(self.gld,ldi)])/self.Labels
        
        ldj = [self.Topics[j].groupby(['Labels']).size()[l] for l in range(self.Labels)]
        ldj = [float(e)/sum(ldj) for e in ldj]
        ldj_var = sum([(a-b)*(a-b) for a,b in zip(self.gld,ldj)])/self.Labels

        
        ldij = [self.Topics[i].groupby(['Labels']).size()[l]+self.Topics[j].groupby(['Labels']).size()[l] for l in range(self.Labels)]
        ldij = [float(e)/sum(ldij) for e in ldij]        
        ldij_var = sum([(a-b)*(a-b) for a,b in zip(self.gld,ldij)])/self.Labels
        
        '''max: 1, min 0, avg: 0.5'''
        ld_var = ldij_var - ldi_var - ldj_var
        
        '''max: N^2 / 4, min 1, avg: N^2 / 8 ~10^5'''
        size = len(self.Topics[i].index)*len(self.Topics[j].index)
        
        '''max: T^2 / 4, min 1, avg: T^2 / 8 ~10^3'''
        topic_size = len(self.Topics[i]['Dominant_Topic'].unique())*len(self.Topics[j]['Dominant_Topic'].unique())

        #size = len(self.Topics[i].index)+len(self.Topics[j].index)
        #topic_size = len(self.Topics[i]['Dominant_Topic'].unique())+len(self.Topics[j]['Dominant_Topic'].unique())
        
        return self.lc * ld_var + self.sc * size + self.tc * topic_size
    
    def reduceTopics(self,m):
        while len(self.Topics)>m:
            #select pair of dataframes in self.Topics with minimum score
            min_score = 1e+20
            for i in range(len(self.Topics)):
                for j in range(len(self.Topics)):
                    if i==j:    continue
                    score = self.VarScore(i,j)
                    if score < min_score: min_i,min_j,min_score = i,j,score
                    
            '''combine min_ith and min_jth dataframes'''                      
            newdf = pd.concat([self.Topics[min_i],self.Topics[min_j]],ignore_index=True)
            del self.Topics[max(min_i,min_j)]
            del self.Topics[min(min_i,min_j)]
            self.Topics.append(newdf)
            
    def getTail(self):
        for i in range(len(self.Topics)):
            if len(self.Topics[i].groupby(['Labels']).size())!=self.Labels:    return i
            if min([self.Topics[i].groupby(['Labels']).size()[l] for l in range(self.Labels)])<5:   return i
        return None
            
    def Merge(self,T,m=None):
        self.Labels = len(T['Labels'].unique())
        
        self.gld = []
        for l in range(self.Labels):
            self.gld.append(T.groupby(['Labels']).size()[l])
        self.gld = [float(e)/sum(self.gld) for e in self.gld]
        #print('global label distribution:',self.gld)
        
        self.Topics = []
        for dt in list(T['Dominant_Topic'].unique()):
            self.Topics.append(T.loc[(T['Dominant_Topic']==dt)].reset_index())
            
        #print('Number of initial topics:',len(self.Topics))         
            
        while True:
            i = self.getTail()
            if i is None:   break
            df1 = self.Topics[i]
            del self.Topics[i]
            
            j = self.getTail()
            if j is None:   
                self.Topics[-1] = pd.concat([self.Topics[-1],df1],ignore_index=True)
                break
            df2 = self.Topics[j]
            del self.Topics[j]

            self.Topics.append(pd.concat([df1,df2],ignore_index=True))
        
        #print('Number of topics after tail reduction:',len(self.Topics))
        m = len(self.Topics) if m is None else m
        self.reduceTopics(m)
        #print('Number of sents after reduceTopics',len(pd.concat(self.Topics,ignore_index=True).index))
        for i in range(len(self.Topics)):
            self.Topics[i]=self.Topics[i].rename(columns={'Dominant_Topic':'Old_Topic'})
            self.Topics[i].insert(0,'Dominant_Topic',[i]*len(self.Topics[i].index))
            
        return pd.concat(self.Topics,ignore_index=True)
     