

import pandas as pd
#from functools import reduce

import matplotlib.pyplot as plt

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import spacy

from nltk.corpus import stopwords
path = '../Data/BERTAnalysis/'

class TopicModel:
    def __init__(self):
        self.nlp = spacy.load('en', disable=['parser', 'ner'])
        self.stop_words = stopwords.words('english')
        self.stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
        
        
    '''Tokenize words and Clean-up text'''
    def sent_to_words(self,sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
    
    def lemmatization(self,texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = self.nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out
        
    def createDictCorpus(self,data):
        '''Tokenize words and Clean-up text'''
        data_words = list(self.sent_to_words(data))
        #print(data_words[:1])
        
        ''' Remove Stop Words from data_words'''
        data_words_nostops = [[word for word in simple_preprocess(str(doc)) if word not in self.stop_words] for doc in data_words]
        #print(data_words_nostops[:1])
        
        ''' Form Bigrams'''
        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
        self.bigram_mod = gensim.models.phrases.Phraser(bigram)
        data_words_bigrams = [self.bigram_mod[doc] for doc in data_words_nostops]
        #for i in range(10): print(data_words_bigrams[i])
        
        '''Lemmatize bigrams'''
        data_lemmatized = self.lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        
        '''create dictionary and corpus'''
        self.id2word = corpora.Dictionary(data_lemmatized)
        corpus = [self.id2word.doc2bow(text) for text in data_lemmatized]

        #for c in data_lemmatized: print(c)
        #for c in corpus: print(c)

        self.tfidfmodel = gensim.models.TfidfModel(corpus)
        
        #TODO: confirm this
        corpus = self.tfidfmodel[corpus]
        
        return corpus,data_lemmatized
        
    def train(self,data,num_topics):
        corpus,data_lemmatized = self.createDictCorpus(data)

        
        #self.lda_model = gensim.models.LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=self.id2word)
        #self.lda_model = gensim.models.LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=self.id2word,alpha=[0.000001]*num_topics)
        self.lda_model = gensim.models.LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=self.id2word,alpha='asymmetric')
        
        #lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word)
        #for topic_num in range(num_topics): print(topic_num, lda_model.show_topic(topic_num))
        
    def save(self):
        self.lda_model.save(path+'lda.model')
        self.tfidfmodel.save(path+'tfidf.model')
        self.bigram_mod.save(path+'bigram.model')
        self.id2word.save(path+'id2word.dat')
        
    def load(self):
        self.id2word = corpora.Dictionary.load(path+'id2word.dat')
        self.bigram_mod = gensim.models.phrases.Phraser.load(path+'bigram.model')
        self.tfidfmodel = gensim.models.TfidfModel.load(path+'tfidf.model')
        self.lda_model = gensim.models.LdaMulticore.load(path+'lda.model')
        
    def topicModel(self,data,expressions,labels):
        
        #self.train(data,num_topics)
        
        '''Tokenize words and Clean-up text'''
        data_words = list(self.sent_to_words(data))
        #print(data_words[:1])
        
        ''' Remove Stop Words from data_words'''
        data_words_nostops = [[word for word in simple_preprocess(str(doc)) if word not in self.stop_words] for doc in data_words]
        data_words_bigrams = [self.bigram_mod[doc] for doc in data_words_nostops]
        data_lemmatized = self.lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        corpus = [self.id2word.doc2bow(text) for text in data_lemmatized]
        
        #TODO: confirm this
        corpus = self.tfidfmodel[corpus]


        Topics = pd.DataFrame()
        for row in self.lda_model[corpus]:
            row = sorted(row, key=lambda x: (x[1]), reverse=True)  
            #print(row)
            #continue
            topic_num, prop_topic = row[0] # Get the Dominant topic, Perc Contribution and Keywords for each document
            wp = self.lda_model.show_topic(topic_num)
            topic_keywords = ", ".join([word for word, prop in wp])
            Topics=Topics.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
    
        Topics = pd.concat([Topics, pd.Series(data),pd.Series(expressions),pd.Series(labels)], axis=1)
        Topics.columns = ['Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords','Sent','Expressions','Labels']
        return self.merge(Topics)
     
    def merge(self,Topics):
        Topics = Topics.rename(columns={'Dominant_Topic':'Old_Topic'})
        topic_list = list(Topics['Old_Topic'].unique())
        
        nk = Topics.groupby(['Labels']).size()[0]*2/len(topic_list)
        pk = Topics.groupby(['Labels']).size()[1]*2/len(topic_list)

        tt = Topics.groupby(['Old_Topic','Labels']).size()
        Entry = []
        newtopicid = 0
        while len(topic_list)!=0:
            topic = topic_list.pop()
            
            req_p = pk-tt[topic][1] if 1 in tt[topic] else 0
            req_n = nk-tt[topic][0] if 0 in tt[topic] else 0
            
            minv = req_p+req_n
            mint = None
            for t in topic_list:
                nt = tt[t][0] if 0 in tt[t] else 0
                pt = tt[t][1] if 1 in tt[t] else 0
                if abs(req_p-pt)+abs(req_n-nt)<minv:
                    minv = abs(req_p-pt)+abs(req_n-nt)
                    mint = t
            if mint is not None:    topic_list.remove(mint)
            e = Topics.loc[(Topics['Old_Topic']==topic) | (Topics['Old_Topic']==mint)].reset_index()
            e = pd.concat([pd.Series([newtopicid]*len(e.index),name='Dominant_Topic'),e],axis=1)
            Entry.append(e)
            newtopicid+=1
        return pd.concat(Entry,ignore_index=True)
        
    def __Elbow_Solhouette(self):
        #TODO: implemet
        return
    
    '''Return a list of topic models and it's corresponding coherence values'''
    def compute_coherence_values(self, data, limit, start=2, step=3):
        #dictionary,corpus,lemmatizedtext = self.createDictCorpus(data)
        corpus,data_lemmatized = self.createDictCorpus(data)
        
        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            #model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
            model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=self.id2word)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=data_lemmatized, dictionary=self.id2word, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())
    
    
        #limit=40; start=2; step=6;
        x = range(start, limit, step)
        plt.plot(x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc='best')
        plt.show()
        return model_list, coherence_values 
