# BERTAnalysis
Analyse BERT embedding based approach

cleaned the BNC corpus using bnc.py

train the Topic Model using BNC corpus
'''python
python main.py --train --n number_of_bnc_files --num_topics number_of_topics
'''
 include --use_vnic to use vnic data set for training along with bnc dataset

Identify topics of sentences in VNIC dataset using already trained topic model
'''python
python main.py --test
'''

Do a five fold cross validation of per topic(merged) idiom detection

'''python
python main.py --idiomtest
'''

BERTopic

'''pip install bertopic'''

'''python
python main.py --bertopic
'''

