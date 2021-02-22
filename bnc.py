#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 10:50:58 2021

@author: vasu
"""


#import xml.etree.ElementTree as ET
#mytree = ET.parse('../Data/BNC/Texts/A/A0/A00.xml')
#myroot = mytree.getroot()
#for child in myroot:    print(child.tag,child.attrib)
#for child in myroot[1]:    print(child.attrib)
#tags = set([n.tag for n in myroot.iter('*')])
#text = ''.join(['\n' if neighbour.tag!='w' and neighbour.tag!='c' else neighbour.text for neighbour in myroot.iter('*')])
#print(text)
#print(tags)
#for n in myroot.iter('*'):  print(n.text,n.tag)
'''
for child in myroot:    
    print(child.text,child.tag)
    if child.tag=='wtext':
        for ch in child:
            print(ch.text,ch.tag,ch.attrib)
            for c in ch:
                print(c.text,c.tag,c.attrib)
'''
'''
wtext = myroot[-1]
for div in wtext:
    for head in div:
        for e in head:
            #print(e.tag,'' if e.text is None else e.text)
            print(e.tag)
        print('Next head',head.tag,head.attrib)
    print('Next div')
#print(myroot[wtext])
    
'''

from collections import namedtuple
import xml.etree.ElementTree as ET
 
"""Represents all of the info about a single word occurrence"""
Word = namedtuple('Word', ['div1', 'div2', 'div3', 'div4', 'sentence',
        'word', 'c5', 'hw', 'pos', 'text'])
 
class BNCParser(object):
    """A parser for British National Corpus (BNC) files"""
    def __init__(self, parser=None):
        if parser is None:
            parser = ET.XMLParser(encoding = 'utf-8')
        self.parser =  parser
 
    def parse(self, filename):
        """Parse `filename` and yield `Word` objects for each word"""
        tree = ET.parse(filename, self.parser)
        root = tree.getroot()
        divs = [None, 0, 0, 0, 0] # 1-based
        sentence = 0
        word = 0
        for neighbour in root.iter('*'):
            if neighbour.tag == 'div':
                level = int(neighbour.attrib['level'])
                divs[level] += 1
                # Reset all lower-level divs to 0
                for i in range(level + 1, 5):
                    divs[i] = 0
                sentence = 0
            elif neighbour.tag == 's':
                sentence += 1
                word = 0
            elif neighbour.tag == 'w':
                word += 1
                yield Word(divs[1], divs[2], divs[3], divs[4], sentence, word, 
                        neighbour.attrib['c5'], neighbour.attrib['hw'], 
                        neighbour.attrib['pos'], neighbour.text)
                
    def getText(self,filename):        
        #tree = ET.parse(filename, self.parser)
        tree = ET.parse(filename)
        root = tree.getroot()
        txt = ''
        for neighbour in root.iter('*'):
            
            if neighbour.tag == 'div' or neighbour.tag == 'p':
                txt+='\n'

            elif neighbour.tag == 's':
                txt+=' '
                
            elif (neighbour.tag == 'w' or neighbour.tag == 'c') and neighbour.text is not None:  txt+=neighbour.text
        return txt
 
              
source = '../Data/BNC/Texts/A/A0/A00.xml'
parser = BNCParser()
#for word in parser.parse(source):
#    print(word[-1])
    
#txt = ' '.join([word[-1] for word in parser.parse(source)])
#print(txt)
import glob

for filename in glob.iglob('../Data/BNC/Texts/' + '**/**', recursive=True):
    if not filename.endswith('xml'):    continue
    print(filename)
    text = parser.getText(filename)
    with open('../Data/BNC/ParsedTexts/'+filename.split('/')[-1]+'.txt','w') as fp:
        fp.write(text)
    #break
