# -*- coding: utf-8 -*-

import sys, codecs, nltk, re, math
from nltk import FreqDist, bigrams, trigrams
from tabulate import tabulate # libreria per output tabulari

class Corpus:
    def __init__ (self, path, name):
        self.path = path
        self.name = name
        self.pos_tag = []
        self.tokens = self.tokenized_text()
        self.top20 = {
            'noPunct': self.top20_no_punct(),
            'adj': self.top20_partsofspeech('JJ'),
            'verb': self.top20_partsofspeech('V')
        }
        self.top10tags = self.t10t()
        self.top10trigrams = FreqDist(trigrams([b for (a, b) in self.pos_tag])).most_common(10)
        self.bigram_data = {
            'bigrams': list(bigrams(self.tokens)),
            'unique': None,
            'condit' : None,
            'joined' : None
        }
        self.bigram_data['unique'] = set(self.bigram_data['bigrams'])
        self.bigram_data['condit'] = self.bigram_prob_condit()

    def tokenized_text(self): # tokenizza il testo
        corpora = codecs.open(self.path, "r", "utf-8")
        raw = corpora.read()
        phrases = sent_tokenizer.tokenize(raw.lower())
        tokenList = []
        for p in phrases:
            # tokens = nltk.word_tokenize(p.encode('utf8'))
            tokens = nltk.word_tokenize(p)
            tokenList += tokens
        corpora.close()
        self.pos_tag = nltk.pos_tag(tokenList)
        return tokenList

    def top20_no_punct(self):
        punct = re.compile(r'[^\w]') # lista di tutti i segni di punteggiatura
        tokenlist_noPunct = [i for i in self.tokens if not punct.match(i)]
        return FreqDist(tokenlist_noPunct).most_common(20)

    def top20_partsofspeech(self, t):
        a = []
        for token, tag in self.pos_tag:
            if tag.startswith(t):
                a.append(token)
        return FreqDist(a).most_common(20)

    def t10t(self):
        tuples = FreqDist(self.pos_tag).most_common()
        return nltk.FreqDist([b for (a, b), c in tuples]).most_common(10)

    def bigram_joined_prob(self):
        return

    def bigram_prob_condit(self):
        max_found = 0.0
        for big in self.bigram_data['unique']:
            bigram_freq = self.bigram_data['bigrams'].count(big)
            item1_freq = self.tokens.count(big[0])
            current_prob = bigram_freq*1.0 / item1_freq*1.0
            if current_prob > max_found:
                max_found = current_prob
                big_max = big
        return [big_max, max_found]

def main():
    # tabella senza punteggiatura
    headers = ['top 20 token\n'+m.name, 'freq.', 'top 20 token\n'+f.name, 'freq.']
    records = []
    for (a, b), (c, d) in zip(m.top20['noPunct'],f.top20['noPunct']):
        records.append([a, b, c, d])
    print '\n', tabulate(records, headers)
    # tabella aggettivi
    headers = ['top 20 AGGETTIVI\n'+m.name, 'freq.', 'top 20 AGGETTIVI\n'+f.name, 'freq.']
    records = []
    for (a, b), (c, d) in zip(m.top20['adj'],f.top20['adj']):
        records.append([a, b, c, d])
    print '\n', tabulate(records, headers)
    # tabella verbi
    headers = ['top 20 VERBI\n'+m.name, 'freq.', 'top 20 VERBI\n'+f.name, 'freq.']
    records = []
    for (a, b), (c, d) in zip(m.top20['verb'],f.top20['verb']):
        records.append([a, b, c, d])
    print '\n', tabulate(records, headers)
    # tabella top 10 parts of speech
    headers = ['top 10 POS_tags\n'+m.name, 'freq.', 'top 10 POS_tags\n'+f.name, 'freq.']
    records = []
    for (a, b), (c, d) in zip(m.top10tags, f.top10tags):
        records.append([a, b, c, d])
    print '\n', tabulate(records, headers)
    # tabella trigrammi
    headers = ['top 10 trigrams\n'+m.name, 'freq.', 'top 10 trigrams\n'+f.name, 'freq.']
    records = []
    for ((onem, twom, threem), freqm),((onef, twof, threef), freqf) in zip(m.top10trigrams, f.top10trigrams):
        records.append([(onem, twom, threem), freqm, (onef, twof, threef), freqf])
    print '\n', tabulate(records, headers)

    # probabilità congiunta
    # Note: da fare!

    # probabilità condizionata
    headers = ['corpus', 'bigramma', 'p.condizionata']
    records = [
        [m.name, m.bigram_data['condit'][0], m.bigram_data['condit'][1]],
        [f.name, f.bigram_data['condit'][0], f.bigram_data['condit'][1]]
        ]
    print '\n', tabulate(records, headers)
    

# >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< ><
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
files = ['TBM.txt', 'TBF.txt']
m = Corpus(files[0], 'travel blog maschi')
f = Corpus(files[1], 'travel blog femmine')
main()