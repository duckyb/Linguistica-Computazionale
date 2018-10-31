# -*- coding: utf-8 -*-

import sys, codecs, nltk, re, math
from nltk import FreqDist, bigrams, trigrams
from tabulate import tabulate # libreria per output tabulari

class Corpus: # creo un elemento di tipo Corpus, per ciasuno dei quali faccio le dovute analisi
    def __init__ (self, path, name):
        self.path = path # percorso del file
        self.name = name # nome del corpus
        self.pos_tag = [] # [ (part_of_speech, token) ]
        self.tokens = self.tokenized_text() # lista dei token
        self.top20 = {
            'noPunct': self.top20_no_punct(),
            'adj'    : self.top20_partsofspeech('JJ'),
            'verb'   : self.top20_partsofspeech('V'),
            'noun'   : self.top20_partsofspeech('NN'),
        }
        self.top10tags = self.t10t()
        self.top10trigrams = FreqDist(trigrams([b for (a, b) in self.pos_tag])).most_common(10)
        self.bigram_data = {
            'bigrams': list(bigrams(self.tokens)),
            'unique' : None, # lista di bigrammi senza duplicati
            'condit' : [],   # probabilità condizionata
            'joined' : [],   # probabilità congiunta
            'fdist'  : None, # distribuzione di frequenza dei bigrammi
        }
        self.bigram_data['unique'] = set(self.bigram_data['bigrams'])
        self.bigram_prob() # inizializza self.bigram_data['condit'] & self.bigram_data['joined']

    def tokenized_text(self):
        # T O K E N I Z Z A
        corpora = codecs.open(self.path, "r", "utf-8")
        raw = corpora.read()
        phrases = sent_tokenizer.tokenize(raw.lower())
        tokenList = []
        for p in phrases:
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
        
    def bigram_prob(self):
        # C O N D I T I O N A L
        for bigram in self.bigram_data['unique']:
            bigram_freq = self.bigram_data['bigrams'].count(bigram) # quante volte occorre un bigramma
            item1_freq = self.tokens.count(bigram[0]) # quante volte occorre il primo elemento
            prob_condit = bigram_freq*1.0 / item1_freq*1.0 # probabilità condizionata
            self.bigram_data['condit'].append([bigram, prob_condit]) # conservo ogni probabilità
        self.bigram_data['condit'].sort(key=self.getKey, reverse=True) # ordinamento decrescente
        self.bigram_data['condit'] = self.bigram_data['condit'][:10] # stacco le 10 più frequenti
        # J O I N E D
        pos_bigrams = list(bigrams([p for t, p in self.pos_tag]))
        pos_bigrams_fdist = nltk.FreqDist(pos_bigrams)
        pos_bigrams_set = set(pos_bigrams)
        for b in pos_bigrams_set:
            prob_joined = pos_bigrams_fdist[b]*1.0/len(self.tokens)*1.0
            self.bigram_data['joined'].append([b, prob_joined])
        self.bigram_data['joined'].sort(key=self.getKey, reverse=True)
        self.bigram_data['joined'] = self.bigram_data['joined'][:10]

    def LMI_rank(self):
        nouns_freq = self.top20['noun'][:-10] # 10 sostant. più freq.
        nouns_only = [n for n, f in nouns_freq]
        all_JJ = [token for token, tag in self.pos_tag if tag.startswith('JJ')] # lista di tutti gli aggettivi del testo
        fdist_JJ = nltk.FreqDist(all_JJ)
        self.bigram_data['fdist'] = nltk.FreqDist(self.bigram_data['bigrams'])
        # aumento le prestazioni del calcolo ottenendo un insieme di bigrammi composti da aggettivo + uno dei top 10 sostantivi
        smaller_bigr_set = [i for i in self.bigram_data['unique'] if (i[1] in nouns_only) and (i[0] in all_JJ)]
        result = []
        for n in nouns_freq: # per ogni sost
            result.append([n])
            JJeLMI_Tuples = []
            for b in smaller_bigr_set: # per ogni bigramma
                if b[1] == n[0]: # quando il bigramma contiene il sost
                    LMI = 0
                    mynoun_freq = next(freq for noun, freq in nouns_freq if noun == b[1])
                    LMI=(float(mynoun_freq)*self.log2((float(mynoun_freq)/float(len(self.tokens)))/(float(fdist_JJ[b[0]])/float(len(self.tokens))*(float(n[1])/float(len(self.tokens))))))
                    JJeLMI_Tuples.append(((b[0], LMI)))
            JJeLMI_Tuples.sort(key=self.getKey, reverse=True)
            result.append(JJeLMI_Tuples)
        return result

    # S U P P O R T O

    def getKey(self, e): # per ordinare in base al secondo elemento di coppie
        return e[1]
    
    def log2(self, x): # per calcolare log in base 2
        return math.log(x)/math.log(2)

def main():
    #  M E N U - S E L E Z I O N E
    choice = input('Premi:\n\
    1 per top 20 token punteggiatura esclusa\n\
    2 per top 20 aggettivi\n\
    3 per top 20 verbi\n\
    4 per top 10 PoS\n\
    5 per top 10 trigrammi\n\
    6 per probabilità congiunta\n\
    7 per probabilità condizionata\n\
    8 per top 10 sostantivi + agg.\n\
    9 TBD\n\
    0 per uscire\n')

    while choice != 0:
        if choice == 1: # tabella senza punteggiatura
            headers = ['top 20 token\n'+m.name, 'freq.', 'top 20 token\n'+f.name, 'freq.']
            records = []
            for (a, b), (c, d) in zip(m.top20['noPunct'],f.top20['noPunct']):
                records.append([a, b, c, d])
            print '\n', tabulate(records, headers)
        elif choice == 2: # tabella aggettivi
            headers = ['top 20 AGGETTIVI\n'+m.name, 'freq.', 'top 20 AGGETTIVI\n'+f.name, 'freq.']
            records = []
            for (a, b), (c, d) in zip(m.top20['adj'],f.top20['adj']):
                records.append([a, b, c, d])
            print '\n', tabulate(records, headers)
        elif choice == 3: # tabella verbi
            headers = ['top 20 VERBI\n'+m.name, 'freq.', 'top 20 VERBI\n'+f.name, 'freq.']
            records = []
            for (a, b), (c, d) in zip(m.top20['verb'],f.top20['verb']):
                records.append([a, b, c, d])
            print '\n', tabulate(records, headers)
        elif choice == 4: # tabella top 10 parts of speech
            headers = ['top 10 POS_tags\n'+m.name, 'freq.', 'top 10 POS_tags\n'+f.name, 'freq.']
            records = []
            for (a, b), (c, d) in zip(m.top10tags, f.top10tags):
                records.append([a, b, c, d])
            print '\n', tabulate(records, headers)
        elif choice == 5: # tabella trigrammi
            headers = ['top 10 trigrams\n'+m.name, 'freq.', 'top 10 trigrams\n'+f.name, 'freq.']
            records = []
            for ((onem, twom, threem), freqm),((onef, twof, threef), freqf) in zip(m.top10trigrams, f.top10trigrams):
                records.append([(onem, twom, threem), freqm, (onef, twof, threef), freqf])
            print '\n', tabulate(records, headers)
        elif choice == 6: # probabilità congiunta
            headers = ['MASCHI\nbigrammi', 'prob.\ncongiunta', 'FEMMINE\nbigrammi', 'prob.\ncongiunta']
            records = []
            for [bigram_a, freq_a], [bigram_b, freq_b] in zip(m.bigram_data['joined'],f.bigram_data['joined']):
                records.append([bigram_a, str(freq_a*100) + ' %', bigram_b, str(freq_b*100) + ' %'])
            print '\n', tabulate(records, headers)
            # NOTA: SISTEMARE OUTPUT
        elif choice == 7: # probabilità condizionata
            headers = ['MASCHI\nbigrammi', 'prob.\ncondizionata', 'FEMMINE\nbigrammi', 'prob.\ncondizionata']
            records = []
            for [bigram_a, freq_a], [bigram_b, freq_b] in zip(m.bigram_data['condit'],f.bigram_data['condit']):
                records.append([bigram_a, str(freq_a*100) + ' %', bigram_b, str(freq_b*100) + ' %'])
            print '\n', tabulate(records, headers)
        elif choice == 8: # sostantivi più frequenti
            print m.LMI_rank()
        elif choice == 9: # 20 nomi propri di luogo più frequenti
            pass # ! da fare
        choice = input('\nScegli un opzione (0 per uscire): ')
    return

# >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< ><
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
files = ['TBM.txt', 'TBF.txt']
m = Corpus(files[0], 'travel blog maschi')
f = Corpus(files[1], 'travel blog femmine')
main()

# [(u'regular', u'country'), (u'big', u'capital'), (u'invisible', u'capital'), (u'new', u'world'), (u'other', u'country'), (u'own', u'business'), (u'most', u'people'), (u'real', u'job'), (u'normal', u'job'), (u'exotic', u'country'), (u'online', u'business'), (u'asset\u2014time\u2014for', u'money'), (u'insane', u'people'), (u'primary', u'job'), (u'happy', u'people'), (u'only', u'country'), (u'other', u'people'), (u'many', u'people'), (u'independent', u'business'), (u'political', u'capital'), (u'first', u'country'), (u'else', u'i'), (u'amazing', u'country'), (u'middle', u'class'), (u'new', u'capital'), (u'comfortable', u'job'), (u'such', u'money'), (u'capital-rich', u'class'), (u'regular', u'people'), (u'anglo-saxon', u'country'), (u'own', u'capital'), (u'capitalistic', u'class'), (u'western', u'country'), (u'actual', u'business'), (u'enough', u'money'), (u'labor-providing', u'class'), (u'more', u'people'), (u'only', u'class'), (u'borderless', u'world'), (u'old', u'capital'), (u'9-5', u'job'), (u'crossfit', u'class'), (u'nomad', u'capital'), (u'capital-owned', u'class'), (u'new', u'class'), (u'new', u'business'), (u'polarizing', u'country'), (u'much', u'people'), (u'main', u'business'), (u'latin', u'america')]

# [(u'i', 64), [],
#  (u'class', 48), [],
#  (u'world', 42), [],
#  (u'capital', 30), [],
#  (u'people', 28), [],
#  (u'country', 26), [],
#  (u'money', 22), [],
#  (u'job', 20), [],
#  (u'america', 19), [(u'latin', 210.36415292339993)],
#  (u'business', 19), []]