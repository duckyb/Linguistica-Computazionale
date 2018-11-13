# -*- coding: utf-8 -*-
# ============== DIPENDENZE ==============
import sys, codecs, nltk, re, math
from nltk import FreqDist, bigrams, trigrams
from tabulate import tabulate # libreria per output tabulari
# ==============   CLASSI   ==============
class Corpus: # creo classi Corpus, per ciasuna delle quali faccio le dovute analisi
    def __init__ (self, path, name):
        self.path = path # percorso del file
        self.name = name # nome del corpus
        with codecs.open(self.path, 'r', 'utf-8') as doc: # apro il documento
            self.raw = doc.read() # assegno a raw il mio testo
        self.pos_tag = [] # [ (part_of_speech, token), (...) ]
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
        phrases = sent_tokenizer.tokenize(self.raw.lower())
        tokenList = []
        for p in phrases:
            tokens = nltk.word_tokenize(p)
            tokenList += tokens
        self.pos_tag = nltk.pos_tag(tokenList)
        return tokenList

    def top20_no_punct(self):
        punct = re.compile(r'[^\w]') # punctuations
        return FreqDist([i for i in self.tokens if not punct.match(i)]).most_common(20)

    def top20_partsofspeech(self, t):
        return FreqDist([token for token, tag in self.pos_tag if tag.startswith(t)]).most_common(20)

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
        self.bigram_data['condit'].sort(key=getKey, reverse=True) # ordinamento decrescente
        self.bigram_data['condit'] = self.bigram_data['condit'][:10] # stacco le 10 più frequenti
        # J O I N E D
        pos_bigrams = list(bigrams([p for t, p in self.pos_tag]))
        pos_bigrams_fdist = nltk.FreqDist(pos_bigrams)
        pos_bigrams_set = set(pos_bigrams)
        for b in pos_bigrams_set:
            prob_joined = pos_bigrams_fdist[b]*1.0/len(self.tokens)*1.0
            self.bigram_data['joined'].append([b, prob_joined])
        self.bigram_data['joined'].sort(key=getKey, reverse=True)
        self.bigram_data['joined'] = self.bigram_data['joined'][:10]

    def namentity(self):
        # N A M E D   E N T I T Y
        tree = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(self.raw))) # ne_chunk necessita della suddivisione per frasi
        names = []
        for branch in tree:
            NE = ''
            if hasattr(branch, 'label'):
                if branch.label() in ['GPE']:
                    for leaf in branch.leaves():
                        NE = NE + ' ' + leaf[0]
                        names.append(NE)
        return FreqDist(names).most_common(20)

class Combine: # unisco determinati valori di due Corpus
    def __init__(self, foo, bar):
        self.tokens  = foo.tokens + bar.tokens # concatenazione di array
        self.tags    = foo.pos_tag + bar.pos_tag # pos_tag combinati
        self.top20NN = FreqDist([token for token, tag in self.tags if tag.startswith('NN')]).most_common(20) # (sost, freq) combinati
        self.nouns   = [n for n, fre in self.top20NN] # sost combinati
        self.adj     = [token for token, tag in self.tags if tag.startswith('JJ')] # aggettivi combinati
        self.fdistJJ = nltk.FreqDist(self.adj) # fdist aggettivi combinati
        self.bigrams = foo.bigram_data['bigrams'] + bar.bigram_data['bigrams'] # bigrammi combinati
        self.unique  = set(self.bigrams) # bigrammi senza duplicati
        self.small   = [i for i in self.unique if (i[1] in self.nouns) and (i[0] in self.adj)] # lista contenente solo i bigrammi che mi interessano
        self.fdistBI = nltk.FreqDist(self.small) # fdist bigrammi combinati                    
    
    def lmi_tables(self): # calcolo LMI dei bigrammi selezionati e produco tabelle
        headers = ['aggettivo', 'local mutual\ninformation']
        for n in self.top20NN: # per ciascuno dei 20 sostantivi
            records = []
            LMI = 0.0
            JJeLMI_Tuples = []
            for b in self.small: # controllo ogni bigramma (aggettivo, sostantivo) del testo
                if b[1]==n[0] and b[0] in self.adj: # se il sostantivo è quello che cerco
                    freq_NN = self.tokens.count(b[1])
                    freq_JJ = self.tokens.count(b[0])
                    freq_observed = self.fdistBI[b]
                    freq_expected = ((freq_JJ*1.0)*(freq_NN*1.0))/(len(self.tokens)*1.0)
                    LMI = (freq_observed*1.0)*math.log((freq_observed*1.0)/(freq_expected*1.0), 2)
                    JJeLMI_Tuples.append((b[0], LMI)) # genero una tupla JJ + LMI
            JJeLMI_Tuples.sort(key=getKey, reverse=True) # ho finito; ordino le tuple x LMI
            records = [e for e in JJeLMI_Tuples] # inserisco le nuove tuple nel record pronte per l'output
            print 'Sostantivo: '+ str(n[0]) +' - Occorrenze: '+ str(n[1])+'\n'
            print tabulate(records, headers, floatfmt=".2f"), '\n'

# S U P P O R T O
def getKey(e): 
    # per ordinare in base al secondo elemento di coppie
    # usato dalle classi: Corpus, Combine
    return e[1]

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
        records = [] # reset dei records da stampare
        if choice == 1: # tabella senza punteggiatura
            headers = ['top 20 token\n'+m.name, 'freq.', 'top 20 token\n'+f.name, 'freq.']
            for (a, b), (c, d) in zip(m.top20['noPunct'],f.top20['noPunct']):
                records.append([a, b, c, d])
            print '\n', tabulate(records, headers)
        elif choice == 2: # tabella aggettivi
            headers = ['top 20 AGGETTIVI\n'+m.name, 'freq.', 'top 20 AGGETTIVI\n'+f.name, 'freq.']
            for (a, b), (c, d) in zip(m.top20['adj'],f.top20['adj']):
                records.append([a, b, c, d])
            print '\n', tabulate(records, headers)
        elif choice == 3: # tabella verbi
            headers = ['top 20 VERBI\n'+m.name, 'freq.', 'top 20 VERBI\n'+f.name, 'freq.']
            for (a, b), (c, d) in zip(m.top20['verb'],f.top20['verb']):
                records.append([a, b, c, d])
            print '\n', tabulate(records, headers)
        elif choice == 4: # tabella top 10 parts of speech
            headers = ['top 10 POS_tags\n'+m.name, 'freq.', 'top 10 POS_tags\n'+f.name, 'freq.']
            for (a, b), (c, d) in zip(m.top10tags, f.top10tags):
                records.append([a, b, c, d])
            print '\n', tabulate(records, headers)
        elif choice == 5: # tabella trigrammi
            headers = ['top 10 trigrams\n'+m.name, 'freq.', 'top 10 trigrams\n'+f.name, 'freq.']
            for ((onem, twom, threem), freqm),((onef, twof, threef), freqf) in zip(m.top10trigrams, f.top10trigrams):
                records.append([(onem, twom, threem), freqm, (onef, twof, threef), freqf])
            print '\n', tabulate(records, headers)
        elif choice == 6: # probabilità congiunta
            headers = ['MASCHI\nbigrammi', 'prob.\ncongiunta', 'FEMMINE\nbigrammi', 'prob.\ncongiunta']
            for [bigram_a, freq_a], [bigram_b, freq_b] in zip(m.bigram_data['joined'],f.bigram_data['joined']):
                records.append([bigram_a, str(freq_a*100) + ' %', bigram_b, str(freq_b*100) + ' %'])
            print '\n', tabulate(records, headers, floatfmt=".2f") # l'ultimo parametro serve a visualizzare 2 valori decimali
        elif choice == 7: # probabilità condizionata
            headers = ['MASCHI\nbigrammi', 'prob.\ncondizionata', 'FEMMINE\nbigrammi', 'prob.\ncondizionata']
            for [bigram_a, freq_a], [bigram_b, freq_b] in zip(m.bigram_data['condit'],f.bigram_data['condit']):
                records.append([bigram_a, str(freq_a*100) + ' %', bigram_b, str(freq_b*100) + ' %'])
            print '\n', tabulate(records, headers)
        elif choice == 8: # sostantivi più frequenti combinati
            mf = Combine(m, f)
            mf.lmi_tables() # stampo tabelle della local mutual information su due Corpus combinati

        elif choice == 9: # 20 nomi propri di luogo più frequenti
            headers = ['MASCHI\nGPE', 'frequenza', 'FEMMINE\nGPE', 'frequenza']
            for (GPE1, F1),(GPE2, F2) in zip(m.namentity(), f.namentity()):
                records.append([GPE1, F1, GPE2, F2])
            print '\n', tabulate(records, headers)
        choice = input()
    return
# ==================================================================
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
files = ['TBM.txt', 'TBF.txt']
m = Corpus(files[0], 'travel blog maschi')
f = Corpus(files[1], 'travel blog femmine')
main()