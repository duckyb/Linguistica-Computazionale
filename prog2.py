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

# S U P P O R T O

def getKey(e): # per ordinare in base al secondo elemento di coppie
    return e[1]
    
def log2(x): # per calcolare log in base 2
    return math.log(x)/math.log(2)

def main():
    #  M E N U - S E L E Z I O N E
    # choice = input('Premi:\n\
    # 1 per top 20 token punteggiatura esclusa\n\
    # 2 per top 20 aggettivi\n\
    # 3 per top 20 verbi\n\
    # 4 per top 10 PoS\n\
    # 5 per top 10 trigrammi\n\
    # 6 per probabilità congiunta\n\
    # 7 per probabilità condizionata\n\
    # 8 per top 10 sostantivi + agg.\n\
    # 9 TBD\n\
    # 0 per uscire\n')
    choice = 8

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
        elif choice == 8: # sostantivi più frequenti combinati

            comb = { # combino elementi di m ed f
                'tokens' : [], # token combinati
                'tags'   : [], # pos_tag combinati
                'top20NN': [], # (sost, freq) combinati
                'nouns'  : [], # sost combinati
                'adj'    : [], # aggettivi combinati
                'fdistJJ': [], # fdist aggettivi combinati
                'bigrams': [], # bigrammi combinati
                'fdistBI': [], # fdist bigrammi combinati
                'unique' : [], # bigrammi senza duplicati
            }
            comb['tokens'] = m.tokens
            comb['tokens'].extend(f.tokens)
            comb['tags'] = m.pos_tag
            comb['tags'].extend(f.pos_tag) 
            NN_list = []
            for token, tag in comb['tags']:
                if tag.startswith('NN'):
                    NN_list.append(token)
            comb['top20NN'] = FreqDist(NN_list).most_common(20)
            comb['nouns'] = [n for n, fre in comb['top20NN']]
            comb['adj'] = [token for token, tag in comb['tags'] if tag.startswith('JJ')]
            comb['fdist_JJ'] = nltk.FreqDist(comb['adj'])
            comb['bigrams'] = m.bigram_data['bigrams']
            comb['bigrams'].extend(f.bigram_data['bigrams'])
            comb['fdistBI'] = nltk.FreqDist(comb['bigrams'])
            comb['unique'] = set(comb['bigrams'])
            smaller_bigr_set = [i for i in comb['unique'] if (i[1] in comb['nouns']) and (i[0] in comb['adj'])]
            result = []
            for n in comb['top20NN']: # per ogni sost
                result.append([n])
                JJeLMI_Tuples = []
                for b in smaller_bigr_set: # per ogni bigramma
                    if b[1] == n[0]: # quando il bigramma contiene il sost
                        LMI = 0
                        mynoun_freq = next(freq for noun, freq in comb['top20NN'] if noun == b[1])
                        LMI=(float(mynoun_freq)*log2((float(mynoun_freq)/float(len(comb['tokens'])))/(float(comb['fdist_JJ'][b[0]])/float(len(comb['tokens']))*(float(n[1])/float(len(comb['tokens']))))))
                        JJeLMI_Tuples.append(((b[0], LMI)))
                JJeLMI_Tuples.sort(key=getKey, reverse=True)
                result.append(JJeLMI_Tuples)
            print result


        elif choice == 9: # 20 nomi propri di luogo più frequenti
            pass # ! da fare
        choice = 0 # uscita automatica
    return

# >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< ><
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
files = ['TBM.txt', 'TBF.txt']
m = Corpus(files[0], 'travel blog maschi')
f = Corpus(files[1], 'travel blog femmine')
main()

# [[(u'i', 64)], [(u'else', 836.5950414261893)], [(u'class', 48)], [(u'capital-rich', 627.4462810696419), (u'labor-providing', 627.4462810696419), (u'crossfit', 627.4462810696419), (u'capital-owned', 627.4462810696419), (u'capitalistic', 579.4462810696419), (u'only', 483.4462810696418), (u'new', 449.82517459886947), (u'middle', 389.64485817107186)], [(u'world', 42)], [(u'borderless', 507.0154959359367), (u'new', 393.5970277740108)], [(u'capital', 30)], [(u'invisible', 362.1539256685262), (u'political', 362.1539256685262), (u'old', 362.1539256685262), (u'nomad', 344.60505064689147), (u'big', 332.1539256685262), (u'new', 281.14073412429343), (u'own', 254.60505064689153)], [(u'people', 28)], [(u'insane', 366.01033062395777), (u'regular', 366.01033062395777), (u'happy', 338.01033062395777), (u'most', 282.01033062395777), (u'many', 277.2524305835731), (u'much', 272.9963439671116), (u'more', 244.9963439671116), (u'other', 237.63138060376542)], [(u'country', 26)], [(u'regular', 339.8667355793894), (u'polarizing',
# 339.8667355793894), (u'amazing', 313.8667355793894), (u'anglo-saxon', 313.8667355793894), (u'western', 313.8667355793894), (u'exotic', 272.6577105606393), (u'first', 272.6577105606393), (u'only', 261.86673557938934), (u'other', 220.65771056063932)], [(u'money', 22)], [(u'asset\u2014time\u2014for', 287.57954549025254), (u'enough', 265.57954549025254), (u'such', 217.8411954585217)], [(u'job', 20)], [(u'primary', 261.43595044568417), (u'real', 221.43595044568414), (u'normal', 221.43595044568414), (u'9-5', 214.99738854793688), (u'comfortable', 209.73670043126103)], [(u'america', 19)], [(u'latin', 210.36415292339993)], [(u'business', 19)], [(u'online', 229.36415292339993), (u'main', 229.36415292339993), (u'independent', 210.36415292339993), (u'actual', 210.36415292339993), (u'new', 178.05579827871918), (u'own', 161.24986540969797)]]