from nltk.corpus import wordnet

import nltk
nltk.download('wordnet')


def get_wordnet_description(word):
    synsets = wordnet.synsets(word)
    if not synsets:
        return None
    return synsets[0].definition()



            
def get_synonyms(word):
    synonyms = []
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            synonyms.append(lemma.name())
    return synonyms

def get_aliases(word):
    aliases = []
    for synset in wordnet.synsets(word):
        for name in synset.lemma_names():
            aliases.append(name)
    return aliases

description = get_wordnet_description("Assembly")
print("DESCRIPTION:",description)

synn=get_synonyms("Assembly")
alias=get_aliases("Assembly")
print("SYNONYM: ",synn,"ALIAS:",alias)
delimiter = "________"

f=open("entity_context.txt","w")  
with open("entity_id", "r") as file:   # write entity2id file here
    for line in file:
            line = line.strip()
            #synn=get_synonyms(line)
            #alias=get_aliases(line)

            synn=get_synonyms("US")
            alias=get_aliases("US")
           
            for s in synn:
                f.write(line+"\t"+"hasSynonym"+"\t"+s+"\n")
            for al in alias:
              f.write(line+"\t"+"hasAlias"+"\t"+al+"\n")

