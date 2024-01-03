import numpy as np
import re
from sentence_transformers import SentenceTransformer, util
import numpy as np
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

import nltk


def similarity(path, relations):
 if path is not None:
  rel_embedding = model.encode(relations, convert_to_tensor=True)
  path_embedding = model.encode(path, convert_to_tensor=True)
  top_k=1
  cos_scores = util.pytorch_cos_sim(path_embedding, rel_embedding)[0]
  top_results = np.argpartition(-cos_scores.cpu(), range(top_k))[0:top_k]
  for idx in top_results[0:top_k]:
    yh=0
    print(relations[idx], "(Score: %.4f)" % (cos_scores[idx]))

  return relations[idx],cos_scores[idx]
  #return relations[idx],cos_scores[idx]/10


import json
with open('rel_info.json') as f:
    data = f.read()
js = json.loads(data)

kv=[]
for key in js:
    kv.append(js[key])

entitylist=[]
with open("entity_mapping.txt","r") as file:
  for line in file:
   entity,_=line.strip().split("\t")
   entitylist.append(entity)
#print(entitylist)

filename="give context file here"
with open(filename,"r") as f:
 for line in f:
   try:
    print(line)
    e1,r,e2=line.strip().split("\t")
    print(e1,r,e2)
    first_ent,s1 =similarity(e1,entitylist)
    relation,s2 =similarity(r,kv)
    sec_ent,s3 =similarity(e2,entitylist)
    resolved=first_ent+"\t"+relation+"\t"+sec_ent
    #print("relation name:",r,"most similar::::::::::", most_similar_string)
    if(s1>0.6 and s2>0.6 and s3>0.6):
     with open(filename+"aligned_with_dataset.txt","a+") as foo:
        foo.write(resolved)
        foo.write("\n")
   except:
    pass
