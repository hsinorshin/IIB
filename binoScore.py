from binoculars import Binoculars
from datasets import load_dataset, Dataset
import pandas as pd
import sys
import torch
torch.cuda.empty_cache()

# batch processing
# batch=int(sys.argv[1])
# num_rows = 
# start=batch*100
# if start>=num_rows:
#     exit(0)
# end=min((batch+1)*50,num_rows)

# dataset, training set (80%) without dataset from ROC and HellaSWAG
df = load_dataset("yaful/MAGE", split='train')
df = df.filter(lambda row: not("roct" in row["src"] or "hswag" in row["src"]))

# split into 2626 shards, approx 100 texts each shard
df = df.shard(2626,int(sys.argv[1]),contiguous=True)

output_filepath="dataseteus/MAGE_train_bino.csv"

texts = df['text']

bino= Binoculars()
ppl, binoScore = bino.compute_ppl_bino(texts)

out_df=pd.DataFrame({'shard':[int(sys.argv[1])]*len(texts),'label':df['label'],'src':df['src'],'ppl': ppl, 'binoScore': binoScore})

if int(sys.argv[1])==0:
    out_df.to_csv(output_filepath,mode='a',header=True, encoding='utf-8')
else:
    out_df.to_csv(output_filepath,mode='a',header=False, encoding='utf-8')
