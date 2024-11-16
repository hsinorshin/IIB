from binoculars import Binoculars
import pandas as pd
import sys
import torch
torch.cuda.empty_cache()

batch=int(sys.argv[1])
filepath = "wp_paraphrased.csv"
output_filepath="wp_ppl_xppl.csv"

df = pd.read_csv(filepath)
num_rows = df.shape[0]
start=batch*100
if start>=num_rows:
    exit(0)

end=min((batch+1)*100,num_rows)

# columns = ['source','key','text','version_name','binoScore']

text_datas = list(df.iloc[start:end]['text'])

bino = Binoculars()
ppl,xppl=bino.compute_ppl_xppl(text_datas)

d={'ppl':ppl,'xppl':xppl}
output_df = pd.DataFrame(data=d,index=[i for i in range(start,end)])
output_df.to_csv(output_filepath,mode='a',header=False, encoding='utf-8')
