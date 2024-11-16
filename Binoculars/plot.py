import enum
from operator import index
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from constants import DATASETS

def plot(subplot,x,y,labels,xlabel,ylabel,title):
    plt.xticks(range(0,4))
    plt.plot(x,list(zip(*y)),label=labels,marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


# # 'ppl' by dataset, by author
# for i,dataset in enumerate(DATASETS):
#     df=pd.read_csv('datasets/{}_by_author.csv'.format(dataset))
#     sources=list(set(df['source']))
#     df.set_index(['source','paraphraser','iteration'],inplace=True)
    
#     y=[]
#     for source in sources:
#         y.append(df.loc[[(source,'original',0),(source,'palm',1),(source,'palm',2),(source,'palm',3)],'ppl'])

    
#     plot(i+1,[0,1,2,3],y,sources,'iteration','ppl','Perplexity for {} paraphrased by PalM'.format(dataset))

author_count=defaultdict(lambda: [0 for _ in range(4)])
author_y=defaultdict(lambda: [0 for _ in range(4)])
for dataset in DATASETS:
    df_author=pd.read_csv('datasets/{}_by_author.csv'.format(dataset))
    df_data=pd.read_csv('datasets/{}_data.csv'.format(dataset))
    source_version_count = df_data.groupby(by=["source", "version_name"]).size()
    df_author.set_index(['source','paraphraser','iteration'],inplace=True)

    for (source,version),count in source_version_count.items():
        arr=version.split('_')
        paraphraser=arr[0]
        iteration=len(arr)
        if paraphraser=='original':
            author_count[source][0]+=count
            author_y[source][0]+=((df_author.loc[(source,'original',0),'ppl']/df_author.loc[(source,'original',0),'xppl'])*count)
        elif paraphraser=='palm':
            author_count[source][iteration]+=count
            author_y[source][iteration]+=((df_author.loc[(source,paraphraser,iteration),'ppl']/df_author.loc[(source,paraphraser,iteration),'xppl'])*count)

labels=[]
ys=[]
for author,y_arr in author_y.items():
    labels.append(author)
    ys.append([i/j for i,j in zip(y_arr,author_count[author])])

plot(1,[0,1,2,3],ys,labels,'Iteration','B','B score for each author, para:PalM')


plt.tight_layout()
plt.legend(loc='lower right')
plt.show()
    


