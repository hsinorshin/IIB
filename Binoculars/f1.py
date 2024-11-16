import string
import pandas as pd
from collections import defaultdict
import statistics as s
import math
from operator import add

from constants import BINOCULARS_ACCURACY_THRESHOLD,DATASETS

#modes=['traditional','alternative]


def getList(df:pd.DataFrame,col:string) -> list:
    return df[col].tolist()

def col_by_key(key,col,data_filepath,metric='mean') -> pd.DataFrame:
    df=pd.read_csv(data_filepath)
    
    if metric=='mean':
        return df.groupby(key)[col].mean()

    elif metric=='sum':
        return df.groupby(key)[col].sum()
    
    else:
        return 'Give valid metric pls'
        
 
 
for dataset in DATASETS:
    df= col_by_key(['source','version_name'],['ppl','xppl'],'datasets/{}_data.csv'.format(dataset))

    out_filepath='datasets/{}_by_author.csv'.format(dataset)
    out_df=pd.DataFrame(columns=['source','paraphraser','iteration','ppl','xppl'])

    for index,row in df.iterrows():
        source,arr=index
        arr=arr.split('_')
        if arr[0]=='original':
            paraphraser,iter='original',0
        else:
            paraphraser,iter=arr[0],len(arr)

        if math.isnan(out_df.index.max()):
            out_df.loc[0]=[source,paraphraser,iter]+row.tolist()
        else:
            out_df.loc[out_df.index.max()+1]=[source,paraphraser,iter]+row.tolist()

        out_df.to_csv(out_filepath,index=False)

   

  


def f1_mean_var_table(authors,binoScores,versions,mode='traditional',threshold=BINOCULARS_ACCURACY_THRESHOLD)->pd.DataFrame:
    #authors=set(authors_arr)
    versions=[version.split('_') for version in versions]
    # paraphrasers=set(paraphrasers[-1] for paraphrasers in versions)
    paraphraser_iter=[(paraphrasers[-1],len(paraphrasers)) for paraphrasers in versions]

    TP=defaultdict(int)
    FP=defaultdict(int)
    FN=defaultdict(int)
    bino_arr=defaultdict(list) #array of binoculars score for each paraphraser iteration
    

    for i,(author,binoScore,(paraphraser,iteration)) in enumerate(zip(authors,binoScores,paraphraser_iter)):
        if author!='Human' and binoScore<threshold:
            TP[f"{paraphraser}_{iteration}"]+=1
        elif author=='Human' and binoScore<threshold:
            FP[f"{paraphraser}_{iteration}"]+=1
        elif author!='Human' and binoScore>=threshold:
            FN[f"{paraphraser}_{iteration}"]+=1
        
        if not math.isnan(binoScore):
            bino_arr[f"{paraphraser}_{iteration}"].append(binoScore)

    table = pd.DataFrame(columns=['paraphraser', 'iteration', 'f1','performance drop from $T^{n-1}$', 'mean B', 'var B'])
    ori='original_1'
    f1_ori=TP[ori]/(TP[ori]+0.5*(FP[ori]+FN[ori]))
    mean_ori=s.mean(bino_arr[ori])
    var_ori=s.variance(bino_arr[ori])
    table.loc[0]=['original',0,f1_ori,'{}%'.format(0), '{0:.3g}'.format(mean_ori),'{0:.3g}'.format(var_ori)]

    for paraphraser,iteration in sorted(list(set(paraphraser_iter)),key=lambda x: (x[0], x[1])):
        if paraphraser!='original':
            index=f"{paraphraser}_{iteration}"
            f1= TP[index]/(TP[index] + 0.5*(FP[index]+FN[index]))
            if iteration==1:
                drop_percent=(f1_ori-f1)/f1_ori
            else:
                prev=table.loc[table.index.max(),'f1']
                drop_percent=(prev-f1)/prev

            mean = s.mean(bino_arr[index])
            var = s.variance(bino_arr[index])

            table.loc[table.index.max()+1]=[paraphraser,iteration,f1,'{0:.3g}%'.format(drop_percent*100),'{0:.3g}'.format(mean),'{0:.3g}'.format(var)]

        
    return table

def aggregate_f1(datasets:list[str]):
    total_f1=pd.Series() #array of summed f1*entry size for each para_iter
    total_entries=0 #total number of texts from all datasets for each para_iter
    for dataset in datasets:
        filepath="{}_paraphrased.csv".format(dataset)
        f1_filepath="{}_paraphrased_output.csv".format(dataset)
        df=pd.read_csv(filepath)
        f1_df=pd.read_csv(f1_filepath)
        entries=df.shape[0]
        if total_f1.empty:
            total_f1=entries*f1_df['f1']
        else:
            total_f1=pd.Series(map(add,total_f1,entries*f1_df['f1']))
        total_entries+=entries

    d={'paraphraser': f1_df['paraphraser'],'iteration':f1_df['iteration'], 'f1':total_f1/total_entries}
    table = pd.DataFrame(data=d)
    table.set_index(['paraphraser','iteration'],inplace=True)
    f1_drop=[0]
    f1_ori=table.loc[('original',0),'f1']
    for (paraphraser,iteration),row in table.iterrows():
        if paraphraser!='original':
            if iteration==1:
                drop_percent=(f1_ori-row['f1'])/f1_ori
            else:
                prev=table.loc[(paraphraser,iteration-1),'f1']
                drop_percent=(prev-row['f1'])/prev

            f1_drop.append(drop_percent*100)

    table['performance drop from $T^{n-1}$']=f1_drop
    print(table.to_markdown())


# for dataset in DATASETS:
#     df = pd.read_csv('{}_paraphrased.csv'.format(dataset))
#     table=f1_mean_var_table(getList(df,'source'),getList(df,'binoScore'),getList(df,'version_name'))
#     # pyperclip.copy(table.to_markdown(index=False))
#     table.to_csv('{}_paraphrased_output.csv'.format(dataset),index=False)

# aggregate_f1(DATASETS)