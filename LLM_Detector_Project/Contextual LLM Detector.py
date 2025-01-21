from operator import contains
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

from sklearn.metrics import f1_score, roc_curve, RocCurveDisplay, auc, confusion_matrix, precision_recall_curve

name = "Neural Net",
clf=MLPClassifier(alpha=1, max_iter=1000, random_state=42)

# Filepaths of features
feature_fp="dataseteus/MAGE_train_style.csv"
bino_fp="dataseteus/MAGE_train_bino.csv"
LIWC_fp="dataseteus/MAGE_train_LIWC.csv"

feature_df=pd.read_csv(feature_fp)
bino_df=pd.read_csv(bino_fp)
LIWC_df=pd.read_csv(LIWC_fp)

# Copy to fine-tune Binoculars threhold later
df_tune=bino_df

X_train=pd.concat([feature_df,bino_df['ppl'],LIWC_df],axis=1)
Y_train=bino_df['src']
Y_train=[src.split('_')[0] for src in Y_train]

# Train text domain classifier 
clf = make_pipeline(SimpleImputer(strategy="mean"), QuantileTransformer(), clf)
clf.fit(X_train, Y_train)

# Testing on SOT 
DATASETS_SOT=['xsum','sci_gen','tldr','eli5','wp','yelp','cmv']
pred=pd.Series()
for dataset in DATASETS_SOT:
    data_fp = "dataseteus/train/{}_data.csv".format(dataset)
    bino_fp = "dataseteus/train/{}_feature_train.csv".format(dataset)
    LIWC_fp = "dataseteus/train/{}_LIWC_train.csv".format(dataset)

    data_df = pd.read_csv(data_fp)
    bino_df = pd.read_csv(bino_fp)
    LIWC_df = pd.read_csv(LIWC_fp)
   
    # Filtered for original text
    indices = data_df.index[(data_df["version_name"]=="original")].tolist()
    data_df = data_df.iloc[indices]
    bino_df = bino_df.iloc[indices]
    LIWC_df = LIWC_df.iloc[indices]

    if pred.empty:
        pred=bino_df['binoScore']
        actual=data_df['source']
        X_test=pd.concat([bino_df.drop('binoScore',axis=1),data_df['ppl'],LIWC_df],axis=1)
   
    else:
        pred=pd.concat([pred,bino_df['binoScore']],ignore_index=True)
        actual=pd.concat([actual,data_df['source']],ignore_index=True)
        X_test=pd.concat([X_test,pd.concat([bino_df.drop('binoScore',axis=1),data_df['ppl'],LIWC_df],axis=1)],ignore_index=True)

actual = pd.Series(np.where(actual=='Human',1,0))

# Predict text domain of test data
pred_sources = pd.Series(clf.predict(X_test))

for i,dataset in enumerate(['cmv','tldr','sci','eli5','wp','yelp','xsum',’squad’]):

    # Index of test data labelled with current domain
    indexes = pred_sources.index[(pred_sources.str.contains(dataset))].tolist()

    # Using training data to find best bino threshold for the dataset
    tmp_tune=df_tune[df_tune['src'].str.contains(dataset)]
    X_tune=tmp_tune['binoScore']
    Y_tune=tmp_tune['label']

    f1_array=[]
    # Search around global accuracy threshold, 1 is 'Human', 0 is 'AI'
    for t in np.linspace(0.9015310749276843-0.05,0.9015310749276843+0.05,1000):
        Y_pred=[1 if score>t else 0 for score in X_tune]
        f1_array.append(f1_score(Y_tune,Y_pred,pos_label=0))

    Y_pred_ori=[1 if score>0.9015310749276843 else 0 for score in pred.iloc[indexes]]
    print(dataset)
    print('Global accuracy threshold f1 = ', f1_score(actual.iloc[indexes].tolist(),Y_pred_ori,pos_label=0))

    # Find tuned threshold
    threshold=np.linspace(0.9015310749276843-0.05,0.9015310749276843+0.05,1000)[np.argmax(f1_array)]
    print('New threshold is',threshold)
    print('F1 with new threshold',f1_score(actual.iloc[indexes].tolist(),[1 if score>threshold else 0 for score in pred.iloc[indexes]],pos_label=0))
    print('Size of dataset = ',len(indexes))
