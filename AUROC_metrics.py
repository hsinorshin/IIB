from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import pandas as pd
import numpy as np
import matplotlib.pyplot at plt

Y_train=pd.read_csv("dataseteus/train/yelp_data.csv")["source"]
X_train=pd.read_csv("dataseteus/train/yelp_feature_train.csv")["binoScore"]
Y_train=np.where(Y_train=="Human",1,0)
# Ideally remove rows with NaN, below is only a temporary measure
#X_train.fillna(0.90,inplace=True)

fpr,tpr,thresholds=roc_curve(Y_train,X_train,pos_label=1)
roc_auc=auc(fpr,tpr)
# print('FPR',fpr[:10])
# print('TPR',tpr[:10])
# print('Threshold',thresholds)

display=RocCurveDisplay(fpr=fpr,tpr=tpr,roc_auc=roc_auc,estimator_name='Binoculars')
display.plot()
plt.show()
