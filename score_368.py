import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

sub = pd.read_csv('sub/submission_prateek_0p60.csv')
tgt = pd.read_csv('test_variant_annotation_0711_tfidf.csv')

# subset to 368 rows from OncoKB
tgt = tgt.loc[tgt['class'].notnull()]

tgt = tgt.merge(sub,how='left',on=['ID'])

# sub.sort_values('image_name',inplace=True)
# tgt.sort_values('image_name',inplace=True)
# sub.reset_index(inplace=True)
# tgt.reset_index(inplace=True)
# print sub.head()
# print tgt.head()
# sub = sub.loc[0:0,:]
# tgt = tgt.loc[0:0,:]

print(log_loss(tgt['class'], tgt[ ['class1','class2','class3','class4',
    'class5','class6','class7','class8','class9'] ].values, 
    labels=[1,2,3,4,5,6,7,8,9]))

'''


m = tgt.merge(sub,on=['image_name'])

ma = np.array(m[['class','Type_1_y','Type_2_y','Type_3_y'] ].values)

idx = ma[:,0].astype(int)

n = idx.shape[0]

logloss = np.zeros(n)

for i in range(n):
    probs = ma[i,1:]
    probs /= sum(probs)
    logloss[i] = -np.log(probs[idx[i]-1])
    
print(logloss.mean())

m['logloss'] = logloss

'''
