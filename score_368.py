import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# sub = pd.read_csv('sub/submission_prateek_0p60.csv')

# sub = pd.read_csv('sub/sub_xgb1d2.csv')
# sub = pd.read_csv('sub/sub_xgb1f1.csv')
# sub = pd.read_csv('sub/sub_xgb1f2.csv')
# sub = pd.read_csv('sub/sub_xgb1g1.csv')
# sub = pd.read_csv('sub/sub_xgb1g2.csv')
# sub = pd.read_csv('sub/sub_xgb1h1.csv')
# sub = pd.read_csv('sub/sub_xgb1i1.csv')
# sub = pd.read_csv('sub/sub_xgb1j1.csv')

# sub = pd.read_csv('sub/owl1_xgb.csv')
# sub = pd.read_csv('sub/owl1_xgb_fold_0.csv')
# sub = pd.read_csv('sub/owl2_xgb.csv')
# sub = pd.read_csv('sub/owl3_xgb.csv')
# sub = pd.read_csv('sub/owl4_xgb.csv')
# sub = pd.read_csv('sub/owl5_xgb.csv')
# sub = pd.read_csv('sub/owl6_xgb.csv')
# sub = pd.read_csv('sub/owl7_xgb.csv')
# sub = pd.read_csv('sub/owl8_xgb.csv')

# sub = pd.read_csv('sub/owl1_lgb.csv')
# sub = pd.read_csv('sub/owl2_lgb.csv')
sub = pd.read_csv('sub/owl3_lgb.csv')

# sub = pd.read_csv('sub/kevin1.csv')
# sub = pd.read_csv('sub/kevin2.csv')

# sub = pd.read_csv('sub/swanny1.csv')
# sub = pd.read_csv('sub/swanny2.csv')
# sub = pd.read_csv('sub/swanny3.csv')
# sub = pd.read_csv('sub/swanny4.csv')
# sub = pd.read_csv('sub/swanny5.csv')

# sub = pd.read_csv('sub/sub_nlp1j1.csv')
# sub = pd.read_csv('sub/sub_nlp1j2.csv')
# sub = pd.read_csv('sub/sub_nlp1k2.csv')
# sub = pd.read_csv('sub/sub_nlp2a1.csv')
# sub = pd.read_csv('sub/sub_nlp2c1.csv')

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


# m = tgt.merge(sub,on=['image_name'])

ma = np.array(tgt[['class','class1','class2','class3','class4',
    'class5','class6','class7','class8','class9']].values)

idx = ma[:,0].astype(int)

n = idx.shape[0]

logloss = np.zeros(n)

for c in [0,0.0008,0.001,0.002,0.003,0.004, 0.005,0.01]:
    for i in range(n):
        probs = ma[i,1:]
        probs = np.clip(probs,c,1.0-c)
        probs /= sum(probs)
        logloss[i] = -np.log(probs[idx[i]-1])
    
    print(c,logloss.mean())

tgt['logloss'] = logloss

