from sklearn import *
import sklearn
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from scipy.special import logit, expit

# owl5 with lightgbm
# midx = '1'
# params = {
#     'learning_rate': 0.03333,
#     'max_depth': 4,
#     'objective': 'multiclass',
#     'metric': 'multi_logloss',
#     'num_class': 9,
#     'verbose': 0
# }

# midx = '2'
# params = {
#     'learning_rate': 0.03333,
#     'max_depth': 5,
#     'objective': 'multiclass',
#     'metric': 'multi_logloss',
#     'num_class': 9,
#     'verbose': 0
# }

# midx = '3'
# params = {
#     'learning_rate': 0.03333,
#     'max_depth': 6,
#     'objective': 'multiclass',
#     'metric': 'multi_logloss',
#     'num_class': 9,
#     'verbose': 0
# }

# midx = '4'
# params = {
#     'learning_rate': 0.03333,
#     'max_depth': 7,
#     'objective': 'multiclass',
#     'metric': 'multi_logloss',
#     'num_class': 9,
#     'verbose': 0
# }

# midx = '5'
# params = {
#     'learning_rate': 0.03333,
#     'max_depth': 8,
#     'objective': 'multiclass',
#     'metric': 'multi_logloss',
#     'num_class': 9,
#     'verbose': 0
# }

# midx = '6'
# params = {
#     'learning_rate': 0.03333,
#     'max_depth': 6,
#     'feature_fraction': 0.8, 
#     'bagging_fraction': 0.8,
#     'objective': 'multiclass',
#     'metric': 'multi_logloss',
#     'num_class': 9,
#     'verbose': 0
# }

# midx = '7'
# params = {
#     'learning_rate': 0.03333,
#     'max_depth': 6,
#     'feature_fraction': 0.7, 
#     'bagging_fraction': 0.9,
#     'objective': 'multiclass',
#     'metric': 'multi_logloss',
#     'num_class': 9,
#     'verbose': 0
# }

# midx = '8'
# params = {
#     'learning_rate': 0.03333,
#     'max_depth': 6,
#     'feature_fraction': 0.9, 
#     'bagging_fraction': 0.7,
#     'objective': 'multiclass',
#     'metric': 'multi_logloss',
#     'num_class': 9,
#     'verbose': 0
# }

# midx = '9'
# params = {
#     'learning_rate': 0.03333,
#     'max_depth': 6,
#     'feature_fraction': 0.9, 
#     'bagging_fraction': 0.9,
#     'objective': 'multiclass',
#     'metric': 'multi_logloss',
#     'num_class': 9,
#     'verbose': 0
# }

midx = '10'
params = {
    'learning_rate': 0.03333,
    'max_depth': 6,
    'feature_fraction': 0.7, 
    'bagging_fraction': 0.7,
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': 9,
    'verbose': 0
}

# params = {}
# params['max_bin'] = 10
# params['learning_rate'] = 0.01 # shrinkage_rate
# params['boosting_type'] = 'gbdt'
# params['objective'] = 'regression_l1'
# params['metric'] = 'mae'          # or 'mae'
# params['sub_feature'] = 0.5      # feature_fraction 
# params['bagging_fraction'] = 0.85 # sub_row
# params['bagging_freq'] = 40
# params['num_leaves'] = 512        # num_leaf
# params['min_data_in_leaf'] = 500         # min_data_in_leaf
# # params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
# params['min_hessian'] = 0.001     # min_sum_hessian_in_leaf
# params['verbose'] = 0
# params['feature_fraction_seed'] = 123
# params['bagging_seed'] = 123
# params['lambda_l1'] = 1
# params['lambda_l2'] = 0

# midx = '2'
# params = {
#     'eta': 0.01,
#     'max_depth': 5,
#     'subsample': 0.7,
#     'colsample_bytree': 0.5,
#     'colsample_bylevel': 1.0,
#     'min_child_weight': 1,
#     'alpha': 0.0,
#     'lambda': 1.0,
#     'gamma': 0.0,
#     'num_parallel_tree': 5,
#     'objective': 'multi:softprob',
#     'eval_metric': 'mlogloss',
#     'num_class': 9,
#     'seed': 1,
#     'silent': 1
# }


train = pd.read_csv('training_variants.txt')
test = pd.read_csv('test_variants.txt')
trainx = pd.read_csv('training_text.txt', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
testx = pd.read_csv('test_text.txt', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

# blup variable list
vw0 = ['Molecular.Genetics','Role.in.Cancer','Ref','Alt']
vw = ['Molecular_Genetics','Role_in_Cancer','Ref','Alt']

train_anno1 = pd.read_csv("train_variant_annotation_0720_tfidf.csv")
train_anno1 = train_anno1[['ID']+vw0]
train = pd.merge(train, train_anno1, how='left', on='ID')

train = pd.merge(train, trainx, how='left', on='ID').fillna('')
y = train['Class'].values
train = train.drop(['Class'], axis=1)

test_anno1 = pd.read_csv("test_variant_annotation_0720_tfidf2.csv")
test_anno1 = test_anno1[['ID']+vw0]
test = pd.merge(test, test_anno1, how='left', on='ID')

test = pd.merge(test, testx, how='left', on='ID').fillna('')
pid = test['ID'].values

# concat train and test
df_all = pd.concat((train, test), axis=0, ignore_index=True)

# merge blups
anno_rename_dict = {'Func.refgene':'Func_refgene','ExonicFunc.refgene':'ExonicFunc_refgene','Tissue.Type':'Tissue_Type',
    'Molecular.Genetics':'Molecular_Genetics','Role.in.Cancer':'Role_in_Cancer','Translocation.Partner':'Translocation_Partner',
    'Other.Germline.Mut':'Other_Germline_Mut',
    'GeneName of train_variant_annotation_0711':'Gene',
    'GeneName of test_variant_annotation_0711':'Gene',
    'Protein_Change of train_variant_annotation_0711':'Variation',
    'Protein_Change of test_variant_annotation_0711':'Variation'}

df_all.rename(columns=anno_rename_dict,inplace=True)

for i,v in enumerate(vw):
    wb = pd.read_csv('blup1/blup1_'+v+'.csv',low_memory=False)
    df_all = df_all.merge(wb, how='left', on=[v])

df_all.drop(vw,inplace=True,axis=1)

df_all['Gene_Share'] = df_all.apply(lambda r: sum([1 for w in r['Gene'].split(' ') if w in r['Text'].split(' ')]), axis=1)
df_all['Variation_Share'] = df_all.apply(lambda r: sum([1 for w in r['Variation'].split(' ') if w in r['Text'].split(' ')]), axis=1)

# df_all['RefAltV'] = df_all['Variation'].str[0] + df_all['Variation'].str[-1]

#commented for Kaggle Limits
for i in range(56):
    df_all['Gene_'+str(i)] = df_all['Gene'].map(lambda x: str(x[i]) if len(x)>i else '')
    df_all['Variation'+str(i)] = df_all['Variation'].map(lambda x: str(x[i]) if len(x)>i else '')


gen_var_lst = sorted(list(train.Gene.unique()) + list(train.Variation.unique()))
print(len(gen_var_lst))
gen_var_lst = [x for x in gen_var_lst if len(x.split(' '))==1]
print(len(gen_var_lst))
i_ = 0
#commented for Kaggle Limits
for gen_var_lst_itm in gen_var_lst:
    if i_ % 100 == 0: print(i_)
    df_all['GV_'+str(gen_var_lst_itm)] = df_all['Text'].map(lambda x: str(x).count(str(gen_var_lst_itm)))
    i_ += 1

for c in df_all.columns:
    if df_all[c].dtype == 'object':
        if c in ['Gene','Variation']:
            lbl = preprocessing.LabelEncoder()
            df_all[c+'_lbl_enc'] = lbl.fit_transform(df_all[c].values)  
            df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
            df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' ')))
        elif c != 'Text':
            lbl = preprocessing.LabelEncoder()
            df_all[c] = lbl.fit_transform(df_all[c].values)
        if c=='Text': 
            df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
            df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' '))) 

train = df_all.iloc[:len(train)]
test = df_all.iloc[len(train):]

class cust_regression_vals(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        x = x.drop(['Gene', 'Variation','ID','Text'],axis=1).values
        return x

class cust_txt_col(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        return x[self.key].apply(str)

print('Pipeline...')
fp = pipeline.Pipeline([
    ('union', pipeline.FeatureUnion(
        n_jobs = -1,
        transformer_list = [
            ('standard', cust_regression_vals()),
            ('pi1', pipeline.Pipeline([('Gene', cust_txt_col('Gene')), 
                                       ('count_Gene', 
                                        feature_extraction.text.CountVectorizer(analyzer=u'char', 
                                                                                ngram_range=(1, 8))), 
                                        ('tsvd1', decomposition.TruncatedSVD(n_components=20, 
                                                                             n_iter=25, random_state=12))])),
            ('pi2', pipeline.Pipeline([('Variation', cust_txt_col('Variation')), 
                                       ('count_Variation', 
                                        feature_extraction.text.CountVectorizer(analyzer=u'char', 
                                                                                ngram_range=(1, 8))), 
                                        ('tsvd2', decomposition.TruncatedSVD(n_components=20, 
                                                                             n_iter=25, random_state=12))])),
            # ('pi3', pipeline.Pipeline([('RefAltV', cust_txt_col('RefAltV')), 
            #                            ('count_RefAlt', 
            #                             feature_extraction.text.CountVectorizer(analyzer=u'char', 
            #                                                                     ngram_range=(1, 2))), 
            #                             ('tsvd3', decomposition.TruncatedSVD(n_components=10, 
            #                                                                 n_iter=25, random_state=12))])),
            
            #commented for Kaggle Limits
            ('pi4', pipeline.Pipeline([('Text', cust_txt_col('Text')), ('tfidf_Text', 
                                       feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2))), 
                                        ('tsvd4', decomposition.TruncatedSVD(n_components=50, 
                                                                             n_iter=25, random_state=12))]))
        ])
    )])

train = fp.fit_transform(train); print(train.shape)
test = fp.transform(test); print(test.shape)

y = y - 1 #fix for zero bound array

# d_test = lgb.Dataset(test)

denom = 0
ntree = 0
fold = 5 #Change to 5, 1 for Kaggle Limits

for i in range(fold):
    print('\nfold ' + str(i+1) + '/' + str(fold))
    x1, x2, y1, y2 = model_selection.train_test_split(train, y, test_size=0.18, random_state=i)
 
    d_train = lgb.Dataset(x1,label=y1)
    d_valid = lgb.Dataset(x2,label=y2)

    watchlist = [d_train, d_valid]
    model = lgb.train(params, d_train, 1000,  watchlist, verbose_eval=50, 
                      early_stopping_rounds=100)
    ntreei = model.best_iteration
    score1 = metrics.log_loss(y2, model.predict(x2), 
                              labels = list(range(9)))
    print(score1)
    # ntreei /= params['num_parallel_tree']
    ntree += (ntreei + 80)
    #if score < 0.9:
    if denom != 0:
        pred = model.predict(test, num_iteration=ntreei+80)
        preds += logit(pred)
    else:
        pred = model.predict(test, num_iteration=ntreei+80)
        preds = logit(pred.copy())
    denom += 1
    # submission = pd.DataFrame(pred, columns=['class'+str(c+1) for c in range(9)])
    # submission['ID'] = pid
    # submission.to_csv('sub/owl'+midx+'_xgb_fold_'  + str(i) + '.csv', index=False)

    # feature importance
    # imp = pd.DataFrame(model.get_fscore().items(), columns=['feature','importance']).sort_values('importance', ascending=False)
    # print(imp.head(n=20))

# bagged prediction
preds /= denom
ntree /= denom
print(ntree)

# fit on full data
d_train = lgb.Dataset(train,label=y)
watchlist = [d_train]
model = lgb.train(params, d_train, ntree,  watchlist, verbose_eval=50)
pred = model.predict(test)

# average with bagged pred
preds = expit((preds + logit(pred))/2)

submission = pd.DataFrame(preds, columns=['class'+str(c+1) for c in range(9)])
submission['ID'] = pid
fs = 'sub/owl'+midx+'_lgb.csv'
submission.to_csv(fs, index=False)
print(fs)

# feature importance
# fnames = list(df_all.columns.values) + ['svd' + str(i) for i in range(90)]
# ['svd_v' + str(i) for i in range(20)] + ['svd_ar' + str(i) for i in range(10)] + \

fnames = list(df_all.columns.values[4:]) + ['svd_g' + str(i) for i in range(20)] + \
    ['svd_v' + str(i) for i in range(20)] + \
    ['svd_t' + str(i) for i in range(50)]
imp1 = pd.DataFrame({'feature':fnames,'gain':model.feature_importance(importance_type='gain'),
'split':model.feature_importance(importance_type='split')})
# imp = pd.DataFrame(model.feature_importance(importance_type='gain').items(),columns=['fnum','importance'])
# fd = pd.DataFrame({'fnum':['f{}'.format(i) for i in range(train.shape[1])],'feature':fnames})
# imp1 = imp.merge(fd,how='left',on=['fnum']).sort_values('importance',ascending=False)
imp1 = imp1.sort_values('gain',ascending=False)
print(imp1.head(n=30))
imp1.to_csv('imp/imp_owl'+midx+'_lgb.csv')
