from sklearn import tree
from sklearn.tree import _tree
import pandas as pd
import numbers
import numpy as np


def get_num_cat_col(df, excol=None):
    '''
    :param df: 待处理的数据集
    :param excol: 需要排除的列，如cust_no等
    :return:
    '''
    df = df.drop(excol, axis=1)
    allFeatures = list(df.columns)
    num_col = []
    for col in allFeatures:
        uniq_valid_vals = [i for i in df[col] if i == i]
        uniq_valid_vals = list(set(uniq_valid_vals))
        if len(uniq_valid_vals) >= 10 and isinstance(uniq_valid_vals[0], numbers.Real):
            num_col.append(col)
    cat_col = [i for i in allFeatures if i not in num_col]
    return num_col, cat_col



def AssignBin(x, cutOffPoints):
    numBin = len(cutOffPoints)
    if numBin == 0:
        if np.isnan(x):
            return 'Bin -1'
        else:
            return 'Bin 0'
    else:
        if np.isnan(x):
            return 'Bin -1'
        if x <= cutOffPoints[0]:
            return 'Bin 0'
        elif x > cutOffPoints[-1]:
            return 'Bin {}'.format(numBin)
        else:
            for i in range(0,numBin):
                if cutOffPoints[i] < x <=cutOffPoints[i+1]:
                    return 'Bin {}'.format(i+1)



def smbinning(df,col,fpd, bins=3):
    T = df.shape[0]
    df = df.loc[df[col].notnull()]
    y = df[[fpd]]
    x_test = df[[col]]
    mytree = tree.DecisionTreeClassifier(min_samples_leaf=int(0.1*T),
                                         criterion="entropy",
                                         max_leaf_nodes=bins)
    mytree.fit(x_test, y)
    cutpoint = mytree.tree_.threshold
    cutpoint = cutpoint[cutpoint != _tree.TREE_UNDEFINED]
    cutpoint.sort()
    return cutpoint

def BadRateMonotone(df, col, target):
    df = df.loc[~df[col].isin(['Bin -1'])]
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    combined = zip(regroup['total'], regroup['bad'])
    badRate = [x[1]*1.0/x[0] for x in combined]
    badRateMonotone = [badRate[i] < badRate[i+1] for i in range(len(badRate)-1)]
    Monotone = len(set(badRateMonotone))
    if Monotone == 1:
        return True
    else:
        return False



def Calc_IV(df, col, target):
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    T = sum(regroup['total'])
    B = sum(regroup['bad'])
    regroup['good'] = regroup['total'] - regroup['bad']
    G = T - B
    regroup['bad_pcnt'] = regroup['bad'].map(lambda x: x*1.0/B)
    regroup['good_pcnt'] = regroup['good'].map(lambda x: x * 1.0 / G)
    regroup['WOE'] = regroup.apply(lambda x: np.nan if x.bad_pcnt == 0.0 else np.log(x.good_pcnt*1.0/x.bad_pcnt),axis = 1)
    regroup['IV'] = regroup.apply(lambda x: np.nan if x.bad_pcnt == 0.0 else (x.good_pcnt-x.bad_pcnt)*np.log(x.good_pcnt*1.0/x.bad_pcnt), axis = 1)
    IV = regroup['IV'].sum()
    return {'iv_df': regroup, 'iv': IV}


def calc_iv_cont(df, col, target, bins=3):
    for i in range(bins, 1, -1):
        cutpoints = smbinning(df, col, target, bins=i)
        col1 = str(col)+'_Bin'
        df[col1] = df[col].map(lambda x: AssignBin(x, cutpoints))
        if BadRateMonotone(df, col1, target):
            break
    res = Calc_IV(df, col1, target)
    return res


def BadRateEncoding(df, col, target):
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    regroup['bad_rate'] = regroup.apply(lambda x: x.bad*1.0/x.total, axis = 1)
    br_dict = regroup[[col, 'bad_rate']].set_index([col]).to_dict(orient='index')
    for k, v in br_dict.items():
        br_dict[k] = v['bad_rate']
    badRateEnconding = df[col].map(lambda x: br_dict[x])
    return {'encoding': badRateEnconding, 'br_rate': br_dict}


