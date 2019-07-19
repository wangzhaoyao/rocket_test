import pandas as pd
import numpy as np
import numbers



def find_sp_char_col(df):
    num_col = list(df.describe().T.index)
    col_lst = [col for col in list(df.columns) if len(set(df[col]))>20 and isinstance(df[col][0], numbers.Real)]
    res = [col for col in col_lst if col not in num_col]
    print('highly suspect following col contains special char,please cheak following col carefully'+','.join(res))
    return res

def replace_sp_char(df,sp_char_lst=['"F","','F'], tar=np.nan):
    for i in sp_char_lst:
        df.replace(i, tar, inplace=True)
    return df

def fillna(df, col=None, by=0, method=3):
    '''
    :param df: 待处理的数据集
    :param col: 待处理的特征集，传入list
    :param by: 待填充的值
    :param method: 不同的method对应不同的填充方法，method1对应多列填充同样的一个值，method2对应不同的列填充不同的值，
    method3默认整个数据集填充同一个值
    :return:
    '''
    if method == 1:
        for i in col:
            df[i].fillna(by, inplace=True)
    elif method == 2:
        for i, j in zip(col, by):
            df[i].fillna(j, inplace=True)
    else:
        df.fillna(by, inplace=True)
    return df

def drop_duplcate_rows(df):
    df = df.drop_duplicates()
    return df

def drop_useless_col(df, threshold = 0.95, remain_col=[]):
    '''
    :param df: 需要处理的数据集，删除以下变量1：单一箱体占比95%以上的变量 2.缺失率占比95%以上变量
    :return: 处理后的数据集
    '''
    col_lst = list(df.columns)
    remove_col = []
    for col in col_lst:
        if MaximumBinPcnt(df, col) > threshold:
            print('drop more than 95% columns ' + col)
            remove_col.append(col)
            continue
        if na_pct(df, col) > threshold:
            print('drop null pct more than 95% columns' + col)
            remove_col.append(col)
    col_lst = [col for col in col_lst if col not in remove_col]
    col_lst = col_lst + remain_col
    return df[col_lst]

def na_pct(df, col):
    N = df.shape[0]
    total = df[col].isnull().sum()
    pcnt = total*1.0/N
    return pcnt

def MaximumBinPcnt(df, col):
    N = df.shape[0]
    total = list(df[col].value_counts())[0]
    pcnt = total*1.0/N
    return pcnt

if __name__=='__main__':
    df = pd.read_csv(r"..\rawdata\zyf_multi_credit_draw_all_1_eduab_pre.csv", encoding='utf-8', sep='\\t', engine='python')
    df = df.loc[(df['target15'] == 0) | (df['target15'] == 1)]
    df = drop_duplcate_rows(df)
    df = replace_sp_char(df)
    df = drop_useless_col(df,remain_col=['target15'])
    df.to_csv(r"..\rawdata\zyf_multi_credit_draw_all_1_eduab_pre1.csv", encoding='gbk', index=False)