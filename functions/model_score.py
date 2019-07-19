import pandas as pd

def ks(data, preds, fpd, n):
    pred = data[preds]
    bad = data[fpd]
    df = pd.DataFrame({'bad': bad, 'pred': pred})
    df['good'] = 1 - df.bad
    df = df.sort_values(by='pred', ascending=True)
    lst = [i*1/n for i in range(1, n)]
    cut_points = df['pred'].quantile(lst)
    cut_point = [0]+list(cut_points)+[1]
    total = data.shape[0]
    good_sum = df['good'].sum()
    bad_sum = total-df['bad']
    df_ks = pd.DataFrame({'index': [], 'total': [], 'bad': [], 'good': [], 'bad_rate': []})
    for i in range(1, n+1):
        temp = df.loc[(df['pred'] < cut_point[i]) & (df['pred'] >= cut_point[i-1])]
        t_total = temp.shape[0]
        t_good = temp['good'].sum()
        t_bad = t_total-t_good
        t_bad_rate = t_bad/t_total
        t_df = pd.DataFrame({'index': [i], 'total': [t_total], 'bad': [t_bad],'good': [t_good],'bad_rate': [t_bad_rate]})
        df_ks = pd.concat([df_ks, t_df], axis=0)
    df_ks = df_ks.sort_values(by='index', ascending=True)
    df_ks = df_ks.set_index('index')
    badsum = df_ks['bad'].sum()
    goodsum = df_ks['good'].sum()
    total = df_ks['total'].sum()
    df_ks['acc_good'] = df_ks['good'].cumsum()
    df_ks['acc_bad'] = df_ks['bad'].cumsum()
    df_ks['accum_good_rate'] = df_ks['acc_good'].apply(lambda x: x/goodsum)
    df_ks['accum_bad_rate'] = df_ks['acc_bad'].apply(lambda x: x/badsum)
    df_ks['ks'] = df_ks['accum_good_rate']-df_ks['accum_bad_rate']
    return df_ks

