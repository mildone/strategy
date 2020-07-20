import datetime
import re

import QUANTAXIS as QA
import core.Util as uti

def getStocklist():
    """
    get all stock as list
    usage as:
    QA.QA_util_log_info('GET STOCK LIST')
    stocks=getStocklist()
    """
    data=QA.QAFetch.QATdx.QA_fetch_get_stock_list('stock')
    stocklist=data.index.get_level_values('code').to_list()
    return stocklist


def loadLocalData(stocks, start_date='2019-03-15', end='cur'):
    """
    data() as pdDataFrame
    stocks could be list of all the stock or some. if you pass single one e.g. 000001 it will get one only
    to get dedicated stock, using below method, and notice stockp() will be dataFrame
    stockp = data.select_code(stock)

    """
    if(end == 'cur'):
        cur = datetime.datetime.now()
        mon = str(cur.month)
        day = str(cur.day)
        if (re.match('[0-9]{1}', mon) and len(mon) == 1):
            mon = '0' + mon
        if (re.match('[0-9]{1}', day) and len(day) == 1):
            day = '0' + day

        et = str(cur.year) + '-' + mon + '-' + day
    else:
        et = end

    data = QA.QA_fetch_stock_day_adv(stocks, start_date, et)
    return data

def change(dd):
    from functools import reduce
    pp_array = [float(close) for close in dd.close]
    temp_array = [(price1, price2) for price1, price2 in zip(pp_array[:-1], pp_array[1:])]
    change = list(map(lambda pp: reduce(lambda a, b: round((b - a) / a, 3), pp), temp_array))
    change.insert(0, 0)
    dd['change']=change
    return dd


if __name__ == '__main__':
    stock = getStocklist()

    codelist3 = QA.QA_fetch_stock_block_adv().get_block('云计算').code[:]
    codelist1 = QA.QA_fetch_stock_block_adv().get_block('华为概念').code[:]
    codelist2 = QA.QA_fetch_stock_block_adv().get_block('5G概念').code[:]
    # codelist4 = QA.QA_fetch_stock_block_adv().get_block('国产软件').code[:]
    codelist2.extend(codelist3)
    codelist2.extend(codelist1)

    codelist = list(set(codelist2))
    m = loadLocalData(stock)
    num = 0
    win = 0
    ind = m.add_func(change)
    data_forbacktest = m.select_time('2020-01-01', '2020-07-15')
    for items in data_forbacktest.panel_gen:
        num = 0
        win = 0
        for item in items.security_gen:
            daily_ind = ind.loc[item.index]
            num += 1
            print('{} at {} with {} and {}'.format(item.code[0], item.date[0], item.close[0], daily_ind.change.iloc[0]))
            if (daily_ind.change.iloc[0] > 1):
                win += 1
        print('------' + str(items.date[0]) + '------' + str(win) + '/' + str(num))
    '''
    for i in stock:
        try:
            sample = m.select_code(i).data
            #sample['change'] = uti.change(sample.close)
            num += 1
            if(sample.close[-1]>sample.close[-2]):
                win +=1
        except:
            pass

    print('{} / {} '.format(win,num))
    '''
