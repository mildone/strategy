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

