
import datetime
import re

import pandas as pd
import QUANTAXIS as QA

try:
    assert QA.__version__ >= '1.1.0'
except AssertionError:
    print('pip install QUANTAXIS >= 1.1.0 请升级QUANTAXIS后再运行此示例')
    import QUANTAXIS as QA
import numpy as np



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtk
import mpl_finance as mpf
import quant.Util as uti


def PriceBias(day, short=20, mid=60, long=120, type='SML',zoom=100):
    import quant.MACD as macd
    day['long'] = QA.EMA(day.close, long)
    day['mid'] = QA.EMA(day.close, mid)
    day['short'] = QA.EMA(day.close, short)
    day['BIAS'] = (day.close - day.long) * 100 / day.long
    day['CS'] = (day.close - day.short) * 100 / day.short
    day['SM'] = (day.short - day.mid) * 100 / day.mid
    day['ML'] = (day.mid - day.long) * 100 / day.long
    print(day)
    if(zoom > day.shape[0]):
        day = day[0:]
    else:
        day = day[0-zoom:]
    quotes = macd.MINcandlestruct(day, uti.dayindex, uti.dayformate)
    # N = sample.index.get_level_values(index).shape[0]
    N = day.shape[0]
    ind = np.arange(N)

    def format_date(x, pos=None):
        thisind = np.clip(int(x + 0.5), 0, N - 1)
        return day.index.get_level_values(uti.dayindex)[thisind].strftime(uti.dayformate)

    fig = plt.figure()
    fig.set_size_inches(40.5, 20.5)
    ax2 = fig.add_subplot(3, 1, 1)
    ax2.set_title("candlestick", fontsize='xx-large', fontweight='bold')

    mpf.candlestick_ochl(ax2, quotes, width=0.6, colorup='r', colordown='g', alpha=1.0)
    ax2.plot(ind, day.long, 'r-', label='EMA' + str(long),linewidth = 0.7)
    ax2.plot(ind, day.mid, 'blue', label='EMA' + str(mid),linewidth =0.7)
    ax2.plot(ind, day.short, 'purple', label='EMA' + str(short),linewidth = 0.7)
    #ax2.plot(100, 30, 'go', markersize=12, markeredgewidth=0.5,
             #markerfacecolor='None', markeredgecolor='green')
    '''
    for i in range(N):
        if (day.single[i] == 1):
            ax2.axvline(x=i, ls='--', color='red')
        if (day.single[i] == 3):
            ax2.axvline(x=i, ls='--', color='green')
    '''
    ax2.xaxis.set_major_formatter(mtk.FuncFormatter(format_date))
    ax2.grid(True)
    ax2.legend(loc='best')
    fig.autofmt_xdate()

    ax3 = fig.add_subplot(3, 1, 2, sharex=ax2)
    bar_red = np.where(day.close > day.open, day.volume, 0)
    bar_green = np.where(day.close < day.open, day.volume, 0)
    ax3.bar(ind, bar_red, color='red')
    ax3.bar(ind, bar_green, color='green')
    # x3.bar(ind,day.BIAS,color='blue')
    # ax3.axhline(y=0,ls='--',color='yellow')
    ax3.grid(True)
    ax3.xaxis.set_major_formatter(mtk.FuncFormatter(format_date))
    fig.autofmt_xdate()

    ax4 = fig.add_subplot(3, 1, 3, sharex=ax2)
    if ('S' in type):
        ax4.plot(ind, day.CS, 'r-', label='CS',linewidth = 1)
    if ('M' in type):
        ax4.plot(ind, day.SM, 'blue', label='SM',linewidth = 1)
    if ('L' in type):
        ax4.bar(ind, day.ML, color='grey')
    if ('B' in type):
        ax4.bar(ind,day.BIAS, color = 'grey')
    # ax3.axhline(y=0,ls='--',color='yellow')
    ax4.grid(True)
    ax4.xaxis.set_major_formatter(mtk.FuncFormatter(format_date))
    fig.autofmt_xdate()

    plt.show()


def forceANA(code,zo=100,ty = 'SMLB',cg = 'stock', st = 20, mi = 60, ln = 120):
    cur = datetime.datetime.now()
    mon = str(cur.month)
    day = str(cur.day)
    if (re.match('[0-9]{1}', mon) and len(mon) == 1):
        mon = '0' + mon
    if (re.match('[0-9]{1}', day) and len(day) == 1):
        day = '0' + day

    et = str(cur.year) + '-' + mon + '-' + day
    if(cg == 'stock'):
        start = '2010-01-01'
        dd = QA.QA_fetch_stock_day_adv(code,start,et).data
        PriceBias(dd,type = ty,zoom = zo, short = st, mid = mi, long = ln)
    elif(cg == 'index'):
        start = '2019-10-01'
        dd = QA.QA_fetch_index_day_adv(code, start, et).data
        PriceBias(dd, type=ty, zoom=zo, short = st, mid = mi, long = ln)



if __name__ == "__main__":
    forceANA('300097',zo=200,ty = 'SMB', cg = 'stock', st = 20, mi = 60, ln = 120)


