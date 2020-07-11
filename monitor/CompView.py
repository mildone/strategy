import QUANTAXIS as QA
import dateutil

try:
    assert QA.__version__ >= '1.1.0'
except AssertionError:
    print('pip install QUANTAXIS >= 1.1.0 请升级QUANTAXIS后再运行此示例')
    import QUANTAXIS as QA
from abupy import ABuRegUtil

import re
from matplotlib import gridspec
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtk
import mpl_finance as mpf
import core.Util as uti
import core.zoom as zo

def compView(code, start, end,short = 20, mid = 60, long = 120,zoom=300,cg='stock'):
    endn = end
    td = uti.prepareData(code,end=endn,cg=cg)



    wk = zo.wds(td,duration='w')
    zo.divergence(wk)
    wqu = uti.candlestruct(wk)

    t15 = uti.prepareData(code,start = '2020-01-01',frequence='15min',end=endn,cg=cg)
    t60 = uti.prepareData(code,start = '2020-01-01',frequence='60min',end=endn,cg=cg)
    uti.divergence(t15)
    uti.divergence(t60)
    uti.divergence(td)

    #t60qu = uti.candlestruct(t60)
    #t15qu = uti.candlestruct(t15)

    fig = plt.figure()
    gs = gridspec.GridSpec(8, 8)
    fig.set_size_inches(80.5, 70.5)
    ax2 = fig.add_subplot(gs[0:4, 0:4])

    if (zoom > t15.shape[0]):
        t15=t15
    else:
        t15 = t15[0 - zoom:]
    t15qu = uti.candlestruct(t15)
    N1 = t15.shape[0]
    ind1 = np.arange(N1)

    def format1_date(x, pos=None):
        thisind = np.clip(int(x + 0.5), 0, N1 - 1)
        return t15.index.get_level_values(uti.index)[thisind]


    ax2.set_title("15 min", fontsize='xx-large', fontweight='bold')
    mpf.candlestick_ochl(ax2, t15qu, width=0.6, colorup='r', colordown='g', alpha=1.0)
    #ax2.plot(ind,t15.sh,'r-',label='short')
    ax2.grid(True)

    ax2.plot(ind1, t15.sh, 'r-', label='short', linewidth=0.7)
    ax2.plot(ind1, t15.mi, 'blue', label='mid', linewidth=0.7)
    ax2.plot(ind1, t15.long, 'purple', label='long', linewidth=0.7)

    ratio = t15.low.median() * 0.03
    ax2.plot(N1 - short, t15.low[N1 - short] - ratio, '^', markersize=4, markeredgewidth=2, markerfacecolor='None',
             markeredgecolor='red')
    # ax2.axvline(x=N-short,ls='--',color='purple')
    ax2.plot(N1 - mid, t15.low[N1 - mid] - ratio, '^', markersize=4, markeredgewidth=2, markerfacecolor='None',
             markeredgecolor='blue')
    ax2.plot(N1 - long, t15.low[N1 - long] - ratio, '^', markersize=4, markeredgewidth=2, markerfacecolor='None',
             markeredgecolor='purple')

    #ax2.xaxis.set_major_formatter(mtk.FuncFormatter(format_date))
    ax2.legend(loc='best')
    ax2.xaxis.set_major_formatter(mtk.FuncFormatter(format1_date))
    fig.autofmt_xdate()

    ax3 = fig.add_subplot(gs[0:4, 4:8])

    if (zoom > t60.shape[0]):
        t60=t60
    else:
        t60 = t60[0 - zoom:]
    t60qu = uti.candlestruct(t60)
    N6 = t60.shape[0]
    ind6 = np.arange(N6)

    def format6_date(x, pos=None):
        thisind = np.clip(int(x + 0.5), 0, N6 - 1)
        return t60.index.get_level_values(uti.index)[thisind]
    ax3.set_title("60 min", fontsize='xx-large', fontweight='bold')
    mpf.candlestick_ochl(ax3, t60qu, width=0.6, colorup='r', colordown='g', alpha=1.0)
    ax3.grid(True)
    ax3.plot(ind6, t60.sh, 'r-', label = 'short',linewidth=0.7)
    ax3.plot(ind6, t60.mi, 'blue', label='mid', linewidth=0.7)
    ax3.plot(ind6, t60.long, 'purple', label='long', linewidth=0.7)

    ratio = t60.low.median() * 0.03
    ax3.plot(N6 - short, t60.low[N6 - short] - ratio, '^', markersize=4, markeredgewidth=2, markerfacecolor='None',
              markeredgecolor='red')
    # ax2.axvline(x=N-short,ls='--',color='purple')
    ax3.plot(N6 - mid, t60.low[N6 - mid] - ratio, '^', markersize=4, markeredgewidth=2, markerfacecolor='None',
              markeredgecolor='blue')
    ax3.plot(N6 - long, t60.low[N6 - long] - ratio, '^', markersize=4, markeredgewidth=2, markerfacecolor='None',
              markeredgecolor='purple')
    ax3.legend(loc='best')
    ax3.xaxis.set_major_formatter(mtk.FuncFormatter(format6_date))
    fig.autofmt_xdate()



    ax31 = fig.add_subplot(gs[4:8, 4:8])
    if(zoom>td.shape[0]):
        td=td
    else:
        td=td[0-zoom:]
    tqu = uti.candlestruct(td)
    N = td.shape[0]
    ind = np.arange(N)

    def format_date(x, pos=None):
        thisind = np.clip(int(x + 0.5), 0, N - 1)
        return td.index.get_level_values(uti.dayindex)[thisind]
    ax31.set_title("day", fontsize='xx-large', fontweight='bold')
    mpf.candlestick_ochl(ax31, tqu, width=0.6, colorup='r', colordown='g', alpha=1.0)
    ax31.plot(ind,td.sh,'r-',label='short',linewidth=0.7)
    ax31.plot(ind, td.mi, 'blue', label='mid', linewidth=0.7)
    ax31.plot(ind,td.lo, 'purple',label='long', linewidth=0.7)
    ratio = td.low.median()*0.03
    ax31.plot(N - short, td.low[N - short] - ratio, '^', markersize=4, markeredgewidth=2, markerfacecolor='None',
             markeredgecolor='red')
    # ax2.axvline(x=N-short,ls='--',color='purple')
    ax31.plot(N - mid, td.low[N - mid] - ratio, '^', markersize=4, markeredgewidth=2, markerfacecolor='None',
             markeredgecolor='blue')
    ax31.plot(N - long, td.low[N - long] - ratio, '^', markersize=4, markeredgewidth=2, markerfacecolor='None',
             markeredgecolor='purple')

    ax31.grid(True)
    ax31.xaxis.set_major_formatter(mtk.FuncFormatter(format_date))
    ax31.legend(loc='best')
    fig.autofmt_xdate()

    ax21 = fig.add_subplot(gs[4:8, 0:4])
    NW = wk.shape[0]
    indw = np.arange(NW)

    def format6_date(x, pos=None):
        thisind = np.clip(int(x + 0.5), 0, NW - 1)
        return wk.index.get_level_values(uti.dayindex)[thisind]
    ax21.set_title("week", fontsize='xx-large', fontweight='bold')
    mpf.candlestick_ochl(ax21, wqu, width=0.6, colorup='r', colordown='g', alpha=1.0)

    ax21.plot(indw, wk.sh, 'r-', label='short', linewidth=0.7)
    ax21.plot(indw, wk.mi, 'blue', label='mid', linewidth=0.7)
    ax21.plot(indw, wk.lo, 'purple', label='long', linewidth=0.7)
    ratio = wk.low.median() * 0.03
    if(NW>short):
        ax21.plot(NW - short, wk.low[NW - short] - ratio, '^', markersize=4, markeredgewidth=2, markerfacecolor='None',
              markeredgecolor='red')
    else:
        pass
    # ax2.axvline(x=N-short,ls='--',color='purple')
    if(NW>mid):
        ax21.plot(NW - mid, wk.low[NW - mid] - ratio, '^', markersize=4, markeredgewidth=2, markerfacecolor='None',
              markeredgecolor='blue')
    if(NW>long):
        ax21.plot(NW - long, wk.low[NW - long] - ratio, '^', markersize=4, markeredgewidth=2, markerfacecolor='None',
              markeredgecolor='purple')


    ax21.grid(True)
    ax21.legend(loc='best')
    fig.autofmt_xdate()


    plt.show()

if __name__ == '__main__':
    compView('515050','2019-01-01','cur',zoom=1000,cg='index')






