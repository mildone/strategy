import QUANTAXIS as QA
try:
    assert QA.__version__>='1.1.0'
except AssertionError:
    print('pip install QUANTAXIS >= 1.1.0 请升级QUANTAXIS后再运行此示例')
    import QUANTAXIS as QA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtk
import mpl_finance as mpf



def wds(df,duration='W'):
    #now support W stands for Week, M stands for Month, Q stands for quater
    df['date'] = pd.to_datetime(df.index.get_level_values('date'))
    df.set_index("date", inplace=True)
    period = duration

    weekly_df = df.resample(period).last()
    weekly_df['open'] = df['open'].resample(period).first()
    weekly_df['high'] = df['high'].resample(period).max()
    weekly_df['low'] = df['low'].resample(period).min()
    weekly_df['close'] = df['close'].resample(period).last()
    weekly_df['volume'] = df['volume'].resample(period).sum()
    weekly_df['amount'] = df['amount'].resample(period).sum()
    weekly_df.reset_index('date',inplace=True)
    weekly_df.dropna(axis=0,subset = ["open", "high","close","low"])
    return weekly_df


def divergence(wk, short=10, mid=20, long=30):
    wk['lo'] = pd.Series.rolling(wk.close, long).mean()
    wk['mi'] = pd.Series.rolling(wk.close, mid).mean()
    wk['sh'] = pd.Series.rolling(wk.close, short).mean()

    wk['short'] = pd.Series.ewm(wk.close, span=short, min_periods=short - 1, adjust=True).mean()
    wk['mid'] = pd.Series.ewm(wk.close, span=mid, min_periods=mid - 1, adjust=True).mean()
    wk['long'] = pd.Series.ewm(wk.close, span=long, min_periods=long - 1, adjust=True).mean()

    wk['BIAS'] = (wk.close - wk.long) * 100 / wk.long
    wk['CS'] = (wk.close - wk.short) * 100 / wk.short
    wk['SM'] = (wk.short - wk.mid) * 100 / wk.mid
    wk['ML'] = (wk.mid - wk.long) * 100 / wk.long
    return wk



def Plot(sample, short=10, mid=20, long=30):

    quotes = []
    N = sample.shape[0]
    ind = np.arange(N)
    for i in range(len(sample)):
        li = []
        datef = ind[i]  # 日期转换成float days
        open_p = sample.open[i]
        close_p = sample.close[i]
        high_p = sample.high[i]
        low_p = sample.low[i]
        li = [datef, open_p, close_p, high_p, low_p]
        t = tuple(li)
        quotes.append(t)

    def format_date(x, pos=None):
        thisind = np.clip(int(x + 0.5), 0, N - 1)
        return sample.date[thisind]

    fig = plt.figure()
    fig.set_size_inches(30.5, 20.5)
    print('call here')
    ax2 = fig.add_subplot(3, 1, 1)
    ax2.set_title("weekly candlestick", fontsize='xx-large', fontweight='bold')

    mpf.candlestick_ochl(ax2, quotes, width=0.6, colorup='r', colordown='g', alpha=1.0)
    ax2.plot(ind,sample.lo,'r-',label='MA'+str(long))
    ax2.plot(ind,sample.mi,'blue',label='MA'+str(mid))
    ax2.plot(ind,sample.sh,'grey',label='MA'+str(short))
    ax2.xaxis.set_major_formatter(mtk.FuncFormatter(format_date))
    ax2.grid(True)
    ax2.legend(loc='best')
    fig.autofmt_xdate()

    ax1 = fig.add_subplot(3, 1, 2, sharex=ax2)

    ax1.grid(True)
    ax1.plot(ind, sample.CS, 'red',label='CS')
    ax1.plot(ind, sample.SM, 'blue',label='SM')
    ax1.plot(ind,sample.ML,'orange',label='ML')
    ax1.bar(ind,sample.BIAS,color='grey')
    ax1.xaxis.set_major_formatter(mtk.FuncFormatter(format_date))
    # ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax1.get_yticks())))
    ax1.legend(loc='best')
    fig.autofmt_xdate()
    plt.show()










if __name__ == "__main__":
    test = QA.QA_fetch_stock_day_adv('000977','2010-01-01','2020-07-01').data
    wk = wds(test,duration='W')


    divergence(wk)
    print(wk)
    Plot(wk)
