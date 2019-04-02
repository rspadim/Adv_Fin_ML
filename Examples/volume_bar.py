import pandas as pd
import numpy as np
import numba
from tqdm import tqdm, tqdm_notebook


def create_up_down_sell_buy(data, col_price='<LAST>', col_bid='<BID>', col_ask='<ASK>', col_vol='<VOLUME>', col_up_down='up_down', col_buy_sell='buy_sell'):
    data[col_up_down] = np.NaN
    data[col_buy_sell] = np.NaN
    data_trade = data[data[col_vol] > 0]
    data_trade.loc[(data_trade[col_price] >= data_trade[col_ask]) & (data_trade[col_ask] != 0) & (data_trade[col_bid] < data_trade[col_ask]), col_buy_sell] = 1
    data_trade.loc[(data_trade[col_price] <= data_trade[col_bid]) & (data_trade[col_bid] != 0) & (data_trade[col_bid] < data_trade[col_ask]), col_buy_sell] = -1
    data_trade.loc[data_trade[col_price] > data_trade[col_price].shift(1), col_up_down] = 1
    data_trade.loc[data_trade[col_price] < data_trade[col_price].shift(1), col_up_down] = -1
    data_trade[col_buy_sell].fillna(method='ffill', inplace=True)
    data_trade[col_buy_sell].fillna(0, inplace=True)
    data_trade[col_up_down].fillna(method='ffill', inplace=True)
    data_trade[col_up_down].fillna(0, inplace=True)



#@numba.jit()
def create_bars(data, bar_type='volume', bar_volume=1000, col_time='date_time_ts', col_price='<LAST>', col_volume='<VOLUME>', col_index='#', cur_bar=False):
    bars = []
    if(cur_bar == False):
        cur_bar_start_ts = None
        cur_bar_end_ts = None
        cur_bar_start_index = 0
        cur_bar_end_index = 0
        cur_bar_open = 0.
        cur_bar_high = 0.
        cur_bar_low = 0.
        cur_bar_close = 0.
        cur_bar_volume = 0.
        cur_bar_dollar = 0.
        cur_bar_ticks = 0
    else:
        cur_bar_start_ts = cur_bar['start_ts']
        cur_bar_end_ts = cur_bar['end_ts']
        cur_bar_start_index = cur_bar['start_index']
        cur_bar_end_index = cur_bar['end_index']
        cur_bar_open = cur_bar['open']
        cur_bar_high = cur_bar['high']
        cur_bar_low = cur_bar['low']
        cur_bar_close = cur_bar['close']
        cur_bar_volume = cur_bar['volume']
        cur_bar_dollar = cur_bar['dollar']
        cur_bar_ticks = cur_bar['ticks']

    if (bar_type == 'time'):
        if(cur_bar is not False):
            using_Timestamp = (type(cur_bar_start_ts) == pd.Timestamp)
            if (cur_bar is not False):
                if using_Timestamp:
                    cur_bar_end = cur_bar_start_ts.floor(str(bar_volume)+"s") + pd.Timedelta(bar_volume, unit='seconds')
                else:
                    cur_bar_end = np.floor(cur_bar_start_ts / bar_volume) * bar_volume + bar_volume
        for i in tqdm(range(data.shape[0])):
            i = data[i]
            time, price, vol, ll = i[col_time], i[col_price], i[col_volume], i[col_index]
            dollar = price * vol
            if (cur_bar is False):
                using_Timestamp = (type(time) == pd.Timestamp)
                cur_bar = True
                if using_Timestamp:
                    cur_bar_start_ts = time.floor(str(bar_volume) + "s")
                else:
                    cur_bar_start_ts = np.floor(time / bar_volume)
                cur_bar_end_ts = time
                cur_bar_start_index = ll
                cur_bar_end_index = ll
                cur_bar_open = price
                cur_bar_high = price
                cur_bar_low = price
                cur_bar_close = price
                cur_bar_volume = 0
                cur_bar_dollar = 0
                cur_bar_ticks = 0
                cur_bar_end = cur_bar_start_ts + pd.Timedelta(bar_volume, unit='seconds')
            while True:
                if (time >= cur_bar_end):
                    bars.append([
                        cur_bar_start_ts, cur_bar_end_ts,
                        cur_bar_start_index, cur_bar_end_index,
                        cur_bar_open, cur_bar_high,
                        cur_bar_low, cur_bar_close,
                        cur_bar_volume, cur_bar_dollar,
                        cur_bar_ticks
                    ])
                    if using_Timestamp:
                        cur_bar_start_ts = time.floor(str(bar_volume) + "s")
                    else:
                        cur_bar_start_ts = np.floor(time / bar_volume)
                    cur_bar_end_ts = time
                    cur_bar_start_index = ll
                    cur_bar_end_index = ll
                    cur_bar_open = price
                    cur_bar_high = price
                    cur_bar_low = price
                    cur_bar_close = price
                    cur_bar_volume = 0
                    cur_bar_dollar = 0
                    cur_bar_ticks = 0
                    if using_Timestamp:
                        cur_bar_end = cur_bar_start_ts.floor(str(bar_volume) + "s") + pd.Timedelta(bar_volume, unit='seconds')
                    else:
                        cur_bar_end = np.floor(cur_bar_start_ts / bar_volume) * bar_volume + bar_volume
                else:
                    cur_bar_volume += vol
                    cur_bar_dollar += dollar
                    cur_bar_ticks += 1
                    cur_bar_high = cur_bar_high if cur_bar_high > price else price
                    cur_bar_low = cur_bar_low if cur_bar_low < price else price
                    break
        return bars, [
            'start_ts','end_ts','start_index','end_index',
            'open','high','low','close','volume','dollar','ticks'
        ]

    for i in tqdm(range(data.shape[0])):
    #for i in range(data.shape[0]):
        i=data[i]
        time, price, vol, ll = i[col_time], i[col_price], i[col_volume], i[col_index]
        dollar = price * vol
        if(cur_bar is False):
            cur_bar = True
            cur_bar_start_ts = time
            cur_bar_end_ts = time
            cur_bar_start_index = ll
            cur_bar_end_index = ll
            cur_bar_open = price
            cur_bar_high = price
            cur_bar_low = price
            cur_bar_close = price
            cur_bar_volume = 0
            cur_bar_dollar = 0
            cur_bar_ticks = 0
        while True:
            cur_bar_end_ts, cur_bar_end_index, cur_bar_close = time, ll, price
            if(
                (bar_type == 'volume' and cur_bar_volume + vol >= bar_volume) or
                (bar_type == 'dollar' and cur_bar_dollar + dollar >= bar_volume) or
                (bar_type == 'ticks' and cur_bar_ticks + 1 >= bar_volume)
            ):
                # new bar
                if(bar_type == 'volume'):
                    diff_volume = bar_volume - cur_bar_volume
                    diff_dollar = price * diff_volume
                elif (bar_type == 'dollar'):
                    diff_dollar = bar_volume - cur_bar_dollar
                    diff_volume = diff_dollar / price
                else:
                    diff_dollar = dollar
                    diff_volume = vol
                cur_bar_dollar += diff_dollar
                cur_bar_volume += diff_volume
                cur_bar_high = cur_bar_high if cur_bar_high > price else price
                cur_bar_low = cur_bar_low if cur_bar_low < price else price
                bars.append([
                    cur_bar_start_ts, cur_bar_end_ts,
                    cur_bar_start_index, cur_bar_end_index,
                    cur_bar_open, cur_bar_high,
                    cur_bar_low, cur_bar_close,
                    cur_bar_volume, cur_bar_dollar,
                    cur_bar_ticks
                ])
                cur_bar_start_ts = time
                cur_bar_end_ts = time
                cur_bar_start_index = ll
                cur_bar_end_index = ll
                cur_bar_open = price
                cur_bar_high = price
                cur_bar_low = price
                cur_bar_close = price
                cur_bar_volume = 0
                cur_bar_dollar = 0
                cur_bar_ticks = 0

                vol -= diff_volume
                dollar -= diff_dollar
                if(vol == 0):
                    break
            else:
                cur_bar_volume += vol
                cur_bar_dollar += dollar
                cur_bar_ticks += 1
                cur_bar_high = cur_bar_high if cur_bar_high > price else price
                cur_bar_low = cur_bar_low if cur_bar_low < price else price
                break
    if(cur_bar is not False and cur_bar_volume > 0):
        bars.append([
            cur_bar_start_ts,
            cur_bar_end_ts,
            cur_bar_start_index,
            cur_bar_end_index,
            cur_bar_open,
            cur_bar_high,
            cur_bar_low,
            cur_bar_close,
            cur_bar_volume,
            cur_bar_dollar,
            cur_bar_ticks
        ])
    return bars, [
        'start_ts','end_ts','start_index','end_index',
        'open','high','low','close','volume','dollar','ticks'
    ]

if __name__=='__main__':
    if True:
        data = pd.read_csv('../data/201602010800_201904011014.csv.zip', compression='zip', sep='\t')
        for i in ['<ASK>', '<BID>', '<LAST>']:
            data[i].fillna(method='bfill', inplace=True)
            data[i].fillna(method='ffill', inplace=True)
            data[i].fillna(0, inplace=True)
    #    data['<BID>'].fillna(method='backfill', inplace=True)
    #    data['<LAST>'].fillna(method='backfill', inplace=True)
        data['<VOLUME>'].fillna(0, inplace=True)
        data['<DATE>'] = data['<DATE>'].str.replace('.','-')
        data['date_time'] = data['<DATE>'] + " " + data['<TIME>']
        data['date_time_ts'] = pd.to_datetime(data['date_time'], format='%Y-%m-%d %H:%M:%S.%f')
        data['#'] = data.index.values
        data['buy_sell'] = 0
        data['up_down'] = 0
        df_trades = data[(data['<LAST>'] != 0 & data['<LAST>'].notna()) & (data['<VOLUME>'] != 0 & data['<VOLUME>'].notna())]
        create_up_down_sell_buy(data, col_price='<LAST>', col_bid='<BID>', col_ask='<ASK>', col_up_down='up_down', col_buy_sell='buy_sell')
        data.to_pickle("../data/201602010800_201904011014.pickle")
    else:
        data = pd.read_pickle('../data/201602010800_201904011014.pickle')
        df_trades = data[(data['<LAST>'] != 0 & data['<LAST>'].notna()) & (data['<VOLUME>'] != 0 & data['<VOLUME>'].notna())]

    bar_volume = 60
    bar_type = 'time'
    bars, bars_columns = create_bars(
        df_trades[['date_time_ts', '<LAST>', '<VOLUME>', '#']].values,
        bar_type=bar_type,
        bar_volume=bar_volume,
        col_time=0,
        col_price=1,
        col_volume=2,
        col_index=3
    )
    bars = pd.DataFrame(bars, columns=bars_columns)
    print("Bars shape:", bars.shape)
    bars.to_pickle("../data/201602010800_201904011014." + bar_type + "." + str(bar_volume) + ".pickle")
    pass