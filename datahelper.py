"""
Builder of a dataset
author: Alsu Vakhitova
"""
import json
from datetime import datetime, date, time
import pandas as pd
import numpy as np
import glob
from datetime import timedelta

def load_data(path):
    pattern = path + '*.json'
    files = glob.glob(pattern)
    df_total = pd.DataFrame()
    for file in files:
        with open(file) as f:
            data = json.load(f)

        columns = ['time'] + list(next(iter(data[0]['input_data']['rwis_data'].values())).keys())

        arr = np.array(columns)

        for item in data:
            batch = item['input_data']['rwis_data']
            for time in batch.keys():
                d = datetime.strptime(time[:-4], "%Y-%m-%d %H:%M")
                row = [d] + list(batch[time].values())
                arr = np.vstack([arr, row])
        df = pd.DataFrame(columns=arr[0], data=arr[1:])
        # This part adds dummy variables that indicate quarter of the day and month, didn't prove to be useful
        # day_part = []
        # month = []
        # for time in df['time']:
        #     month.append(time.month)
        #     one = time.replace(hour=0, minute=0, second=0, microsecond=0)
        #     two = time.replace(hour=6, minute=0, second=0, microsecond=0)
        #     three = time.replace(hour=12, minute=0, second=0, microsecond=0)
        #     four = time.replace(hour=18, minute=0, second=0, microsecond=0)
        #     five = time.replace(hour=23, minute=59, second=59, microsecond=0)
        #     if one <= time and time <= two:
        #         day_part.append(1)
        #         continue
        #     if two < time and time <= three:
        #         day_part.append(2)
        #         continue
        #     if three < time and time <= four:
        #         day_part.append(3)
        #         continue
        #     if four < time and time <= five:
        #         day_part.append(4)
        #         continue
        # df['day_part'] = day_part
        # df['month'] = month
        df = df.sort_values(by=['time']).drop_duplicates().reset_index(drop=True)
        df_total = pd.concat([df_total, df])
    df_total = df_total.replace(9999, np.NaN)
    df_total = df_total.dropna()
    return df_total.sort_values(by=['time']).drop_duplicates().reset_index(drop=True)

def get_xy(path, num_hours, error_minutes):
    """
    Returnes x and y dataframes
    :param path: path to input file
    :param num_hours: number of hours to make forecast
    :param error_minutes: acceptable error in minutes when building a y-set: y = x + num_hours +- error
    :return: x and y sets
    """
    df = load_data(path)
    hour = timedelta(hours=1)
    minute = timedelta(minutes=1)
    x = pd.DataFrame()
    y = pd.DataFrame()
    for i in range(df.shape[0]):
        time = df.at[i, 'time']
        pr_time1 = time + num_hours * hour - error_minutes*minute
        pr_time2 = time + num_hours * hour + error_minutes*minute
        b = df[(pr_time1 <= df['time']) & (df['time'] <= pr_time2)]
        if not b.empty:
            x = x.append(df.iloc[[i]])
            closest_time = min(b['time'].tolist(), key=lambda d: abs(d - time))
            y = y.append(b[b['time'] == closest_time].iloc[0])
    return x.loc[:, x.columns != 'time'].reset_index(drop=True), y.loc[:, y.columns != 'time'].reset_index(drop=True)