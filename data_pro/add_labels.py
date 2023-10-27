import pandas as pd
import os
import akshare as ak
import datetime

# result_path = "D:/DA/nlp/datas/content"
# file_list = os.listdir(result_path)
output_path = "D:/DA/nlp/AnnualReportAnalysis/datas/content_with_labels"


def add_label(x, df_price, foward_days = 5, threshold = 0.02, threshold_very = 0.06):
    publish_date = x.publish_date.strftime("%Y-%m-%d")
    df_price["日期"] = pd.to_datetime(df_price["日期"])
    try:
        last_date = df_price[df_price["日期"] < publish_date].iloc[-1].name
    except IndexError:
        return "No data"
    this_date_index = last_date + 1
    next_date_index = this_date_index + foward_days
    
    if next_date_index >= df_price.shape[0]-1:
        return "No data"
    else:
        this = df_price[df_price.index == this_date_index]["开盘"].values[0]
        next_ = df_price[df_price.index == next_date_index]["开盘"].values[0]
        change = (next_ - this)/this
        if change > threshold_very:
            return "very positive"
        elif change > threshold:
            return "positive"
        elif change < -threshold_very:
            return "very negative"
        elif change < -threshold:
            return "negative"
        else:
            return "neutral"


def process_label(df, file_name):
    # df = pd.read_csv(os.path.join(result_path, file_name))
    df["date_time"] = pd.to_datetime(df["date_time"])  # 时间
    df["date"] = pd.to_datetime(df["date_time"].dt.date)
    df["time"] = df["date_time"].dt.time
    df["hour"] = df["date_time"].dt.hour

    start_date = df["date"].min() - datetime.timedelta(days = 10)
    end_date = df["date"].max() + datetime.timedelta(days = 25)
    start_date = start_date.strftime("%Y%m%d")
    end_date = end_date.strftime("%Y%m%d")
    stock_code = df["symbol"].unique()[0][2:] # 股票代码

    df['publish_date'] = df.apply(lambda x:x['date'] if x['hour'] <15 else x['date'] + datetime.timedelta(days = 1), axis=1)

    df_price = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date=start_date, end_date=end_date, adjust="")

    

    df = df.dropna(axis=0, how="any", subset=["publish_date"])

    df["label"] = df.apply(lambda x:add_label(x, df_price = df_price), axis=1)

    file_name += ".csv"
    out_path = os.path.join(output_path, file_name)
    df.to_csv(out_path, index = False)


def main():
    file_name = "./datas/98只股票数据-加股票名字.csv"
    df1 = pd.read_csv(file_name)
    for group_name, group_data in df1.groupby("symbol"):
        process_label(group_data, group_name)
        print(group_name + "\t is done")

if __name__ == "__main__":
    # for id,file_name in enumerate(file_list):
    #     print(id, file_name)
    #     process_label(file_name)

    main()