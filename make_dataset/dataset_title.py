import os
import datetime
import pandas as pd
import numpy as np
from lxml import etree
from tqdm import tqdm
import json


result_path = "D:\DA\nlp\AnnualReportAnalysis\datas\content_with_labels"
columns = ["instruction","input","output"]
instruction = "What is the sentiment of this news? Answer:{very negative/negative/neutral/positive/very positive}"
file_list = os.listdir(result_path)

def get_dataset_title():
    dataset = pd.DataFrame(columns= columns)
    for file_name in tqdm(file_list):
        df = pd.read_csv(os.path.join(result_path, file_name))
        df["input"] = df.apply(lambda x: f'新闻标题为：\"{x["title"]}\"。', axis = 1)
        df["instruction"] = instruction
        df['output'] = df["label"]
        tmp = df[columns+["date"]]
        dataset = pd.concat([dataset, tmp])
    
    train_dataset = dataset[columns]
    data_list = []
    for item in train_dataset.itertuples():
        if item.output == "No data":
            continue
        tmp = {}
        tmp["instruction"] = item.instruction
        tmp["input"] = item.input
        tmp["output"] = item.output
        data_list.append(tmp)
    with open("./datas/dataset_title_news.json", "w+", encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii= False)
    


if __name__ =="__main__":
    get_dataset_title()