import os
import pandas as pd
from tqdm import tqdm
import argparse
import datasets


columns = ["instruction","input","output"]
# instruction = "What is the sentiment of this news? Answer:{very negative/negative/neutral/positive/very positive}"


def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"prompt": context, "response": target}


def get_dataset_title():
    dataset = pd.DataFrame(columns= columns)
    file_list = os.listdir(args.result_path)
    for file_name in tqdm(file_list):
        df = pd.read_csv(os.path.join(args.result_path, file_name))
        df["input"] = df.apply(lambda x: f'新闻标题为：\"{x["title"]}\"。', axis = 1)
        df["instruction"] = args.instruction
        df['output'] = df["label"]
        tmp = df[columns+["date"]]
        dataset = pd.concat([dataset, tmp])
    
    train_dataset = dataset[columns]
    for item in tqdm(train_dataset.itertuples(), desc="Processing"):
        if item.output == "No data":
            continue
        tmp = {}
        tmp["instruction"] = item.instruction
        tmp["input"] = item.input
        tmp["output"] = item.output
        formatted_example = format_example(tmp)
        yield formatted_example
    
    


if __name__ =="__main__":
    # dataset_title()
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default="D:/DA/nlp/AnnualReportAnalysis/datas/content_with_labels")
    parser.add_argument("--save_path", type=str, default="./datas/hfDataset_news_title")
    parser.add_argument("--instruction", type=str, default="What is the sentiment of this news? Answer:{very negative/negative/neutral/positive/very positive}")

    args = parser.parse_args()

    dataset = datasets.Dataset.from_generator(
        lambda: get_dataset_title()
    )
    dataset.save_to_disk(args.save_path)