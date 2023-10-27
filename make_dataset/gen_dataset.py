import argparse
import json
from tqdm import tqdm
import datasets
import transformers

MODEL_NAME = "THUDM/chatglm-6b"


def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}


def load_data(data_path) -> list:
    with open(data_path, encoding="utf-8") as f:
        examples = json.load(f)
        return examples


def save_data(save_path, examples):
    with open(save_path, "w", encoding="utf-8") as f:
        output = []
        for example in examples:
            formatted_example = format_example(example)
            output.append(formatted_example)
        f.write(json.dumps(output, ensure_ascii=False))
            

def dataset_title():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./datas/dataset_title.json")
    parser.add_argument("--save_path", type=str, default="./datas/dataset_title_input.json")

    args = parser.parse_args()

    examples = load_data(args.data_path)
    if examples:
        save_data(args.save_path, examples)


def preprocess(tokenizer, config, example, max_seq_length):
    prompt = example["context"]
    target = example["target"]
    return {"prompt": prompt, "response": target}

def read_jsonl(path, max_seq_length, skip_overlength=False):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True)
    config = transformers.AutoConfig.from_pretrained(
        MODEL_NAME, trust_remote_code=True, device_map='auto')
    try:
        with open(path, "r") as f:
            for line in tqdm(f, desc="Processing"):
                example = json.loads(line)
                feature = preprocess(tokenizer, config, example, max_seq_length)
                # if skip_overlength and len(feature["input_ids"]) > max_seq_length:
                #     continue
                # feature["input_ids"] = feature["input_ids"][:max_seq_length]
                yield feature
    except FileNotFoundError:
        print(f"File not found at {path}")
    except json.JSONDecodeError:
        print("Invalid JSON format")


def main():
    # dataset_title()
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, default="./datas/dataset_title.jsonl")
    parser.add_argument("--save_path", type=str, default="./datas/hfDataset")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--skip_overlength", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()

    dataset = datasets.Dataset.from_generator(
        lambda: read_jsonl(args.jsonl_path, args.max_seq_length, args.skip_overlength)
    )
    dataset.save_to_disk(args.save_path)

if __name__ == "__main__":
    main()