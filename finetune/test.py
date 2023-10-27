import datasets
import transformers

MODEL_NAME = "THUDM/chatglm-6b"

dataset = datasets.load_from_disk("./datas/hfDataset")
tokenizer = transformers.AutoTokenizer.from_pretrained(
    MODEL_NAME, trust_remote_code=True)
tokenizer