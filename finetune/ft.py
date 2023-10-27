import datasets
import transformers
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import Trainer
from dataclasses import dataclass, field
from transformers import TrainingArguments, HfArgumentParser
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_int8_training
import torch
import torch.nn as nn
import os
import torch
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING


MODEL_NAME = "/home/user/imported_models/chatglm2-6b/huggingface/THUDM/chatglm2-6b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

@dataclass
class FinetuneArgument:
    dataset_path: str = field(default="/home/user/chatglm_ft/datas/hfDataset")
    model_path: str = field(default="save_model")
    lora_rank: int = field(default=8)


def set_model():
    # init model
    model = AutoModel.from_pretrained(
        MODEL_NAME, trust_remote_code=True, device_map="auto", load_in_8bit = True)
    model = prepare_model_for_int8_training(model, use_gradient_checkpointing=True)
    torch.cuda.empty_cache() # 回收显存
    # model.gradient_checkpointing_enable()
    # model.enable_input_require_grads()
    # model.is_parallelizable = True
    # model.model_parallel = True
    # model.config.use_cache = (
    #     False  # silence the warnings. Please re-enable for inference!
    # )
    target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['chatglm']

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules,
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    return model


def load_data():
    dataset = datasets.load_from_disk(FinetuneArgument.dataset_path)
    train_val = dataset.train_test_split( test_size = 0.1, shuffle=True, seed=42 )
    train_data = train_val["train"].shuffle()
    val_data = train_val["test"].shuffle()
    return train_data, val_data

class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

    def prediction_step(self, model: nn.Module, inputs, prediction_loss_only: bool, ignore_keys = None):
        with torch.no_grad():
            res = model(
                input_ids=inputs["input_ids"].to(model.device),
                labels=inputs["labels"].to(model.device),
            ).loss
        return (res, None, None)

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))

class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def data_collator(features: list) -> dict:
    max_seq_length = 512
    max_ans_length = 64
    model_inputs = {
            "input_ids": [],
            "labels": [],
        }
    for i in range(len(features)):
        query, answer = features[i]["prompt"], features[i]["response"]
        a_ids = tokenizer.encode(text=query, add_special_tokens=True,truncation=True,
                      max_length=max_seq_length)
        b_ids = tokenizer.encode(text=answer, add_special_tokens=True, truncation=True,
                                 max_length=max_ans_length)

        context_length = len(a_ids)
        # input_ids = a_ids + b_ids + [tokenizer.eos_token_id]
        # labels = [tokenizer.pad_token_id] * context_length + b_ids + [tokenizer.eos_token_id]
        input_ids = a_ids + b_ids
        labels = [tokenizer.pad_token_id] * context_length + b_ids

        pad_len = max_seq_length - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
        labels = labels + [tokenizer.pad_token_id] * pad_len


        model_inputs["input_ids"].append(input_ids)
        model_inputs["labels"].append(labels)
    model_inputs["input_ids"] = torch.tensor(model_inputs["input_ids"], dtype=torch.long)
    model_inputs["labels"] = torch.tensor(model_inputs["labels"], dtype=torch.long)

    return model_inputs



def main():
    training_args = TrainingArguments(
        output_dir='./finetuned_model',    # saved model path
        logging_steps = 500,
        # max_steps=10000,
        num_train_epochs = 2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=1000,
        save_steps=500,
        fp16=True,
        # bf16=True,
        torch_compile = False,
        load_best_model_at_end = True,
        evaluation_strategy="steps",
        remove_unused_columns=False,

    )
    train_data, val_data = load_data()
    model = set_model()
    model.print_trainable_parameters()


    trainer = ModifiedTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=data_collator
    )
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16) as autocast, torch.backends.cuda.sdp_kernel(
        enable_flash=False) as disable:
        train_result = trainer.train()
    # save model
    model.save_pretrained(training_args.output_dir)


    prompt = "今天的日期为：9月18日。\n新闻标题为：\报告期内，受宏观经济下行、行业竞争加剧的持续影响，泰一指尚全面收缩“传统互联网营销业务”，实现营业收入 3,856.84 万元，与上年同期（会计差错调整后）相比下降 96.71%，实现利润-47,181.10 万元（会计差错调整后），与上年同期相比下降 53.98%\ \n What is the sentiment of this news? Answer:{very negative/negative/neutral/positive/very positive}"
    model.chat(tokenizer, prompt, history=[])


if __name__ == "__main__":
    main()
