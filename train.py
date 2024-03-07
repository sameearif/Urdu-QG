import argparse
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import AutoModelForSeq2SeqLM, MT5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Trainer
import torch
import re

parser = argparse.ArgumentParser(description="Question Generation Training")
parser.add_argument('--model', choices=['small', 'base', 'large', 'xl', 'xxl'], type=str, help='MT5 model')
parser.add_argument('--epochs', type=int, help='Training epochs')
parser.add_argument('--train_batch_size', type=int, help='Training batch size')
parser.add_argument('--eval_batch_size', type=int, help='Eval batch size')

args = parser.parse_args()

def filter_function(example):
    return not example['is_impossible']

def get_dataset():
    dataset = load_dataset("uqa/UQA", use_auth_token="")
    dataset["train"] = dataset["train"].filter(filter_function)
    dataset["validation"] = dataset["validation"].filter(filter_function)
    return dataset

def add_eos_to_extract_examples(example):
    start_pos = example['answer_start']
    end_pos = start_pos + len(example['answer'])
    example['context'] = example['context'][:start_pos] + '<hl>' + example['context'][start_pos:end_pos] + '<hl>' + example['context'][end_pos:]
    example['context'].lstrip(' ').rstrip(' ')
    sentences = example['context'].split("۔ ")
    context = ""
    for sentence in sentences:
      if '<hl>' in sentence:
        context = '%s <hl>%s۔<hl> ' % (context, sentence.replace('<hl>', '').replace('<hl>', ''))
        example['target_text'] = '%s۔' % (sentence)
      else:
        context = '%s %s۔ ' % (context, sentence)
      
    example['input_text'] = 'extract_answer: %s' % (context)
    example['input_text'] = re.sub('۔+', '۔', example['input_text'])
    example['input_text'] = re.sub(r' ،', '،', example['input_text']).strip()
    example['target_text'] = re.sub('۔+', '۔', example['target_text'])
    example['target_text'] = re.sub(r' ،', '،', example['target_text']).strip()
    return example

def add_eos_to_qg_examples(example):
    start_pos = example['answer_start']
    end_pos = start_pos + len(example['answer'])
    example['context'] = example['context'][:start_pos] + '<hl> ' + example['context'][start_pos:end_pos] + ' <hl>' + example['context'][end_pos:]
    example['input_text'] = 'generate question: %s' % (example['context'])
    example['target_text'] = '%s' % example['question']
    example['input_text'] = re.sub('۔+', '۔', example['input_text'])
    example['input_text'] = re.sub(r' ،', '،', example['input_text']).strip()
    example['target_text'] = re.sub('۔+', '۔', example['target_text'])
    example['target_text'] = re.sub(r' ،', '،', example['target_text']).strip()
    return example

def convert_to_features(example_batch):
    input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'], truncation=True, padding="max_length", max_length=512)
    target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], truncation=True, padding="max_length", max_length=128)

    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids'],
    }

    return encodings


dataset = get_dataset()

tokenizer = MT5Tokenizer.from_pretrained(f'google/mt5-{args.model}')
tokenizer.add_tokens(['<hl>'])

train_dataset_extract = dataset["train"].map(add_eos_to_extract_examples)
train_dataset_qg = dataset["train"].map(add_eos_to_qg_examples)
train_dataset = concatenate_datasets([train_dataset_extract, train_dataset_qg])
train_dataset = train_dataset.map(convert_to_features, batched=True)

valid_dataset_extract = dataset["validation"].map(add_eos_to_extract_examples, load_from_cache_file=False)
valid_dataset_qg = dataset["validation"].map(add_eos_to_qg_examples, load_from_cache_file=False)
valid_dataset = concatenate_datasets([valid_dataset_extract, valid_dataset_qg])
valid_dataset = valid_dataset.map(convert_to_features, batched=True, load_from_cache_file=False)

columns = ['input_ids', 'attention_mask', 'labels']
train_dataset.set_format(type='torch', columns=columns)
valid_dataset.set_format(type='torch', columns=columns)

model = AutoModelForSeq2SeqLM.from_pretrained(f'google/mt5-{args.model}')

training_args = Seq2SeqTrainingArguments(
    output_dir=f'mt5-{args.model}-qg-uqa',
    num_train_epochs=args.epochs,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
)

data_collator = DataCollatorForSeq2Seq(tokenizer)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
)

trainer.train()

