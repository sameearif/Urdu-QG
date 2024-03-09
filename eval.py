import argparse
from transformers import AutoModelForSeq2SeqLM, MT5Tokenizer
from datasets import Dataset, load_dataset, concatenate_datasets
import evaluate
import Levenshtein
from tqdm import tqdm
import re

parser = argparse.ArgumentParser(description="Question Generation Training")
parser.add_argument('--eval_batch_size', type=int, help='Eval batch size')

args = parser.parse_args()

def filter_function(example):
  return not example['is_impossible']

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

def create_batches(data, batch_size):
  batches = []
  for i in range(0, len(data), batch_size):
    batches.append(data[i:i+batch_size])
  return batches

def extract_answer(dataset):
  data = []
  for batch in tqdm(dataset):
    inputs = tokenizer.batch_encode_plus(batch['input_text'], truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    outputs = model.generate(
        input_ids=inputs['input_ids'].to("cuda"),
        attention_mask=inputs['attention_mask'].to("cuda"),
        max_length=128,
        num_beams=10,
        num_return_sequences=5
    )
    batch_idx = 0
    decoded_outputs = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]
    for i in range(0, len(decoded_outputs), 5):
      levenshtein = []
      for output in decoded_outputs[i:i+5]:
        try:
          answer_pred = output.split('<hl>')[1].strip()
          levenshtein.append(Levenshtein.distance(answer_pred, batch['answer'][batch_idx]))
        except:
          answer_pred = ""
          levenshtein.append(-1)
      idx = levenshtein.index(min(levenshtein))
      data.append({'input': re.sub(r'<hl>(.*?)<hl>', decoded_outputs[i:i+5][idx], batch["input_text"][batch_idx]), 'question': batch['question'][batch_idx], 'answer': batch['answer'][batch_idx]})
      batch_idx += 1
  return data

def add_prefix(data):
  new_data = []
  for batch in data:
    new_batch = []
    for i in range(len(batch)):
      new_batch.append({'input_text': 'generate question: ' + batch[i]['input'].replace('extract_answer: ', ''), 'question': batch[i]['question']})
    new_data.append(new_batch)
  return new_data

def generate_question(dataset):
  pred = []
  ques = []
  for batch in tqdm(dataset):
    input = [i["input_text"] for i in batch]
    inputs = tokenizer.batch_encode_plus(input, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    outputs = model.generate(
        input_ids=inputs['input_ids'].to("cuda"),
        attention_mask=inputs['attention_mask'].to("cuda"),
        max_length=128,
    )
    decoded_outputs = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]
    for i in range(len(decoded_outputs)):
      pred.append(decoded_outputs[i])
      ques.append(batch[i]["question"])
  return pred, ques

dataset = load_dataset("uqa/UQA")
dataset["validation"] = dataset["validation"].filter(filter_function)
valid_dataset = dataset["validation"].map(add_eos_to_extract_examples, load_from_cache_file=False)

batch_size = args.eval_batch_size
valid_dataset_batched = create_batches(valid_dataset, batch_size)

rouge = evaluate.load('rouge')
bleu = evaluate.load('bleu')
meteor = evaluate.load('meteor')

model = AutoModelForSeq2SeqLM.from_pretrained("uqa/mt5-large-qg-uqa").to('cuda')
tokenizer = MT5Tokenizer.from_pretrained("uqa/mt5-large-qg-uqa")

data = extract_answer(valid_dataset_batched)
valid_dataset_batched = create_batches(data, batch_size)
valid_dataset_batched = add_prefix(valid_dataset_batched)

pred, ques = generate_question(valid_dataset_batched)

print(rouge.compute(predictions=pred, references=ques))
print(bleu.compute(predictions=pred, references=ques))
print(meteor.compute(predictions=pred, references=ques))
