# Запись результата в файл test.json
import json
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

with open('test.json', 'r') as f:
    data = json.load(f)

for item in data:
    id = item['id']
    text = item['text']
    label = item['label']
    question = label
    inputs = tokenizer(question, text, return_tensors="pt")
    start_positions, end_positions = model(**inputs).values()
    answer_start = torch.argmax(start_positions)
    answer_end = torch.argmax(end_positions) + 1
    extracted_part = {
        'text': text[answer_start:answer_end],
        'answer_start': int(answer_start),
        'answer_end': int(answer_end)
    }
    item['extracted_part'] = extracted_part

with open('predictions.json', 'w') as f:
    json.dump(data, f)