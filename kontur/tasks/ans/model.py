from transformers import AutoTokenizer, AutoModelForQuestionAnswering
# Запись результата в файл test.json
import json
import torch

# Инициализация модели BERT
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Пример обработки текста документа
document_text = "Это пример текста документа"
question = "Какой текст содержится в документе?"
inputs = tokenizer.encode_plus(question, document_text, add_special_tokens=True, max_length=512, return_tensors='pt')
print(inputs['input_ids'])
print(inputs['attention_mask'])
print(inputs['token_type_ids'])

#start_positions, end_positions = model(**inputs).values()

'''
# Получение фрагмента текста документа
answer_start = torch.argmax(start_positions)
answer_end = torch.argmax(end_positions) + 1
extracted_part = {
    'text': document_text[answer_start:answer_end],
    'answer_start': int(answer_start),
    'answer_end': int(answer_end)
}

with open('ans/test.json', 'r') as f:
    data = json.load(f)


for item in data:
    id = item['id']
    print(id)
    text = item['text']
    print(text)
    label = item['label']
    print(label)

    question = label
    inputs = tokenizer.encode_plus(text, text_pair=label, add_special_tokens=True, max_length=512, return_tensors='pt')

    start_positions, end_positions = model(**inputs).values()
    answer_start = torch.argmax(start_positions)
    answer_end = torch.argmax(end_positions) + 1
    extracted_part = {
        'text': text[answer_start:answer_end],
        'answer_start': int(answer_start),
        'answer_end': int(answer_end)
    }
    item['extracted_part'] = extracted_part

with open('ans/pred.json', 'w') as f:
    json.dump(data, f)
    '''