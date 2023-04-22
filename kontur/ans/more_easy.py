from transformers import MobileBertTokenizer, MobileBertForQuestionAnswering
import torch

model_name = 'google/mobilebert-uncased'
tokenizer = MobileBertTokenizer.from_pretrained(model_name)
model = MobileBertForQuestionAnswering.from_pretrained(model_name)

# пример обработки текста документа
document_text = "Это пример текста документа"
question = "Какой текст содержится в документе?"
inputs = tokenizer.encode_plus(question, document_text, add_special_tokens=True, max_length=512, return_tensors='pt')
start_positions, end_positions = model(**inputs).values()
answer_start = torch.argmax(start_positions)
answer_end = torch.argmax(end_positions) + 1
extracted_part = {
    'text': document_text[answer_start:answer_end],
    'answer_start': int(answer_start),
    'answer_end': int(answer_end)
}
