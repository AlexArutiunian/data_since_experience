import json
import random
from tqdm import tqdm
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForQuestionAnswering, AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F

# Задаем путь к файлу с данными для обучения
train_file_path = 'train.json'

# Загружаем данные из файла
with open(train_file_path, 'r', encoding="utf-8") as f:
    data = json.load(f)

# Задаем параметры обучения
epochs = 3
batch_size = 8
learning_rate = 5e-5
adam_epsilon = 1e-8

# Загружаем предобученный токенизатор и модель
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")

# Задаем оптимизатор и расписание обучения
optimizer = AdamW(model.parameters(), lr=learning_rate, eps=adam_epsilon)
total_steps = len(data) * epochs // batch_size
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Создаем функцию для формирования батчей данных
def create_batches(data, batch_size):
    batches = []
    batch = []
    for item in data:
        batch.append(item)
        if len(batch) == batch_size:
            batches.append(batch)
            batch = []
    if batch:
        batches.append(batch)
    return batches

# Создаем функцию для обучения модели на одном батче данных
def train_batch(batch, tokenizer, model, optimizer):
    input_ids = []
    attention_masks = []
    start_positions = []
    end_positions = []

    for item in batch:
        text = item['text']
        label = item['label']
        extracted_part_text = item['extracted_part']['text'][0]
        answer_start = item['extracted_part']['answer_start'][0]
        answer_end = item['extracted_part']['answer_end'][0]

        # Формируем входные данные для модели
        encoded_dict = tokenizer(text, extracted_part_text, return_tensors='pt', padding=True, truncation='only_first')
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        start_positions.append(torch.tensor([answer_start]))
        end_positions.append(torch.tensor([answer_end]))

    # Преобразуем списки в тензоры
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    start_positions = torch.cat(start_positions, dim=0)
    end_positions = torch.cat(end_positions, dim=0)

    # Передаем входные данные в модель и получаем предсказания
    model.train()
    outputs = model(input_ids, attention_mask=attention_masks, start_positions=start_positions, end_positions=end_positions)

    # Вычисляем функцию потерь и выполняем обратное распространение ошибки
    loss = outputs.loss
    loss.backward()

    # Обновляем параметры модели
    optimizer.step()
    scheduler.step()
    model.zero_grad()

    return loss.item()

def train(data, tokenizer, model, optimizer, device, batch_size=16, num_epochs=5):
    train_loss = []
    for epoch in range(num_epochs):
        print(f"EPOCH {epoch + 1}/{num_epochs}")
        epoch_loss = 0
        epoch_steps = 0
        model.train()
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i:i+batch_size]
            texts = [d["text"] for d in batch]
            labels = [d["label"] for d in batch]
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
            labels = torch.tensor([0 if label == "обеспечение исполнения контракта" else 1 for label in labels]).to(device)
            #outputs = model(**inputs)[0]
            labels = labels.unsqueeze(1)
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])[0]
            
            loss = F.binary_cross_entropy_with_logits(outputs, labels.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            epoch_steps += 1
        epoch_loss /= epoch_steps
        print(f"LOSS: {epoch_loss}")
        train_loss.append(epoch_loss)
    return train_loss


# Тренировка модели
model_name = "distilbert-base-uncased"
model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
train_loss = train(data, tokenizer, model, optimizer, device)

torch.save(model.state_dict(), 'model.pt')