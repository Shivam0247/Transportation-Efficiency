import pandas as pd

# Load dataset
dataset_path = 'C:/Users/HP/Desktop/Transportation/dataset/traffic.csv'  # Replace with the actual path
data = pd.read_csv(dataset_path)

# Verify necessary columns
required_columns = ['dialog', 'response']
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"Dataset must contain columns: {required_columns}")

# Clean data: remove duplicates and null values
data = data.dropna(subset=required_columns).drop_duplicates(subset=required_columns)

# Save cleaned data
cleaned_dataset_path = 'cleaned_dataset.csv'
data.to_csv(cleaned_dataset_path, index=False)
print(f"Cleaned dataset saved to {cleaned_dataset_path}")




from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Load cleaned dataset
dataset_path = 'cleaned_dataset.csv'
dataset = load_dataset('csv', data_files={'train': dataset_path}, delimiter=',')

# Load pre-trained model and tokenizer
model_name = "microsoft/DialoGPT-medium"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Preprocessing function
def preprocess_function(examples):
    inputs = tokenizer(examples['dialog'], truncation=True, padding='max_length', max_length=512)
    outputs = tokenizer(examples['response'], truncation=True, padding='max_length', max_length=512)
    inputs['labels'] = outputs['input_ids']
    return inputs

# Apply preprocessing
dataset = dataset.map(preprocess_function, batched=True)
dataset = dataset['train'].train_test_split(test_size=0.1)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_dir='./logs',
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained('./fine_tuned_traffic_chatbot')
tokenizer.save_pretrained('./fine_tuned_traffic_chatbot')
