from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from chatbot import chat_with_bot  # Import the chatbot logic from chatbot.py

app = Flask(__name__)

# Load traffic dataset and NLP model
dataset_path = 'dataset/traffic.csv'

if not os.path.exists(dataset_path):
    print(f"Dataset not found at {dataset_path}. Please check the path.")
    exit()

traffic_data = pd.read_csv(dataset_path)
traffic_data['DateTime'] = pd.to_datetime(traffic_data['DateTime'])

# Load pre-trained model and tokenizer
model_name = "microsoft/DialoGPT-medium"
try:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
except OSError as e:
    print(f"Error loading model: {e}")
    model = tokenizer = None

# Custom dataset for transportation-specific queries (can be expanded)
custom_dataset = {
    "best route": "The best route depends on your current location, traffic, and destination. Consider using real-time navigation apps like Google Maps or Waze.",
    "health": "For healthy traveling, stay hydrated, stretch during long trips, and maintain hygiene by carrying sanitizers and tissues.",
    "security": "Ensure your safety by keeping valuables close, avoiding isolated areas, and using trusted transportation services.",
    "travel tips": "Pack light, keep important documents handy, and double-check your tickets and reservations before traveling.",
    "emergency": "In case of an emergency, contact local authorities or helplines immediately. Always save emergency numbers before your trip."
}

def chatbot_response(message):
    # Check for specific queries in custom dataset
    for key, response in custom_dataset.items():
        if key in message.lower():
            return response

    # Use pre-trained model for general conversation
    inputs = tokenizer.encode(message, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.json.get("message")
    if user_message:
        response = chat_with_bot(user_message, model, tokenizer, traffic_data)
        return jsonify({"response": response})
    return jsonify({"response": "No message received. Please try again."})

if __name__ == "__main__":
    app.run(debug=True)
