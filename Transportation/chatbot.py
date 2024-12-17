import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# Function to fetch traffic data based on user query
def query_traffic_data(query, traffic_data):
    query_lower = query.lower()

    try:
        if 'vehicle count' in query_lower:
            if 'junction' in query_lower:
                junction = int(query_lower.split('junction')[1].strip().split()[0])
                filtered_data = traffic_data[traffic_data['Junction'] == junction]
                
                if 'at' in query_lower:
                    time_str = query_lower.split('at')[1].strip()
                    time_obj = pd.to_datetime(time_str, format='%I %p', errors='coerce')
                    if time_obj is not pd.NaT:
                        filtered_data = filtered_data[filtered_data['DateTime'].dt.hour == time_obj.hour]
                        if not filtered_data.empty:
                            count = filtered_data.iloc[0]['Vehicles']
                            return f"The vehicle count at junction {junction} at {time_str} is {count}."
                        else:
                            return f"No data available for junction {junction} at {time_str}."
                avg_count = filtered_data['Vehicles'].mean()
                return f"The average vehicle count at junction {junction} is {avg_count:.2f}."
            return "Please specify a junction number in your query."

        elif 'busiest time' in query_lower:
            busiest_hour = traffic_data.groupby(traffic_data['DateTime'].dt.hour)['Vehicles'].sum().idxmax()
            return f"The busiest time for traffic is around {busiest_hour}:00 hours."

        elif 'traffic trend' in query_lower:
            hourly_trend = traffic_data.groupby(traffic_data['DateTime'].dt.hour)['Vehicles'].mean()
            trend_str = "\n".join([f"Hour {hour}: {count:.2f} vehicles" for hour, count in hourly_trend.items()])
            return f"Here's the traffic trend by hour:\n{trend_str}"

        else:
            return None  # Signal to use the chatbot model

    except Exception as e:
        return "An error occurred while processing your query. Please try again."

# Function to interact with the chatbot
def chat_with_bot(user_input, model, tokenizer, traffic_data):
    response = query_traffic_data(user_input, traffic_data)
    if response:
        return response
    if model and tokenizer:
        return generate_bot_response(model, tokenizer, user_input)
    return "I'm unable to answer your question at the moment. Please try again later."

# Function to generate a bot response using the DialoGPT model
def generate_bot_response(model, tokenizer, user_input):
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    bot_output = model.generate(
        inputs,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.6,
        num_return_sequences=1
    )
    return tokenizer.decode(bot_output[:, inputs.shape[-1]:][0], skip_special_tokens=True)
