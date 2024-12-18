from tkinter import *
from tkinterhtml import HtmlFrame
import folium
import pandas as pd
import os
from flask import Flask, render_template, request
from dijkstra import dijkstra
from dataRetriever import getGraphPoints

app = Flask(__name__)

# Function to predict fuel consumption
def predict_fuel_consumption(distance, fuel_efficiency):
    """
    Predicts the fuel consumption based on distance and vehicle's fuel efficiency.
    :param distance: Distance between cities (in km).
    :param fuel_efficiency: Vehicle's fuel efficiency (km per liter).
    :return: Predicted fuel consumption (in liters).
    """
    try:
        fuel_consumed = distance / fuel_efficiency
        return round(fuel_consumed, 2)
    except ZeroDivisionError:
        return "Invalid fuel efficiency (cannot be zero)."

# Function to retrieve distance from dataset
def get_distance(source, destination):
    """
    Retrieves distance between source and destination from a CSV dataset.
    :param source: Source city.
    :param destination: Destination city.
    :return: Distance in km or None if route not found.
    """
    try:
        df = pd.read_csv('fuel.csv')
        row = df[(df['Source'] == source) & (df['Destination'] == destination)]
        if not row.empty:
            return row.iloc[0]['Distance (km)']
        else:
            return None
    except Exception as e:
        return None

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/home.html")
def shortestPath():
    return render_template('home.html')

@app.route("/chatbot.html")
def chatbot():
    return render_template('chatbot.html')

@app.route("/Fuel.html", methods=["GET", "POST"])
def Fuel():
    if request.method == "POST":
        # Fetch form inputs
        source = request.form['source']
        destination = request.form['destination']
        fuel_efficiency = float(request.form['fuel_efficiency'])

        # Get distance from dataset
        distance = get_distance(source, destination)
        if distance is None:
            return render_template(
                'fuel_output.html', 
                error=f"No route found between {source} and {destination}."
            )

        # Predict fuel consumption
        fuel_consumed = predict_fuel_consumption(distance, fuel_efficiency)

        # Render the output with prediction details
        return render_template(
            'fuel_output.html', 
            source=source, 
            destination=destination, 
            distance=distance, 
            fuel_efficiency=fuel_efficiency, 
            fuel_consumed=fuel_consumed
        )
    return render_template('Fuel.html')

@app.route("/home.html", methods=["POST"])
def getValue():
    graph, points = getGraphPoints()
    source = request.form['source']
    dest = request.form['dest']
    fuel_efficiency = float(request.form.get('fuel_efficiency', 15))  # Default fuel efficiency = 15 km/l

    # Calculate shortest path and distance
    min_dist, path = dijkstra(graph, 100000, source, dest)
    val = dest
    pts = [(points[dest][0], points[dest][1])]
    total_distance = min_dist

    while path[val] != '-1':
        city = path[val]
        pts.append((points[city][0], points[city][1]))
        val = city

    # Predict total fuel consumption
    total_fuel = predict_fuel_consumption(total_distance, fuel_efficiency)

    # Generate map with route
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=6)
    for key, [lat, long] in points.items():
        folium.Marker([lat, long], popup=key).add_to(m)
    folium.PolyLine(pts).add_to(m)
    m.save('templates/map.html')

    return render_template(
        'output.html', 
        source=source, 
        destination=dest, 
        total_distance=total_distance, 
        fuel_efficiency=fuel_efficiency, 
        total_fuel=total_fuel
    )

@app.route("/about")
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True)
