import folium
from flask import Flask, render_template
import model  # Import the model.py file we created earlier

app = Flask(__name__)

def get_flood_probabilities():
    # Call the function to load data and filter it for heavy rain
    heavy_rain_data = model.get_heavy_rain_data()

    # Predict flood probabilities using the trained model
    flood_probs = model.predict_flood_probabilities(heavy_rain_data)

    return flood_probs

def generate_map(flood_probs):
    # Create a base map
    base_map = folium.Map(location=[heavy_rain_data['Lat'].mean(), heavy_rain_data['Long'].mean()], zoom_start=5)

    # Define a colormap for the markers
    colormap = folium.LinearColormap(colors=['green', 'yellow', 'red'], vmin=0, vmax=1)

    # Add markers to the map for each station
    for index, row in flood_probs.iterrows():
        color = colormap(row['flood_probability'])
        folium.Marker(location=[row['Lat'], row['Long']],
                      popup=f"Station: {row['Station_name']}<br>Probability: {row['flood_probability']:.2%}",
                      icon=folium.Icon(color=color, icon='tint', prefix='fa')).add_to(base_map)

    return base_map._repr_html_()

@app.route('/')
def index():
    flood_probs = get_flood_probabilities()
    map_html = generate_map(flood_probs)
    return render_template('index.html', map_html=map_html)

if __name__ == '__main__':
    app.run(debug=True)
