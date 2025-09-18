from flask import Flask, render_template, request
import pandas as pd

from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__, static_url_path='/static')

model = joblib.load('model.pkl')

# Load the data
data = pd.read_csv('houses.csv') 

data.drop(data[data['Location']=='A'].index, inplace=True)

# Create a label encoder object
label_encoder = LabelEncoder()

@app.template_filter('format_price')
def format_price(price):
    if price is None:
        return ''  # Return an empty string if price is None
    elif price >= 10000000:  # If price is 8 digits or more (e.g., 10 million or more)
        return f'{price / 10000000:.1f} Crore ₹'  # Convert to Crore ₹ format
    elif price >= 100000:  # If price is 6 or 7 digits (e.g., 1 lakh or more)
        return f'{price / 100000:.1f} Lakh ₹'  # Convert to Lakh ₹ format
    else:
        return f'{price:.2f}'  # Default format
@app.route('/')
def home():
    return render_template('Login.html')
@app.route('/Login.html')
def login():
    return render_template('Login.html')
@app.route('/Register.html')
def register():
    return render_template('Register.html')
@app.route('/home.html')
def main():
    unique_locations = data['Location'].unique()
    return render_template('home.html', locations=unique_locations)
@app.route('/aboutus.html')
def about():
    return render_template('aboutus.html')
@app.route('/contact.html')
def contact():
    return render_template('contact.html')
@app.route('/predict', methods=['POST'])
def predict():
    input_location = request.form['location']
    input_bedrooms = int(request.form['bedrooms'])
    input_location_encoded = label_encoder.fit_transform([input_location])[0]
    predictions = {}
    location_bedroom_rows = data[(data['Location'] == input_location) & (data['No. of Bedrooms'] == input_bedrooms)]
    if not location_bedroom_rows.empty:
        features = location_bedroom_rows.drop(columns=['Price', 'Price_per_sqft'], errors='ignore')
        features['Location'] = input_location_encoded
        features = features.reindex(columns=model.feature_names_in_)
        predicted_price = model.predict(features)
        predictions[input_bedrooms] = predicted_price[0]
    else:
        predictions[input_bedrooms] = None
    return render_template('result.html', location=input_location, predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)