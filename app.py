# Importing necessary libraries
from flask import Flask, request, render_template
import numpy as np
import pickle
import joblib

# Corrected loading of the trained model and scalers
loaded_model = joblib.load(open('model.pkl', 'rb'))
loaded_scaler_standard = pickle.load(open('standscaler.pkl', 'rb'))
loaded_scaler_minmax = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Creating Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Extracting form data
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])

        # Creating feature list
        feature_list = [N, P, K, temp, humidity, ph, rainfall]

        print(f"Number of features in feature_list: {len(feature_list)}")

        # Assuming your model expects 7 features
        if len(feature_list) == 7:
            # Scaling features using the loaded scalers
            scaled_features_standard = loaded_scaler_standard.transform([feature_list])
            scaled_features_minmax = loaded_scaler_minmax.transform([feature_list])

            # Combining both scalers' features
            final_features = np.concatenate((scaled_features_standard, scaled_features_minmax), axis=1)

            print(f"Input shape to the model: {final_features.shape}")

            # Making prediction
            prediction = loaded_model.predict(final_features)

            # Crop dictionary
            crop_dict = {
                1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
            }

            # Creating result message
            if prediction[0] in crop_dict:
                crop = crop_dict[prediction[0]]
                result = "{} is the best crop to be cultivated in the given conditions.".format(crop)
            else:
                result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

            return render_template('index.html', result=result)
        else:
            error_message = f"Error processing input: Expected 7 features, but got {len(feature_list)}."
            return render_template('index.html', result=error_message)

    except Exception as e:
        # Handle exceptions (e.g., invalid input format)
        error_message = "Error processing input: {}".format(str(e))
        return render_template('index.html', result=error_message)

# Python main
if __name__ == "__main__":
    app.run(debug=True)
