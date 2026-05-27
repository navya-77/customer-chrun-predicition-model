from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import traceback

app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load('churn_model.pkl')
    print("✅ Successfully loaded the trained model.")
except FileNotFoundError:
    print("❌ Error: 'churn_model.pkl' not found. Please run the model training script first.")
    model = None
except Exception as e:
    print(f"❌ Error loading the model: {e}")
    traceback.print_exc()
    model = None

# A simple home page
@app.route('/')
def home():
    return render_template('index.html')

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': "Model not loaded. Please contact support."}), 500

    try:
        data = request.json
        tenure = float(data.get('tenure'))
        monthly_charges = float(data.get('monthly_charges'))
        total_charges = float(data.get('total_charges'))
        
        # Create a DataFrame from the input data
        input_data = pd.DataFrame([[tenure, monthly_charges, total_charges]],
                                  columns=['tenure', 'monthly_charges', 'total_charges'])
        
        # Make a prediction
        prediction = model.predict(input_data)[0]
        
        # Return the prediction result
        result = "Churn" if prediction == 1 else "No Churn"
        return jsonify({'prediction': result})
    
    except (ValueError, TypeError) as e:
        return jsonify({'error': f"Invalid input format: {e}"}), 400
    except Exception as e:
        return jsonify({'error': f"An unexpected error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)

