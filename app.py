from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import pytesseract
from PIL import Image
import io
# Load the trained model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('count_vectorizer.pkl')  # Load the pre-fitted vectorizer

# Initialize Flask app
app = Flask(__name__)
def output(text):
        # Vectorize the input text
    print(text)
    text_vectorized = vectorizer.transform([text])  # Fix input format

    # Make a prediction
    prediction = model.predict(text_vectorized)

    # Map prediction to label
    result = "Fake News" if prediction[0] == 00 else "Real News"

    # Return the result as JSON
    print(result)
    return ({'result': result})
# Define the home route
@app.route('/')
def home():

    return render_template('test.html')  # Render the HTML frontend

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input text from the form
    text = request.form['text']
    return jsonify(output(text))

@app.route('/predict_image', methods=['POST'])
def predict_image():
    print("ASdfasdf")
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))
    
    # Extract text using OCR
    extracted_text = pytesseract.image_to_string(image)
    print(extracted_text)
    # Call predict function with extracted text
    result = output(extracted_text.strip())
    result['extracted_text']=extracted_text
    return jsonify(result)


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
    
