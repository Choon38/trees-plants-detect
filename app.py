from flask import Flask, request, render_template
from inference_sdk import InferenceHTTPClient
import json

# Initialize the Inference Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="mQNYiR7QrVFLWpXpd5O5"
)

# Create the Flask app instance
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file part"
    
    image_file = request.files['image']
    
    if image_file.filename == '':
        return "No selected file"
    
    # Save the uploaded image temporarily
    image_path = f'images/{image_file.filename}'
    image_file.save(image_path)

    # Perform inference
    result = CLIENT.infer(image_path, model_id="tree-species-identification-rjtsb/1")

    # Optionally save results to a JSON file
    with open('results/output.json', 'w') as f:
        json.dump(result, f, indent=4)

    # Pass results to the template
    return render_template('index.html', 
                           inference_id=result['inference_id'], 
                           time=result['time'], 
                           predictions=result['predictions'])

if __name__ == '__main__':
    app.run(debug=True)