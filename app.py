from flask import Flask, request, jsonify
import boto3
import json

app = Flask(__name__)

# Initialize the Bedrock client
bedrock_client = boto3.client('bedrock-runtime', region_name='ap-southeast-2')

def invoke_model(model_id, prompt):
    try:
        # Construct the request payload
        request_payload = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 4096,
                "stopSequences": [],
                "temperature": 0,
                "topP": 1
            }
        }
        
        # Send the request to the model
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_payload),
            contentType='application/json',
            accept='application/json'
        )
        
        # Decode and extract the result
        response_body = response['body'].read().decode('utf-8')
        result = json.loads(response_body)
        
        # Extract output text from the results list
        if 'results' in result and len(result['results']) > 0:
            output_text = result['results'][0].get('outputText', 'No output text found')
        else:
            output_text = 'No results found'
        
        return output_text
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return 'Error occurred'

@app.route('/chat', methods=['POST'])
def generate_text():
    data = request.json
    prompt = data.get('prompt', '')
    
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    
    model_id = 'amazon.titan-text-lite-v1'
    generated_text = invoke_model(model_id, prompt)
    
    return jsonify({'generatedText': generated_text})

if __name__ == '__main__':
    app.run(debug=True)
