from flask import Flask, request, jsonify, session
import boto3
import json
from dotenv import load_dotenv
import os
from datetime import datetime
import uuid

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
# app.secret_key = os.getenv('SECRET_KEY') 

# Initialize the Bedrock client with environment variables
bedrock_client = boto3.client(
    'bedrock-runtime',
    region_name='ap-southeast-2',
    aws_access_key_id=os.getenv('BEDROCK_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('BEDROCK_SECRET_ACCESS_KEY')
)

# Initialize the DynamoDB client
dynamodb = boto3.resource(
    'dynamodb',
    region_name='ap-southeast-2',
    aws_access_key_id=os.getenv('DYNAMODB_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('DYNAMODB_SECRET_ACCESS_KEY_SECRET')
)

# Get the Dynamo table
table = dynamodb.Table(os.getenv('DYNAMODB_TABLE_NAME'))

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

def store_conversation_in_dynamodb(conversation_id, conversation_history):
    timestamp = datetime.utcnow().isoformat()
    conversation_json = json.dumps(conversation_history)
    table.put_item(
        Item={
            'conversation_id': conversation_id,
            'timestamp': timestamp,
            'conversation': conversation_json
        }
    )

@app.route('/chat', methods=['POST'])
def generate_text():
    data = request.json
    prompt = data.get('prompt', '')
    
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    
    # Retrieve conversation history from session
    conversation_history = session.get('conversation_history', [])
    
    # Append the new prompt to the conversation history
    conversation_history.append({"role": "user", "message": prompt})
    
    # Create a single prompt from the conversation history
    full_prompt = "\n".join([f"{entry['role']}: {entry['message']}" for entry in conversation_history])
    
    model_id = 'amazon.titan-text-lite-v1'
    generated_text = invoke_model(model_id, full_prompt)
    
    # Append the model's response to the conversation history
    conversation_history.append({"role": "model", "message": generated_text})
    
    # Store the updated conversation history back in the session
    session['conversation_history'] = conversation_history
    
    # Store the conversation in DynamoDB
    conversation_id = session.get('conversation_id', str(uuid.uuid4()))
    session['conversation_id'] = conversation_id
    store_conversation_in_dynamodb(conversation_id, conversation_history)
    
    return jsonify({'generatedText': generated_text})

if __name__ == '__main__':
    app.run(debug=True)