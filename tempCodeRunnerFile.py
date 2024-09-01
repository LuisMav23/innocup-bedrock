from flask import Flask, request, jsonify
import logging
import boto3
from botocore.exceptions import ClientError

app = Flask(__name__)

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def generate_conversation(bedrock_client, model_id, system_prompts, messages):
    """
    Sends messages to a model.
    Args:
        bedrock_client: The Boto3 Bedrock runtime client.
        model_id (str): The model ID to use.
        system_prompts (JSON) : The system prompts for the model to use.
        messages (JSON) : The messages to send to the model.

    Returns:
        response (JSON): The conversation that the model generated.
    """
    logger.info("Generating message with model %s", model_id)

    # Inference parameters to use.
    temperature = 0.5
    top_k = 200

    # Base inference parameters to use.
    inference_config = {"temperature": temperature}
    # Additional inference parameters to use.
    additional_model_fields = {"top_k": top_k}

    # Send the message.
    response = bedrock_client.converse(
        modelId=model_id,
        messages=messages,
        system=system_prompts,
        inferenceConfig=inference_config,
        additionalModelRequestFields=additional_model_fields
    )

    # Log token usage.
    token_usage = response['usage']
    logger.info("Input tokens: %s", token_usage['inputTokens'])
    logger.info("Output tokens: %s", token_usage['outputTokens'])
    logger.info("Total tokens: %s", token_usage['totalTokens'])
    logger.info("Stop reason: %s", response['stopReason'])

    return response

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/chat', methods=['POST'])
def handle_chat():
    user_message = request.json.get('message')
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    system_prompts = [{"text": "You are an app that creates playlists for a radio station that plays rock and pop music. Only return song names and the artist."}]
    
    message_1 = {
        "role": "user",
        "content": [{"text": user_message}]
    }
    
    try:
        bedrock_client = boto3.client(service_name='bedrock-runtime')

        messages = [message_1]
        response = generate_conversation(bedrock_client, model_id, system_prompts, messages)
        
        output_message = response['output']['message']
        return jsonify({"response": output_message['content'][0]['text']})

    except ClientError as err:
        message = err.response['Error']['Message']
        logger.error("A client error occurred: %s", message)
        return jsonify({"error": message}), 500

if __name__ == '__main__':
    app.run(debug=True)
