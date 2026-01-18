import boto3
import json

client = boto3.client('sagemaker-runtime', region_name='eu-central-1')

prompt = "WTA tennis Grand Slam tournaments winners list:"
response = client.invoke_endpoint(
    EndpointName='sports-llm-endpoint',
    ContentType='application/json',
    Body=json.dumps({'prompt': prompt, 'max_new_tokens': 500, 'temperature': 0.7})
)
result = json.loads(response['Body'].read().decode())
#print('Result:' + str(result))
print('Prompt: ' + prompt)
print(f'Output: {result["generated_text"]}')