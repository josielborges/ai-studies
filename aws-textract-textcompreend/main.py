import boto3
from botocore.exceptions import EndpointConnectionError

textract = boto3.client('textract', region_name='us-east-2')

try:
    response = textract.analyze_document(
        Document={
            'S3Object': {
                'Bucket': 'josiel-learning',
                'Name': 'investments_report.pdf'
            }
        },
        FeatureTypes=['TABLES', 'FORMS'])


    def get_text_from_blocs(blocks):
        lines = [block['Text'] for block in blocks if block['BlockType'] == 'LINE']
        return '\n'.join(lines)


    text = get_text_from_blocs(response['Blocks'])

    print(text)

    with open("extracted_text.txt", "w") as f:
        f.write(text)

    print("Text extracted and saved to extracted_text.txt")
except EndpointConnectionError:
    print("Error: Could not connect to Textract service")
except Exception as e:
    print(f"Error: {e}")
