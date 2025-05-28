import boto3

comprehend = boto3.client('comprehend', region_name='us-east-2')

with open("extracted_text.txt", "r") as f:
    text = f.read()

sentiment_response = comprehend.detect_sentiment(
    Text=text,
    LanguageCode='pt'
)

print(sentiment_response['Sentiment'])


entities_response= comprehend.detect_entities(
    Text=text,
    LanguageCode='pt'
)


print("Entities found:")
for entity in entities_response['Entities']:
    print(f" - {entity['Text']}: {entity['Type']}")

key_phrases_response = comprehend.detect_key_phrases(
    Text=text,
    LanguageCode='pt'
)


print("Key Phrases Found:")
for key_phrase in key_phrases_response['KeyPhrases']:
    print(f" - {key_phrase['Text']}")
    print("-"*12)