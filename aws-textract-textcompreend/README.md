## Textract and Comprehend with AWS

AWS Textract and AWS Comprehend are powerful tools that enable automated document processing and natural language understanding. Textract extracts text, tables, and forms from scanned documents using OCR (Optical Character Recognition), making it ideal for digitizing paper-based workflows. It can detect key-value pairs and structured data with high accuracy, even from complex layouts.

Comprehend, on the other hand, is a natural language processing (NLP) service that analyzes the extracted text to identify entities, key phrases, sentiment, language, and custom classifications. When combined, Textract and Comprehend provide a complete pipeline for extracting and understanding unstructured data, helping organizations gain valuable insights and automate decision-making processes.

These services are scalable, serverless, and easily integrated into applications via AWS SDKs and APIs.

### Notes

These permissions are needed to execute the codes:

- AmazonS3FullAccess
- AmazonTextractFullAccess
- ComprehendFullAccess

Add on user at https://us-east-1.console.aws.amazon.com/iam/home#/home