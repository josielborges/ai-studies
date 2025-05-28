from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import boto3

def create_investment_pdf(file_path):
    c = canvas.Canvas(file_path)

    c.setFont("Helvetica", size=12)
    c.drawString(100, 750, "Investment Report")

    investments = [
        {"investor": "Josiel", "type": "Stocks", "value": "10.000", "Date" : "01/01/2024"},
        {"investor": "Jussara", "type": "FII", "value": "12.000", "Date" : "01/04/2024"},
        {"investor": "Pedro", "type": "Stocks", "value": "500", "Date" : "01/03/2024"},
    ]

    y = 720

    for inv in investments:
        c.drawString(100, y, f"Name: {inv['investor']}")
        c.drawString(100, y - 15, f"Type: {inv['type']}")
        c.drawString(100, y - 30, f"Value: {inv['value']}")
        c.drawString(100, y - 45, f"Date: {inv['Date']}")
        y -= 75

    c.save()

pdf_file_path = "investments_report.pdf"

create_investment_pdf(pdf_file_path)

s3 = boto3.client('s3')
bucket_name = 'josiel-learning'
object_name = pdf_file_path

s3.upload_file(pdf_file_path, bucket_name, object_name)

print(f"PDF {pdf_file_path} load with success at s3://{bucket_name}/{object_name}")