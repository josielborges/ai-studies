from docx import Document
from transformers import pipeline
import os

summarizer = pipeline("summarization")

def read_docx(file_path):

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    doc = Document(file_path)
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    return "\n".join(text)

def summarize_text(text, max_length=500, min_length=30, do_sample=False):
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=do_sample)
    return summary[0]['summary_text']

def save_summary(summary, output_path):
    with open(output_path, 'w') as f:
        f.write(summary)

if __name__ == "__main__":
    docx_file_path = "data/doc_input.docx"
    output_file_path = "data/summary.txt"

    full_text = read_docx(docx_file_path)
    summary = summarize_text(full_text)

    save_summary(summary, output_file_path)

    print(f"Summary saved to {output_file_path}")