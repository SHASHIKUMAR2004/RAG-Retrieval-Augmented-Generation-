import os
import csv
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd

# Directory containing files
input_directory = './data'

# CSV file to save extracted content
csv_file = './chunks.csv'

# Function to process PDF files
def process_pdf(file_path):
    reader = PdfReader(file_path)
    chunks = []
    for page in reader.pages:
        content = page.extract_text()
        if content:
            chunks.extend(split_content(content))
    return chunks

# Function to process DOCX files
def process_docx(file_path):
    doc = Document(file_path)
    full_text = '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
    return split_content(full_text)

# Function to process CSV files
def process_csv(file_path):
    df = pd.read_csv(file_path)
    content = df.astype(str).apply(lambda x: ' | '.join(x), axis=1).tolist()
    return content  # Each row becomes a chunk

# Chunking logic
def split_content(content):
    lines = content.split('\n')
    chunks = []
    chunk = ''
    for line in lines:
        if "Courses Registered" in line or "Student" in line:
            if chunk:
                chunks.append(chunk.strip())
            chunk = line
        else:
            chunk += '\n' + line
    if chunk:
        chunks.append(chunk.strip())
    return chunks

# Write to CSV
with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['content', 'source'])

    for filename in os.listdir(input_directory):
        filepath = os.path.join(input_directory, filename)

        if filename.endswith('.pdf'):
            chunks = process_pdf(filepath)
        elif filename.endswith('.docx'):
            chunks = process_docx(filepath)
        elif filename.endswith('.csv'):
            chunks = process_csv(filepath)
        else:
            continue  # Skip unsupported files

        for chunk in chunks:
            writer.writerow([chunk, filename])

print(f"✅ File conversion complete. Data saved to {csv_file}")
