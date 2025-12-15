# Document ETL Pipeline (Python | AWS | NLP)

## Overview

This project implements an end-to-end ETL (Extract, Transform, Load) pipeline to process unstructured legal PDF documents and convert them into a clean, structured dataset. The pipeline extracts citations and term-definition pairs from multiple documents using a hybrid approach that combines rule-based techniques and NLP/ML methods.

The final output is a standardized, machine-readable JSON (and CSV) dataset suitable for downstream analytics or search applications.

---

## Key Features

* End-to-end ETL pipeline built using Python
* Processes 15 unstructured legal PDF documents
* Hybrid extraction approach:

  * Deterministic: Regex-based rule parsing
  * Non-deterministic: NLP / ML-assisted information extraction
* Canonicalization and standardization of legal references
* Object-Oriented Programming (OOP) based modular architecture
* Robust error handling and logging
* AWS integration (Amazon S3 for storage)
* Reproducible and well-documented workflow

---

## Tech Stack

* Programming Language: Python
* Core Libraries:

  * pdfplumber / PyPDF2 (PDF extraction)
  * re (Regular Expressions)
  * sentence-transformers / scikit-learn (NLP & ML)
  * numpy, pandas
  * boto3 (AWS S3 integration)
* Cloud: AWS S3
* Version Control: Git, GitHub

---

## Project Structure

```
ETL-Pipeline/
│
├── data/                 # Input PDFs and processed outputs
├── logs/                 # Runtime and error logs
├── src/                  # Source code (ETL modules)
│   ├── extractor.py      # PDF extraction logic
│   ├── transformer.py    # Cleaning, normalization, NLP logic
│   ├── loader.py         # JSON/CSV export and AWS upload
│   └── pipeline.py       # Orchestrates the ETL process
│
├── output/               # Final structured datasets
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation
└── main.py               # Entry point
```

---

## ETL Pipeline Workflow

1. Extract

   * Read and parse unstructured PDF documents
   * Identify raw text, citations, and definition candidates

2. Transform

   * Clean and normalize extracted text
   * Apply regex-based rules for citation detection
   * Use NLP/ML techniques to improve extraction accuracy
   * Canonicalize legal references (example: fed_decree_law_47_2022)

3. Load

   * Export structured data to JSON and CSV formats
   * Upload outputs to AWS S3
   * Log execution details and errors

---

## Output

* Structured JSON dataset containing:

  * Document metadata
  * Extracted citations
  * Term-definition pairs
* Optional CSV exports for analytics use cases

---

## How to Run the Project

### 1. Clone the Repository

```
git clone https://github.com/yaseenmohd2001/ETL-Pipeline.git
cd ETL-Pipeline
```

### 2. Create a Virtual Environment (Optional but Recommended)

```
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

### 4. Configure AWS Credentials

Ensure your AWS credentials are configured using one of the following methods:

* AWS CLI configuration (`aws configure`)
* Environment variables

### 5. Run the ETL Pipeline

```
python main.py
```

---

## Logging and Error Handling

* Runtime performance logs are captured
* Exceptions are logged for debugging and traceability
* Ensures pipeline stability when processing multiple documents

---

## Project Outcome

* Successfully processed multiple legal documents
* Extracted citations and term-definition pairs
* Generated clean, standardized, machine-readable datasets
* Built a reproducible, end-to-end ETL pipeline following industry best practices

---

## Future Improvements

* Add Docker support for deployment
* Extend NLP models for higher extraction accuracy
* Integrate search or knowledge graph capabilities
* Deploy pipeline using AWS Lambda or EC2

---

## Author

Yaseen Mohd

GitHub: [https://github.com/yaseenmohd2001](https://github.com/yaseenmohd2001)

---

## License

This project is for educational and portfolio purposes.
