#  Retail Analytics RAG Chatbot

This project is a Retail Analytics Chatbot built using Python and Streamlit.  
It helps users ask business questions related to sales, customers, churn risk, and product performance.

---

## Features

- Customer Segmentation
- Churn Risk Analysis
- Sales Trends
- Top Products & Customers

---

## Technologies Used

- Python
- Pandas
- Sentence Transformers
- FAISS (or similarity search)
- Streamlit

---

## How it works

1. Retail data is processed and converted into business insights  
2. Insights are stored in a knowledge base  
3. User asks a question  
4. System finds the most relevant answer using similarity search  

---

## Run locally

```bash
streamlit run app.py