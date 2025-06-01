# rag_quote_retrival

# 🧠 RAG-Based Semantic Quote Retrieval and Structured QA

This project implements a Retrieval-Augmented Generation (RAG) system to retrieve and answer natural language queries over a dataset of quotes. The pipeline fine-tunes a sentence transformer on the [Abirate/english_quotes](https://huggingface.co/datasets/Abirate/english_quotes) dataset, builds a FAISS-based retrieval system, evaluates the results using RAGAS, and provides a fully interactive Streamlit application.

---

## 📌 Objective

- Fine-tune a model for semantic understanding of quotes.
- Build a RAG system for contextual quote retrieval and question answering.
- Evaluate the RAG system using the **RAGAS** framework.
- Deploy an interactive **Streamlit** app with structured JSON responses.
