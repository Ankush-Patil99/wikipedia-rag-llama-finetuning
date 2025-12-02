# 📚 Wikipedia RAG + LLaMA Project  
A complete end-to-end Retrieval-Augmented Generation system built using:  
-  FAISS Vector Search  
-  MiniLM-L6 Embeddings  
-  LLaMA Language Model  
-  Wikipedia Chunked Dataset  
-  Evaluation Metrics + Visualizations  

---

## 🚀 Project Overview
This repository contains all artifacts required to reproduce a Wikipedia-based RAG system.  
It includes dataset preparation, embedding generation, vector indexing, retrieval,  
LLM integration, evaluation metrics, fine‑tuning attempts, and visualizations.

---

## 🧩 Components Included

### 1️⃣ **Dataset Preparation**
- JSONL Wikipedia files processed  
- Extracted abstracts + section text  
- Cleaned + normalized text  
- Chunking (400 chars, 80 overlap)
### 2️⃣ **Embeddings**
- Model: `all-MiniLM-L6-v2`  
- Batch encoding  
- 500k chunk embeddings stored  
- Saved as `.npy`  
### 3️⃣ **FAISS Indexing**
- Built using cosine similarity (IndexFlatIP)  
- Normalized vectors  
- Stored: `wiki.index`
### 4️⃣ **RAG Retrieval**
- Embed user query  
- FAISS top‑k search  
- Build context-aware prompt  
- Pass to LLaMA generator
### 5️⃣ **Evaluation**
- ROUGE‑1, ROUGE‑L  
- Exact match  
- Chunk recall inspection  
- Multi-step refinement  
- Radar charts, bar charts, heatmaps  

---

## 📊 Visualizations Included
- Chunk length distribution  
- Embedding similarity heatmap  
- RAG evaluation radar chart  
- Final comparison bar charts  
- Pipeline flow diagram  

---

## 🛠 Technologies Used
- **Transformers**
- **FAISS**
- **SentenceTransformers**
- **PyTorch / CUDA**
- **Matplotlib / Seaborn**
- **Hugging Face Hub**

---

## 📦 Repository Structure
```
📁 embeddings/
📁 faiss_index/
📁 results/
📁 visuals/
📄 wikipedia_chunks.csv
📄 sft_dataset.jsonl
📄 rag_optimized_results.csv
📄 rag_final_evaluation_summary.csv
```

---

## 🏁 Conclusion
This project demonstrates a complete working implementation of a Wikipedia‑scale  
RAG system with retrieval, LLM response generation, evaluations, and visualization  
suitable for GitHub or professional portfolio display.

---

## 👤 Author
**Ankush Patil**  
Machine Learning & NLP Engineer  
GitHub / Kaggle / HuggingFace: *Your profiles here*
