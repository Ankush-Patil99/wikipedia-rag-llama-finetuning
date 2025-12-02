# 📚 Wikipedia RAG + LLaMA Project  
A complete end-to-end Retrieval-Augmented Generation (RAG) system built using:  
- FAISS vector search  
- MiniLM-L6 sentence embeddings  
- LLaMA language model  
- Wikipedia chunked dataset  
- Evaluation metrics + visualizations  

This project demonstrates how to build a production-style RAG pipeline from scratch —  
including data processing, embeddings, vector indexing, retrieval, LLM reasoning,  
evaluation, and analysis.  

---

## 🚀 Project Overview
This repository contains all artifacts required to fully reproduce a Wikipedia-based RAG system.  
It includes: dataset preparation, embedding generation, vector indexing with FAISS,  
retrieval logic, LLaMA integration, evaluation metrics, and exploratory visualizations.  

---

## 🧩 Components Included

### 1️⃣ **Dataset Preparation**
- Process Wikipedia JSONL dumps  
- Extract abstracts + section text  
- Clean + normalize text  
- Generate overlapping chunks (400-char size, 80-char overlap)  

### 2️⃣ **Embeddings**
- Model used: `all-MiniLM-L6-v2`  
- Batch inference for 500k+ chunks  
- Stored as `.npy` for fast loading  

### 3️⃣ **FAISS Indexing**
- Cosine-similarity search using `IndexFlatIP`  
- L2-normalized embedding vectors  
- Stored FAISS index: `wiki.index`  

### 4️⃣ **RAG Retrieval Pipeline**
- Embed user query  
- Perform FAISS top-k retrieval  
- Construct context-aware prompt  
- Generate response using LLaMA  

### 5️⃣ **Evaluation & Analysis**
- ROUGE-1, ROUGE-L  
- Exact match accuracy  
- Chunk-recall inspection  
- Multi-step refinement workflow  
- Visualizations: radar charts, bar charts, heatmaps  

---
flowchart LR

  %% -------------------- DATA PIPELINE --------------------
  A[📥 Raw Wikipedia JSONL] --> B[🧹 Preprocessing]
  B --> C[✂️ Chunking (400 chars, 80 overlap)]
  C --> D[📄 Chunked Dataset (CSV)]

  %% -------------------- EMBEDDINGS + INDEX --------------------
  D --> E[🧠 Embedding Model (MiniLM-L6-v2)]
  E --> F[🔢 Document Embeddings (.npy)]
  F --> G[🔎 Build FAISS Index (IndexFlatIP)]
  G --> H[💾 Save Index (wiki.index)]

  %% -------------------- RETRIEVAL + PROMPTING --------------------
  I[🧑‍💻 User Query] --> J[🔁 Query Embedding]
  J --> K[🔍 FAISS Top-K Search]
  K --> L[📚 Retrieved Chunks]
  L --> M[📝 Build RAG Prompt]

  %% -------------------- GENERATION --------------------
  M --> N[🦙 LLaMA Response Generation]
  N --> O[🗣️ Final Answer]

  %% -------------------- OPTIONAL FINE-TUNING --------------------
  F --> P[🛠 Prepare SFT Dataset]
  P --> Q[🔧 LoRA / QLoRA Fine-Tuning]
  Q --> N

  %% -------------------- EVALUATION --------------------
  O --> R[📏 Evaluation (ROUGE, EM)]
  R --> S[📊 Visualizations]

  %% -------------------- EXPORT --------------------
  S --> T[🌐 Upload to HuggingFace]
  S --> U[💻 Push to GitHub]

---

## 📊 Visualizations Included

- **[Chunk Length Distribution](https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/plots/chunk_length_distribution.png)**  
- **[Embedding Similarity Heatmap](https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/plots/embedding_similarity_heatmap.png)**    
- **[Simulated Training Loss](https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/plots/simulated_training_loss.png)**  

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

<details>
<summary><strong>📁 data/ — RAG results & SFT datasets</strong></summary>

| File | Description | Link |
|------|-------------|------|
| rag_evaluation_summary.csv | Evaluation summary | [Click Here](https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/data/rag_evaluation_summary.csv) |
| rag_final_summary.csv | Final summary metrics | [Click Here](https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/data/rag_final_summary.csv) |
| rag_optimized_results.csv | Optimized RAG results | [Click Here](https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/data/rag_optimized_results.csv) |
| rag_optimized_results_summary.csv | Optimized summary | [Click Here](https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/data/rag_optimized_results_summary.csv) |
| rag_refined_results.csv | Refined evaluation | [Click Here](https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/data/rag_refined_results.csv) |
| rag_refined_results_summary.csv | Refined evaluation summary | [Click Here](https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/data/rag_refined_results_summary.csv) |
| rag_results_summary.csv | Consolidated summary | [Click Here](https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/data/rag_results_summary.csv) |
| rag_test_results.csv | Test-time results | [Click Here](https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/data/rag_test_results.csv) |
| sft_dataset.jsonl | Raw SFT dataset | [Click Here](https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/data/sft_dataset.jsonl) |
| sft_dataset_clean.jsonl | Cleaned SFT dataset | [Click Here](https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/data/sft_dataset_clean.jsonl) |

</details>


<details>
<summary><strong>📁 embeddings/ — Document embeddings</strong></summary>

| File | Description | Link |
|------|-------------|------|
| doc_embeddings.npy | Numpy document embeddings | [Click Here](https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/embeddings/embeddings/doc_embeddings.npy) |

</details>


<details>
<summary><strong>📁 faiss_index/ — FAISS index & chunks</strong></summary>

| File | Description | Link |
|------|-------------|------|
| wiki.index | FAISS index file | [Click Here](https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/faiss_index/faiss_index/wiki.index) |
| wiki_chunks.csv | Chunk metadata | [Click Here](https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/faiss_index/faiss_index/wiki_chunks.csv) |

</details>


<details>
<summary><strong>📁 plots/ — Visualizations</strong></summary>

| Visualization | Link |
|--------------|------|
| Chunk Length Distribution | [Click Here](https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/plots/chunk_length_distribution.png) |
| Embedding Similarity Heatmap | [Click Here](https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/plots/embedding_similarity_heatmap.png) |
| Simulated Training Loss | [Click Here](https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/plots/simulated_training_loss.png) |

</details>


<details>
<summary><strong>📁 results_table/ — Evaluation tables</strong></summary>

| File | Description | Link |
|------|-------------|------|
| rag_evaluation_results.csv | Evaluation results table | [Click Here](https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/results_table/rag_evaluation_results.csv) |
| rag_final_evaluation_summary.csv | Final comparison summary | [Click Here](https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/results_table/rag_final_evaluation_summary.csv) |

</details>


<details>
<summary><strong>📁 src/ — Source code</strong></summary>

| Script | Description | Link |
|--------|-------------|------|
| chunking.py | Chunk generation pipeline | [Click Here](https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/src/chunking.py) |
| embeddings_faiss.py | Embedding + FAISS utilities | [Click Here](https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/src/embeddings_faiss.py) |
| evaluation.py | Evaluation engine | [Click Here](https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/src/evaluation.py) |
| preprocessing.py | Text preprocessing logic | [Click Here](https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/src/preprocessing.py) |
| rag_engine.py | Core RAG engine implementation | [Click Here](https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/src/rag_engine.py) |
| visualization.py | Plotting utilities | [Click Here](https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/src/visualization.py) |

</details>


---
## 📈 Results Summary
- Average ROUGE-1: **0.42**
- Average ROUGE-L: **0.39**
- Exact Match: **18%**
- Retrieval Recall (Top-5): **82%**
- Retrieval Recall (Top-10): **91%**

> These numbers show the system retrieves relevant Wikipedia chunks effectively,  
> and LLaMA generates context-aware summaries with strong overlap.

---

## 🏁 Conclusion
This project demonstrates a complete working implementation of a Wikipedia‑scale  
RAG system with retrieval, LLM response generation, evaluations, and visualization  
suitable for GitHub or professional portfolio display.
---
## 🔮 Future Improvements
- Add full LLaMA fine-tuning using LoRA / QLoRA
- Replace MiniLM with modern embedding models (e5-large, SFR-Embedding)
- Add reranking (Cross-Encoder or ColBERT)
- Deploy API via FastAPI + Docker
- Add streaming UI with Gradio

---

## 👤 Author
**Ankush Patil**  
Machine Learning & NLP Engineer  
📧 **Email**: ankpatil1203@gmail.com  
💼 **LinkedIn**: www.linkedin.com/in/ankush-patil-48989739a  
🌐 **GitHub**: https://github.com/Ankush-Patil99  
