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
| rag_evaluation_summary.csv | Evaluation summary | https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/data/rag_evaluation_summary.csv |
| rag_final_summary.csv | Final summary metrics | https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/data/rag_final_summary.csv |
| rag_optimized_results.csv | Optimized RAG results | https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/data/rag_optimized_results.csv |
| rag_optimized_results_summary.csv | Optimized summary | https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/data/rag_optimized_results_summary.csv |
| rag_refined_results.csv | Refined evaluation | https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/data/rag_refined_results.csv |
| rag_refined_results_summary.csv | Refined evaluation summary | https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/data/rag_refined_results_summary.csv |
| rag_results_summary.csv | Consolidated summary | https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/data/rag_results_summary.csv |
| rag_test_results.csv | Test-time results | https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/data/rag_test_results.csv |
| sft_dataset.jsonl | Raw SFT dataset | https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/data/sft_dataset.jsonl |
| sft_dataset_clean.jsonl | Cleaned SFT dataset | https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/data/sft_dataset_clean.jsonl |

</details>

<details>
<summary><strong>📁 embeddings/ — Document embeddings</strong></summary>

| File | Description | Link |
|------|-------------|------|
| doc_embeddings.npy | Numpy document embeddings | https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/embeddings/embeddings/doc_embeddings.npy |

</details>


<details>
<summary><strong>📁 faiss_index/ — FAISS index & chunks</strong></summary>

| File | Description | Link |
|------|-------------|------|
| wiki.index | FAISS index file | https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/faiss_index/faiss_index/wiki.index |
| wiki_chunks.csv | Chunk metadata | https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/faiss_index/faiss_index/wiki_chunks.csv |

</details>


<details>
<summary><strong>📁 plots/ — Visualizations</strong></summary>

| Visualization | Link |
|--------------|------|
| Chunk Length Distribution | https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/plots/chunk_length_distribution.png |
| Embedding Similarity Heatmap | https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/plots/embedding_similarity_heatmap.png |
| Simulated Training Loss | https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/plots/simulated_training_loss.png |

</details>


<details>
<summary><strong>📁 results_table/ — Evaluation tables</strong></summary>

| File | Description | Link |
|------|-------------|------|
| rag_evaluation_results.csv | Evaluation results table | https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/results_table/rag_evaluation_results.csv |
| rag_final_evaluation_summary.csv | Final comparison summary | https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/results_table/rag_final_evaluation_summary.csv |

</details>


<details>
<summary><strong>📁 src/ — Source code</strong></summary>

| Script | Description | Link |
|--------|-------------|------|
| chunking.py | Chunk generation pipeline | https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/src/chunking.py |
| embeddings_faiss.py | Embedding + FAISS utilities | https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/src/embeddings_faiss.py |
| evaluation.py | Evaluation engine | https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/src/evaluation.py |
| preprocessing.py | Text preprocessing logic | https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/src/preprocessing.py |
| rag_engine.py | Core RAG engine implementation | https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/src/rag_engine.py |
| visualization.py | Plotting utilities | https://huggingface.co/ankpatil1203/Wikipedia-RAG-LLAMA-Project/blob/main/wikipedia-rag-llama-finetuning/src/visualization.py |

</details>


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
