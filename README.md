# L3 FINANCIAL AI AGENT
The L3 Financial AI Agent is a production-ready Retrieval-Augmented Generation (RAG) system that combines fine-tuned LLaMA-3 with real-time SEC filing analysis to answer complex financial questions about public companies. By integrating parameter-efficient fine-tuning, live EDGAR database access, and vector-based semantic search, the system delivers precise, context-aware responses grounded in actual regulatory filings, functioning like an AI financial analyst that can instantly read and interpret any company's 10-K report.

## METHODOLOGY

This project utilizes the LLaMA-3-8B-Instruct foundation model enhanced with domain-specific fine-tuning and multi-stage retrieval architecture. The methodology focuses on four integrated components:

1. **Model Fine-Tuning Pipeline**:
   - Applies LoRA (Low-Rank Adaptation) to adapt the base model using 10,000+ financial Q&A pairs
   - Targets all attention and MLP layers for comprehensive domain adaptation
   - Optimizes for accounting terminology, financial metrics, and regulatory language understanding
   - Achieves 2x faster training with 70% memory reduction via Unsloth optimization

2. **SEC Data Retrieval Engine**:
   - Connects directly to SEC EDGAR database via API integration
   - Automatically fetches latest 10-K filings for any US public company ticker
   - Extracts critical sections: 1A (Risk Factors) and 7 (Management Discussion & Analysis)
   - Combines sections into comprehensive analysis corpus

3. **Vector Database Architecture**:
   - Uses BGE-large-en-v1.5 embeddings to convert text chunks into dense vectors
   - Implements recursive chunking with 1,000-character segments and 500-character overlap
   - Builds FAISS GPU-accelerated index for millisecond similarity search
   - Maintains semantic relationships for accurate context retrieval

4. **Inference and Response Generation**:
   - Retrieves top-k relevant document chunks based on user query
   - Formats query and context into specialized prompt template
   - Generates concise, citation-ready responses citing specific details from filings
   - Parses output to extract clean, readable answers

Key financial features analyzed include revenue drivers, expense components, risk factors, margin impacts, currency effects, regulatory exposures, and forward-looking statements. Irrelevant marketing language and boilerplate text are filtered through embedding similarity scoring.

## PROPOSED IMPLEMENTATION



**Instructions to Follow for Proposed Implementation:**

- **API Configuration**: Obtain HuggingFace token with LLaMA-3 access and SEC-API.io key; configure as environment variables or directly in script initialization
- **Hardware Requirements**: Ensure CUDA-enabled GPU with minimum 8GB VRAM (T4, V100, or A100 recommended); verify GPU availability before model loading
- **Dependency Installation**: Install Unsloth with Colab-specific build, plus xFormers, TRL, PEFT, and SEC-API libraries; use GPU-accelerated FAISS version
- **Model Loading Sequence**: Load base LLaMA-3-8B-Instruct first, then apply LoRA adapters; use 4-bit quantization for inference to reduce memory footprint
- **Workflow Orchestration**: Execute fine-tuning phase first (60 steps, ~15 minutes), save adapters, then proceed to RAG pipeline setup
- **Data Flow Management**: Ensure SEC filing is fully downloaded and vectorized before starting Q&A loop; pre-embed entire document corpus for optimal retrieval speed
- **Error Handling**: Implement retry logic for SEC API timeouts; handle rate limiting with exponential backoff; validate ticker symbols before query execution
- **Scalability Design**: Parameterize ticker input for dynamic company analysis; design modular components to support future expansion to 10-Q and 8-K forms

## USAGE GUIDE

### Installation & Setup

**Prerequisites:**
- Python 3.8+
- CUDA-enabled GPU (T4, V100, or A100 with 8GB+ VRAM)
- HuggingFace Account with Llama-3 access approval
- SEC API Key from [sec-api.io](https://sec-api.io)

```bash
# Core ML libraries
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers trl peft accelerate bitsandbytes

# Financial data & RAG pipeline
pip install sec_api langchain langchain-community sentence-transformers faiss-gpu

# Utilities
pip install torch datasets transformers
