# ML Research Assistant

A RAG system that answers technical questions across 10 foundational ML papers.

## Results

| Metric | Score |
|--------|-------|
| Recall@1 | 96.7% |
| Recall@3 | 100% |
| Recall@5 | 100% |
| Faithfulness | 84.0% |
| Answer Relevancy | 64.7% |

## Papers Covered
- Attention Is All You Need
- BERT, GPT-2
- LoRA, QLoRA
- RAG (Lewis et al.)
- FlashAttention
- Llama 2, Mistral
- Chain of Thought

## Stack
- Embeddings: MiniLM (384d)
- Chunking: Recursive (512 tokens, 50 overlap)
- Generation: Claude Haiku
- Eval: LLM-as-judge + recall@k on 30 held-out queries

## Key Finding
Recursive chunking outperforms fixed-size chunking by 3.4% on recall@1
while using a 5x smaller embedding model — lower cost, same quality.

## Usage
```bash
pip install -r requirements.txt

# First time setup
python rag_complete.py --setup

# Launch demo
python rag_complete.py --demo

# Launch API
python rag_complete.py --api
```

## Example Questions
- How does LoRA reduce trainable parameters?
- What is the key contribution of FlashAttention?
- How does RAG combine retrieval with generation?

## Live Demo
https://ml-research-assistant-production.up.railway.app/
