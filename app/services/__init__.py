"""
app.services package.

Core questionnaire parsing engines:
  - breakdown (v1: regex & deterministic fast-path)
  - breakdown_v2 (v2: skeleton-driven LLM-first)
  - breakdown_vlm (vlm: multimodal vision-first)
  - breakdown_langgraph (langgraph: sequential human-like reading & back-reading)
  - breakdown_preprocessing (NFKC normalization & bilingual table linearization)
  - breakdown_dlq (dead-letter queue)

Core & QA services:
  - llm (chat completion & vision inference)
  - embedding (multilingual embeddings)
  - translate (terminology translation)
  - retrieval (hybrid dual-track vector retrieval)
  - intent (Aho-Corasick & LLM intent classification)
  - ingest (questionnaire chunking & DB ingestion)
  - query_log (query history logger)
"""
