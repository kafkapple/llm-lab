from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from pathlib import Path
import pickle
import json
import torch
from typing import List, Dict, Any, Optional
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import numpy as np
import logging
import os
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import faiss
from sklearn.metrics.pairwise import cosine_similarity
import time
from src.util import sanitize_model_name
from src.logger import setup_logging
from src.data_prep import load_data
from src.eval import save_metrics
from src.util import send_telegram_message
import asyncio
import re
import traceback

class EmotionRAG:
    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.embedding_model = cfg.rag.embedding_model
        self.db_path = Path(cfg.rag.db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.vector_store = None
        
        # 임베딩 모델 초기화 추가
        try:
            self.logger.info(f"Initializing embedding model: {self.embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cuda'},
                encode_kwargs={'device': 'cuda', 'batch_size': 32}
            )
            self.logger.info("Embedding model initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise ValueError(f"Embedding model initialization failed: {str(e)}")
        
        self.cheat_mode = getattr(cfg.rag, 'cheat', False)
        if self.cheat_mode:
            self.logger.info("RAG cheat mode enabled - will always retrieve query text itself with ground truth label")
            self.cfg.rag.k_examples = 1  # cheat 모드에서는 k=1로 고정
        
        # Initialize reranker
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.rerank_tokenizer = AutoTokenizer.from_pretrained(cfg.rag.reranking_model)
            self.rerank_model = AutoModelForSequenceClassification.from_pretrained(cfg.rag.reranking_model)
            if self.logger:
                self.logger.info("Reranker initialized successfully")
            else:
                print("Reranker initialized successfully")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to initialize reranker: {e}")
            else:
                print(f"Warning: Failed to initialize reranker: {e}")
            self.rerank_tokenizer = None
            self.rerank_model = None

        instructions = cfg.prompt.modules.instruction
        definition = cfg.prompt.modules.definition
        output_format = cfg.prompt.modules.output_format
        # 기본 RAG 프롬프트 템플릿
        self.rag_prompt = f"""{instructions}
        {definition}
        ## Retrieved Context:
        Consider the following similar texts and their emotion classifications:
        {'{examples}'}

        {output_format}
        """

        # Reranking 프롬프트 템플릿
        self.rerank_rag_prompt = f"""{instructions}
        {definition}
        ## Retrieved and Reranked Context:
        Consider the following semantically similar texts and their emotion classifications:
        {'{examples}'}

        {output_format}
        """

        # Chain-of-Thought 프롬프트 템플릿
        self.cot_rag_prompt = f"""{instructions}
        {definition}
        ## Chain-of-Thought Analysis
        Let's analyze this emotion step by step using similar examples from our database.

        ## Retrieved Context:
        {'{examples}'}

        ## Analysis Steps:
        1. First, identify key emotional indicators in the text and examples
        2. Compare with similar cases
        3. Step-by-step reasoning
        4. Final classification with confidence assessment

        {output_format}
        """

        # Reranking + Chain-of-Thought 프롬프트 템플릿
        self.cot_rerank_rag_prompt = f"""{instructions}
        {definition}
        ## Chain-of-Thought Analysis with Reranked Examples
        Let's analyze this emotion step by step using semantically reranked examples.

        ## Retrieved and Reranked Context:
        The following examples have been reranked based on semantic similarity:
        {'{examples}'}

        ## Enhanced Analysis Steps:
        1. Initial Assessment:
           - Identify key emotional indicators
           - Note relationships to reranked examples
           - Consider semantic similarity scores

        2. Example-based Analysis:
           - Examine patterns in similar examples
           - Consider ranking relevance
           - Extract emotional patterns

        3. Final Classification:
           - Select primary emotion based on evidence
           - Assess confidence using similarity patterns
           - Provide detailed reasoning chain

        {output_format}
        """

        # 캐시 초기화 추가
        self.embedding_cache = {}
        self.similar_examples_cache = {}
        self.logger.info("Initialized embedding and similar examples cache")
        
        # 캐시 크기 제한 설정
        self.max_cache_size = getattr(cfg.rag, 'max_cache_size', 10000)

    def create_index(self, df):
        """벡터 DB 생성 또는 로드"""
        embedding_model = str(self.cfg.rag.embedding_model)
        embedding_model = embedding_model.split("/")[-1]

        db_file = self.db_path / f"{self.cfg.data.name}_{embedding_model}_vectorstore.pkl"
        
        # 기존 DB 로드 시도
        if self.cfg.rag.load_db and db_file.exists():
            print(f"Loading existing vector DB from {db_file}")
            try:
                with open(db_file, 'rb') as f:
                    self.vector_store = pickle.load(f)
                return
            except Exception as e:
                print(f"Error loading vector DB: {e}")
        
        # 새로운 DB 생성
        print("Creating new vector DB...")
        texts = df['text'].tolist()
        metadatas = [
            {
                'emotion': row.emotion,
                'index': idx
            } for idx, row in df.iterrows()
        ]
        
        self.vector_store = FAISS.from_texts(
            texts,
            self.embeddings,
            metadatas=metadatas
        )
        
        # DB 저장
        if self.cfg.rag.save_db:
            print(f"Saving vector DB to {db_file}")
            self.db_path.mkdir(parents=True, exist_ok=True)
            with open(db_file, 'wb') as f:
                pickle.dump(self.vector_store, f)
    
    def rerank_examples(self, query: str, examples: list, top_k: int = 3) -> list:
        """Rerank retrieved examples using cross-encoder"""
        if not self.rerank_model or not self.rerank_tokenizer:
            if hasattr(self, 'logger'):
                self.logger.warning("Reranker not available, skipping reranking")
            else:
                print("Reranker not available, skipping reranking")
            return examples[:top_k]
        
        try:
            if hasattr(self, 'logger'):
                self.logger.debug("\n=== Reranking Debug Info ===")
                self.logger.debug(f"Query: {query}")
                self.logger.debug(f"Number of examples to rerank: {len(examples)}")
            else:
                print("\n=== Reranking Debug Info ===")
                print(f"Query: {query}")
                print(f"Number of examples to rerank: {len(examples)}")
            
            # Prepare candidate texts and store original similarity scores
            candidates = []
            original_scores = {}  # 원래의 similarity score 저장
            for doc, score in examples:
                example_text = f"Text: {doc.page_content} | Emotion: {doc.metadata.get('emotion', 'unknown')}"
                candidates.append((example_text, doc))
                original_scores[doc.page_content] = score  # 원본 텍스트를 키로 사용
            
            # Prepare inputs for reranking
            pairs = [f"Query: {query} | Candidate: {cand[0]}" for cand in candidates]
            inputs = self.rerank_tokenizer(pairs, return_tensors="pt", padding=True, truncation=True)
            
            # Get reranking scores
            with torch.no_grad():
                scores = self.rerank_model(**inputs).logits.squeeze()
                if len(scores.shape) == 0:
                    scores = scores.unsqueeze(0)
            
            # Combine with original examples and reranking scores
            reranked = [(candidates[i][1], float(scores[i])) for i in range(len(candidates))]
            reranked.sort(key=lambda x: x[1], reverse=True)
            
            # Get top_k reranked examples
            top_reranked = reranked[:top_k]
            
            # Count emotions in top reranked examples
            emotion_counts = {}
            for doc, _ in top_reranked:
                emotion = doc.metadata.get('emotion', 'unknown')
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # Find the most common emotion
            most_common_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
            
            if hasattr(self, 'logger'):
                self.logger.debug("\nEmotion counts in top examples:")
                self.logger.debug(emotion_counts)
                self.logger.debug(f"Most common emotion: {most_common_emotion}")
            else:
                print("\nEmotion counts in top examples:")
                print(emotion_counts)
                print(f"Most common emotion: {most_common_emotion}")
            
            # Filter examples with most common emotion and use original similarity scores
            filtered_results = []
            for doc, _ in top_reranked:
                if doc.metadata.get('emotion') == most_common_emotion:
                    original_score = original_scores[doc.page_content]
                    filtered_results.append((doc, original_score))
            
            if hasattr(self, 'logger'):
                self.logger.debug("\nFiltered Results:")
                for i, (doc, score) in enumerate(filtered_results):
                    self.logger.debug(f"Rank {i+1}: {doc.page_content[:100]}... | Emotion: {doc.metadata.get('emotion')} | Original Score: {score:.4f}")
                self.logger.debug("=== End Reranking Debug Info ===\n")
            else:
                print("\nFiltered Results:")
                for i, (doc, score) in enumerate(filtered_results):
                    print(f"Rank {i+1}: {doc.page_content[:100]}... | Emotion: {doc.metadata.get('emotion')} | Original Score: {score:.4f}")
                print("=== End Reranking Debug Info ===\n")
            
            return filtered_results
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error in reranking: {e}")
            else:
                print(f"Error in reranking: {e}")
            return examples[:top_k]
    
    def exclude_index(self, index: int):
        """현재 쿼리에 대해 특정 인덱스를 검색에서 제외합니다.
        
        Args:
            index (int): 제외할 인덱스
        """
        # 매번 excluded_indices를 초기화하고 현재 인덱스만 설정
        self.exclude_indices = {index}
        if self.logger:
            self.logger.debug(f"Set excluded index: {index}")
            
    def _get_cached_embedding(self, text: str) -> np.ndarray:
        """텍스트의 임베딩을 캐시에서 가져오거나 새로 생성"""
        if text not in self.embedding_cache:
            if len(self.embedding_cache) >= self.max_cache_size:
                # 가장 오래된 항목 제거 (FIFO)
                self.embedding_cache.pop(next(iter(self.embedding_cache)))
            
            self.embedding_cache[text] = self.embeddings.embed_query(text)
            if self.logger:
                self.logger.debug(f"Created new embedding for text: {text[:50]}...")
        return self.embedding_cache[text]
    
    def get_similar_examples(self, query_text: str, query_emotion: str, k: int = None, threshold: float = None) -> List[Document]:
        """유사한 예제를 검색합니다."""
        if self.cheat_mode:
            doc = Document(
                page_content=query_text,
                metadata={
                    'emotion': query_emotion,
                    'similarity': 1.0,
                    'is_cheat': True
                }
            )
            return [(doc, 1.0)]  # (Document, score) 튜플 형태로 반환
        
        k = k or self.cfg.rag.k_examples
        threshold = threshold or self.cfg.rag.threshold
        
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call create_index first.")
            
        try:
            # 정확한 메서드 이름 사용
            docs_and_scores = self.vector_store.similarity_search_with_score(
                query_text,
                k=k*2  # threshold 필터링을 위해 더 많은 후보 검색
            )
            
            # threshold 기반 필터링 및 처리
            processed_results = []
            for doc, score in docs_and_scores:
                if score >= threshold:
                    if not hasattr(self, 'exclude_indices') or doc.metadata.get('index') not in self.exclude_indices:
                        processed_results.append((doc, float(score)))
            
            # k개로 제한
            processed_results = processed_results[:k]
            
            if self.logger:
                self.logger.debug(f"Retrieved {len(processed_results)} examples with threshold {threshold}")
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Error in similarity search: {str(e)}")
            self.logger.error(f"Parameters - k: {k}, threshold: {threshold}")
            return []

    def analyze_retrieval_performance(self, output_dir: str):
        """검색 결과의 감정 클래스 매칭 분석"""
        if not hasattr(self, 'search_results') or not self.search_results:
            print("No search results available for analysis")
            return None
        
        try:
            df = pd.DataFrame(self.search_results)
            
            # 전체 검색 성능 분석
            total_matches = df['emotion_match'].sum()
            total_queries = len(df['query_text'].unique())
            total_retrievals = len(df)
            
            # 클래스별 성능 분석
            class_performance = {}
            for emotion in df['query_emotion'].unique():
                emotion_df = df[df['query_emotion'] == emotion]
                matches = emotion_df['emotion_match'].sum()
                total = len(emotion_df)
                accuracy = matches / total if total > 0 else 0
                
                class_performance[emotion] = {
                    'total_queries': len(emotion_df['query_text'].unique()),
                    'total_retrievals': total,
                    'correct_matches': matches,
                    'accuracy': accuracy
                }
            
            # 결과 저장
            performance_stats = {
                'overall': {
                    'total_queries': total_queries,
                    'total_retrievals': total_retrievals,
                    'correct_matches': total_matches,
                    'accuracy': total_matches / total_retrievals if total_retrievals > 0 else 0
                },
                'by_class': class_performance
            }
            
            # 통계 저장
            stats_path = Path(output_dir) / 'rag_retrieval_stats.json'
            with open(stats_path, 'w') as f:
                json.dump(performance_stats, f, indent=2)
                
            # 상세 분석 결과 출력
            print("\nRAG Retrieval Performance Analysis")
            print("="*50)
            print(f"Overall accuracy: {performance_stats['overall']['accuracy']:.2%}")
            print(f"Total queries: {total_queries}")
            print(f"Total retrievals: {total_retrievals}")
            print("\nClass-wise Performance:")
            for emotion, stats in class_performance.items():
                print(f"\n{emotion}:")
                print(f"  Accuracy: {stats['accuracy']:.2%}")
                print(f"  Queries: {stats['total_queries']}")
                print(f"  Retrievals: {stats['total_retrievals']}")
                print(f"  Correct matches: {stats['correct_matches']}")
                
            return performance_stats
            
        except Exception as e:
            print(f"Error analyzing retrieval performance: {str(e)}")
            return None

    def save_search_results(self, output_path: str):
        """검색 결과를 CSV 파일로 저장"""
        if hasattr(self, 'search_results') and self.search_results:
            df = pd.DataFrame(self.search_results)
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"Saved search results to {output_path}")
            # 저장 후 초기화
            self.search_results = []

    def get_rag_prompt(self, query_text: str, similar_examples: List[Document]) -> str:
        """RAG 프롬프트를 생성합니다."""
        try:
            # cheat 모드 체크
            is_cheat = any(hasattr(doc, 'metadata') and doc.metadata.get('is_cheat', False) for doc in similar_examples)
            
            if is_cheat:
                # cheat 모드에서는 단일 예제만 사용
                example = similar_examples[0]
                example_text = f"Text: {example.page_content}\nEmotion: {example.metadata.get('emotion', '')}\nSimilarity: 1.0"
            else:
                # 일반 모드에서는 유사도 점수와 함께 처리
                example_texts = []
                for doc, score in similar_examples:
                    example_texts.append(f"Text: {doc.page_content}\nEmotion: {doc.metadata.get('emotion', '')}\nSimilarity: {score:.2f}")
                example_text = "\n\n".join(example_texts)

            # 템플릿 이름 정규화 (하이픈을 언더스코어로 변환)
            template_name = self.cfg.model.template.replace("-", "_").lower()
            
            # 템플릿 선택
            if "rerank" in template_name and "cot" in template_name:
                template = self.cot_rerank_rag_prompt
            elif "rerank" in template_name:
                template = self.rerank_rag_prompt
            elif "cot" in template_name:
                template = self.cot_rag_prompt
            else:
                template = self.rag_prompt
            
            # 프롬프트 생성
            final_prompt = template.replace("{examples}", example_text) + f"Text: {query_text}"
            
            return final_prompt
            
        except Exception as e:
            self.logger.error(f"Error in get_rag_prompt: {str(e)}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise 