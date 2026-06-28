import streamlit as st
import os
from pdf_loader import PDFLoader
from vector_store import VectorStore
from rag_model import RAGModel

# 설정
PDF_FOLDER = "data/"       # PDF 문서 폴더
FAISS_DB_PATH = "faiss_db/"  # FAISS 저장 경로
LMSTUDIO_API_URL = "http://172.31.251.119:1234/v1" #  "http://172.31.249.207:1234/v1"

# 필요한 디렉토리 생성
for directory in [PDF_FOLDER, FAISS_DB_PATH, "static"]:
    os.makedirs(directory, exist_ok=True)

try:
    # Streamlit 페이지 설정
    st.set_page_config(
        page_title="RAG QA System",
        page_icon="📚",
        layout="wide"
    )

    # PDF 문서 로딩
    @st.cache_resource
    def initialize_system():
        try:
            pdf_loader = PDFLoader(PDF_FOLDER)
            documents = pdf_loader.load_pdfs()
            
            # FAISS 벡터 DB 설정
            vector_store = VectorStore(FAISS_DB_PATH)
            if not os.path.exists(os.path.join(FAISS_DB_PATH, "index.faiss")):
                vector_store.create_vector_store(documents)
            vector_store.load_vector_store()
            
            # LM Studio 모델 설정
            rag_model = RAGModel(LMSTUDIO_API_URL)
            
            return vector_store, rag_model
        except Exception as e:
            st.error(f"시스템 초기화 중 오류 발생: {str(e)}")
            raise e

    vector_store, rag_model = initialize_system()
    
    # Streamlit UI
    st.title("📚 RAG QA System - PDF 문서 기반 검색")

    query = st.text_input("💡 질문을 입력하세요:")

    if query:
        with st.spinner("🔎 문서 검색 중..."):
            retrieved_docs = vector_store.search(query, k=3)
            answer = rag_model.generate_answer(query, retrieved_docs)

        st.write("## 📝 답변:")
        st.write(answer)

        st.write("## 📄 참고 문서:")
        for doc in retrieved_docs:
            st.write(doc.page_content[:500] + "...")
            
except Exception as e:
    st.error(f"오류가 발생했습니다: {str(e)}")
