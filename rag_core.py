# rag_core.py
import os
from typing import List, Tuple
from dotenv import load_dotenv  # 加载.env文件

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --- 加载环境变量 ---
load_dotenv()

# --- 配置 ---
# 选择嵌入模型: "openai" 或 "huggingface"
EMBEDDING_MODEL = "huggingface"
# 选择LLM: "openai" 或 "ollama"
LLM_MODEL = "ollama"    #使用ollama 本地运行
# Ollama 模型名称
OLLAMA_MODEL_NAME = "qwen:7b"  # 或者 "llama3", "mistral" 等
# ChromaDB 持久化路径 (Windows路径)
CHROMA_DB_PATH = os.path.join(os.getcwd(), "vector_db")


class RAGEngine:
    def __init__(self):
        self.embedding = self._get_embedding_model()
        self.vector_store = None
        self.qa_chain = None
        self.llm = self._get_llm_model()

    def _get_embedding_model(self):
        """初始化嵌入模型"""
        if EMBEDDING_MODEL == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in .env file.")
            return OpenAIEmbeddings(openai_api_key=api_key)
        else:  # 默认使用 HuggingFace
            model_name = "shibing624/text2vec-base-chinese"
            return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})

    def _get_llm_model(self):
        """初始化大语言模型"""
        if LLM_MODEL == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in .env file.")
            return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)
        else:  # 默认使用 Ollama
            try:
                from langchain_community.llms import Ollama
                return Ollama(model=OLLAMA_MODEL_NAME)
            except ImportError:
                raise ImportError(
                    "Ollama is not installed. Please install it with 'pip install langchain-community' and ensure Ollama is running.")

    def load_and_process_documents(self, file_path: str):
        """加载文档，分块，并构建向量库"""
        print(f"Loading and processing {file_path}...")
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError("Unsupported file type. Please use PDF, DOCX, or TXT.")

        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(
                documents=texts,
                embedding=self.embedding,
                persist_directory=CHROMA_DB_PATH
            )
        else:
            self.vector_store.add_documents(texts)

        print("Documents processed and vector store updated.")
        self.create_qa_chain()

    def load_existing_vector_store(self):
        """加载已存在的向量库"""
        if os.path.exists(CHROMA_DB_PATH):
            self.vector_store = Chroma(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=self.embedding
            )
            self.create_qa_chain()
            print("Existing vector store loaded.")
            return True
        return False

    def create_qa_chain(self):
        """创建问答链"""
        if self.vector_store is None or self.llm is None:
            raise ValueError("Vector store or LLM is not initialized.")

        template = """请根据以下提供的背景信息，来回答用户的问题。
        如果你不知道答案，就说你不知道，不要试图编造答案。
        答案应该简洁明了，并尽量使用中文。

        背景信息：
        {context}

        问题：
        {question}

        有帮助的答案："""

        PROMPT = PromptTemplate(
            template=template, input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

    def query(self, question: str) -> Tuple[str, List]:
        """进行查询"""
        if not self.qa_chain:
            return "请先上传文档以初始化知识库。", []

        result = self.qa_chain.invoke({"query": question})
        answer = result.get('result', '抱歉，我无法回答这个问题。')
        source_documents = result.get('source_documents', [])

        return answer, source_documents


# 全局RAG引擎实例
rag_engine = RAGEngine()
# 在服务启动时尝试加载已存在的向量库
rag_engine.load_existing_vector_store()