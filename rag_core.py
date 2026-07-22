import os
import sys

# 解决可能出现的控制台中文乱码问题
sys.stdout.reconfigure(encoding='utf-8')

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ================= 1. 配置 API Key =================
# 这里请替换成你刚才申请的 DeepSeek API Key
os.environ["OPENAI_API_KEY"] = "你的_API_KEY_放这里"
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com"

class RAGSystem:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        
        print("🤖 [系统] 正在加载 Embedding 模型 (首次运行会自动下载模型，请耐心等待1-3分钟)...")
        # 使用 BGE 模型生成 Embedding (这是简历的核心亮点之一)
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
        
        # 初始化大模型 (调用 DeepSeek)
        self.llm = ChatOpenAI(model="deepseek-chat", temperature=0)
        self.vector_store = None

    def ingest_document(self, file_path):
        """读取文档、分块并存入向量数据库"""
        print(f"📄 [文档] 正在解析: {file_path}")
        # 1. 加载PDF文档 
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # 2. 文本分块 (Chunking)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,  # 每个块大约400个字符
            chunk_overlap=50 # 重叠50个字符防断句
        )
        splits = text_splitter.split_documents(docs)
        print(f"✂️ [分块] 文档已被切分为 {len(splits)} 个小块。")

        # 3. 存入 Chroma 向量数据库
        print("💾 [数据库] 正在生成向量并存入本地数据库...")
        self.vector_store = Chroma.from_documents(
            documents=splits, 
            embedding=self.embeddings, 
            persist_directory=self.persist_directory
        )
        print("✅ [成功] 知识库构建完成！\n")

    def ask_question(self, question):
        """根据文档回答问题"""
        if not self.vector_store:
            self.vector_store = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)

        # 1. 检索最相关的3个文本块
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        # 2. 构建 Prompt 模板 (简历亮点：构建RAG增强生成链路)
        system_prompt = (
            "你是一个专业的企业知识库助手。请根据以下检索到的背景上下文中回答问题。\n"
            "如果你不知道答案，就说不知道，不要编造。\n\n"
            "上下文: {context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # 3. 组装 RAG 链
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # 4. 执行提问
        print(f"🤔 [AI思考中] 正在分析问题: '{question}' ...")
        response = rag_chain.invoke({"input": question})
        return response["answer"]

# ================= 测试代码 =================
if __name__ == "__main__":
    rag = RAGSystem()
    pdf_path = "test.pdf" 
    
    if os.path.exists(pdf_path):
        rag.ingest_document(pdf_path)
    else:
        print(f"❌ 找不到文件！请把PDF文件改名为 {pdf_path} 并放在本目录下。")
        sys.exit()

    # ====== 提问环节 ======
    # 你可以在这里修改你想问的问题，最好是跟你的 PDF 相关的
    question = "请用一段话总结这份文档的核心内容。"
    answer = rag.ask_question(question)
    
    print("\n" + "="*40)
    print("🌟 AI 回答:")
    print(answer)
    print("="*40 + "\n")
