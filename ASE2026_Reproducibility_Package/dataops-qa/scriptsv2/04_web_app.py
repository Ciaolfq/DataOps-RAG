import gradio as gr
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
import os
from typing import Dict

# 配置DeepSeek API
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"

class DataOpsQASystem:
    def __init__(self, vector_db_path: str):
        """初始化问答系统"""
        print("Initializing QA system...")
        
        # 使用英文embedding模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # 英文模型
            model_kwargs={'device': 'cpu'}
        )
        
        self.vectorstore = Chroma(
            persist_directory=vector_db_path,
            embedding_function=self.embeddings
        )
        
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0.3,
            max_tokens=2000
        )
        
        # 英文提示模板，要求英文回答
        prompt_template = """
You are a DataOps expert. Answer the question based on the context below.

Context:
{context}

Question: {question}

Requirements:
1. If relevant cases exist in the context, cite specific examples
2. If the question involves solutions, indicate which solution type (S1-S10)
3. If applicable, state which DataOps stage the problem belongs to (Setup, Preprocess, Develop, Execute, Integrate, Deploy)
4. Be concise, accurate, and helpful

Answer in English.
"""
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True
        )
        
        print("QA system initialized successfully!")
    
    def ask(self, question: str) -> Dict:
        print(f"Processing: {question}")
        result = self.qa_chain.invoke({"query": question})
        return {
            "question": question,
            "answer": result["result"],
            "sources": [
                {
                    "title": doc.metadata.get("title", ""),
                    "type": doc.metadata.get("type", ""),
                    "category": f"{doc.metadata.get('level1', '')} - {doc.metadata.get('level2', '')}"
                }
                for doc in result["source_documents"][:3]
            ]
        }

# 初始化
qa_system = DataOpsQASystem("../chroma_db")

# 定义聊天函数
def chat(message, history):
    """聊天函数，返回字符串即可"""
    if not message:
        return ""
    
    try:
        result = qa_system.ask(message)
        response = result["answer"]
        
        if result["sources"]:
            response += "\n\n---\n\n**Sources:**\n"
            for i, s in enumerate(result["sources"][:3], 1):
                response += f"{i}. [{s['type']}] {s['title']}\n"
                if s['category'] and s['category'] != ' - ':
                    response += f"   Category: {s['category']}\n"
        
        return response
    except Exception as e:
        return f"Error: {str(e)}"

# 示例问题（英文）
examples = [
    "What are the main stages of DataOps?",
    "How to resolve Delta Lake and Spark version incompatibility?",
    "What should I do when encountering API call errors?",
    "What are the best practices for data pipeline monitoring?",
    "How to handle schema evolution in data lakes?"
]

# 使用 ChatInterface
demo = gr.ChatInterface(
    fn=chat,
    title="🤖 DataOps QA System",
    examples=examples
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)