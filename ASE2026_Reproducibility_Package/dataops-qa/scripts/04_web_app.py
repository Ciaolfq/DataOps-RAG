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
        print("正在初始化问答系统...")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="shibing624/text2vec-base-chinese",
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
        
        prompt_template = """
You are a DataOps expert. Answer the question based on the context below.

Context:
{context}

Question: {question}

Answer in Chinese.
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
        
        print("问答系统初始化完成！")
    
    def ask(self, question: str) -> Dict:
        print(f"正在思考: {question}")
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

# 示例问题
examples = [
    "What are the main stages of DataOps?",
    "How to resolve Delta Lake and Spark version incompatibility?",
    "What should I do when encountering API call errors?"
]

# 使用 ChatInterface，去掉不支持的参数
demo = gr.ChatInterface(
    fn=chat,
    title="🤖 DataOps QA System",
    examples=examples
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)