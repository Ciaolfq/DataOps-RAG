import os
from typing import List, Dict
from langchain_classic.chains import RetrievalQA  # 从 classic 导入
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
import json

# 配置DeepSeek API
os.environ["OPENAI_API_KEY"] = ""  # 替换为您的API密钥
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"

class DataOpsQASystem:
    def __init__(self, vector_db_path: str):
        """初始化问答系统"""
        print("正在初始化问答系统...")
        
        # 加载embedding模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name="shibing624/text2vec-base-chinese",
            model_kwargs={'device': 'cpu'}
        )
        
        # 加载向量数据库
        self.vectorstore = Chroma(
            persist_directory=vector_db_path,
            embedding_function=self.embeddings
        )
        
        # 配置DeepSeek LLM
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0.3,
            max_tokens=2000
        )
        
        # 创建自定义提示模板
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
        
        # 创建检索QA链
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 5}
            ),
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True
        )
        
        print("问答系统初始化完成！")
    
    def ask(self, question: str) -> Dict:
        """提问并获取回答"""
        print(f"正在思考: {question}")
        
        # 获取回答
        result = self.qa_chain.invoke({"query": question})
        
        return {
            "question": question,
            "answer": result["result"],
            "sources": [
                {
                    "title": doc.metadata.get("title", ""),
                    "type": doc.metadata.get("type", ""),
                    "category": f"{doc.metadata.get('level1', '')} - {doc.metadata.get('level2', '')}",
                    "solution": doc.metadata.get("solution", ""),
                    "content": doc.page_content[:200] + "..."
                }
                for doc in result["source_documents"]
            ]
        }

def main():
    # 检查向量数据库是否存在
    if not os.path.exists("../chroma_db"):
        print("错误: 向量数据库不存在，请先运行 02_create_vector_db.py")
        return
    
    # 初始化问答系统
    qa_system = DataOpsQASystem("../chroma_db")
    
    print("\n" + "="*60)
    print("DataOps 问答系统已启动！")
    print("输入 'exit' 退出，输入 'clear' 清屏")
    print("="*60)
    
    while True:
        question = input("\n👤 请输入您的问题: ")
        if question.lower() == 'exit':
            break
        if question.lower() == 'clear':
            os.system('cls' if os.name == 'nt' else 'clear')
            continue
        
        result = qa_system.ask(question)
        
        print("\n🤖 回答:")
        print(result["answer"])
        
        print("\n📚 参考来源:")
        for i, source in enumerate(result["sources"][:3], 1):
            print(f"  {i}. [{source['type']}] {source['title']}")
            if source['category'] and source['category'] != ' - ':
                print(f"     类别: {source['category']}")
            if source['solution']:
                print(f"     解决方案: {source['solution']}")
        print("-"*60)

if __name__ == "__main__":
    main()