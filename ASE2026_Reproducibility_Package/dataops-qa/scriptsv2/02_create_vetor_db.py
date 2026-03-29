import json
from langchain_text_splitters import RecursiveCharacterTextSplitter  # ✅ 正确
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

# 加载文档
try:
    with open('../documents.json', 'r', encoding='utf-8') as f:
        documents = json.load(f)
    print(f"加载了 {len(documents)} 个文档")
except FileNotFoundError:
    print("错误: documents.json 不存在，请先运行 01_load_data.py")
    exit(1)

# 初始化embedding模型（使用英文模型）
print("正在加载embedding模型...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # 英文模型
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ".", ",", " ", ""]  # 英文分隔符
)

# 准备文本和元数据
texts = []
metadatas = []

for doc in documents:
    # 分割长文本
    chunks = text_splitter.split_text(doc["content"])
    
    for i, chunk in enumerate(chunks):
        texts.append(chunk)
        # 为每个chunk保存元数据
        metadata = {
            "id": doc["id"],
            "title": doc.get("title", ""),
            "type": doc.get("type", ""),
            "level1": doc.get("level1", ""),
            "level2": doc.get("level2", ""),
            "solution": doc.get("solution", ""),
            "chunk_index": i,
            "total_chunks": len(chunks)
        }
        metadatas.append(metadata)

print(f"生成了 {len(texts)} 个文本块")

# 创建向量数据库
print("正在创建向量数据库...")
vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas,
    persist_directory="../chroma_db"
)

# 对于最新版本，persist() 可能不需要，但保留以防万一
try:
    vectorstore.persist()
    print("向量数据库已持久化")
except:
    print("向量数据库已创建（自动持久化）")

print(f"向量数据库已保存到 ../chroma_db")
print(f"包含 {vectorstore._collection.count()} 个向量")