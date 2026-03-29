import json
import os
from typing import List, Dict
import pandas as pd

def load_json_data(file_path: str) -> List[Dict]:
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"已加载 {len(data)} 条记录 from {file_path}")
    return data

def load_finding(file_path: str) -> str:
    """加载finding文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"已加载finding文件，长度: {len(content)} 字符")
        return content
    except FileNotFoundError:
        print(f"警告: {file_path} 不存在，使用空字符串")
        return ""

def prepare_documents(so_data: List[Dict], gh_data: List[Dict], finding: str) -> List[Dict]:
    """准备文档列表用于向量化"""
    documents = []
    
    # 1. 添加finding作为全局知识
    if finding:
        documents.append({
            "id": "finding_001",
            "title": "DataOps Research Findings",
            "content": finding,
            "type": "finding",
            "source": "research"
        })
    
    # 2. 处理SO数据
    for idx, item in enumerate(so_data[:500]):  # 限制数量
        # 构建文档内容（英文格式）
        content = f"Title: {item.get('title', '')}\n"
        content += f"Question: {item.get('body', '')[:1000]}\n"
        
        # 如果有采纳答案，加入
        if 'accepted_answer' in item and item['accepted_answer']:
            answer_body = item['accepted_answer'].get('answer_body', '')
            if answer_body:
                content += f"Answer: {answer_body[:1000]}\n"
        
        # 如果有分类信息
        if 'level1' in item:
            content += f"Category: {item.get('level1', '')} - {item.get('level2', '')}\n"
        
        # 如果有solution信息
        if 'solution' in item:
            content += f"Solution: {item.get('solution', '')}\n"
        
        documents.append({
            "id": f"so_{idx}_{item.get('id', '')}",
            "title": item.get('title', ''),
            "content": content,
            "type": "stackoverflow",
            "level1": item.get('level1', ''),
            "level2": item.get('level2', ''),
            "solution": item.get('solution', '')
        })
    
    # 3. 处理GH数据
    for idx, item in enumerate(gh_data[:200]):  # 限制数量
        content = f"Issue Title: {item.get('title', '')}\n"
        content += f"Description: {item.get('body', '')[:1000]}\n"
        
        # 如果有评论，加入
        if 'comments' in item and item['comments']:
            comments = "\n".join([f"- {c.get('body', '')[:200]}" for c in item['comments'][:3]])
            content += f"Comments:\n{comments}\n"
        
        if 'level1' in item:
            content += f"Category: {item.get('level1', '')} - {item.get('level2', '')}\n"
        
        documents.append({
            "id": f"gh_{idx}_{item.get('id', '')}",
            "title": item.get('title', ''),
            "content": content,
            "type": "github",
            "level1": item.get('level1', ''),
            "level2": item.get('level2', '')
        })
    
    print(f"总共准备了 {len(documents)} 个文档")
    return documents

if __name__ == "__main__":
    # 加载数据
    so_data = load_json_data('../data/so.json')
    gh_data = load_json_data('../data/gh.json')
    finding = load_finding('../data/finding.txt')
    
    # 准备文档
    documents = prepare_documents(so_data, gh_data, finding)
    
    # 保存文档列表供下一步使用
    with open('../documents.json', 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    
    print("数据准备完成，已保存到 documents.json")