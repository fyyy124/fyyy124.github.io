---
title: RAG学习（三）
date: 2026-04-22 20:03:19
tags:
---
参考整理源自[https://github.com/datawhalechina/all-in-rag](https://github.com/datawhalechina/all-in-rag)

## 一、向量嵌入
向量嵌入（Embedding）是一种将真实世界中复杂、高维的数据对象（如文本、图像、音频、视频等）转换为数学上易于处理的、低维、稠密的连续数值向量的技术。

Embedding 的真正意义在于，它产生的向量不是随机数值的堆砌，而是对数据语义的数学编码。

1. 核心原则：在 Embedding 构建的向量空间中，语义上相似的对象，其对应的向量在空间中的距离会更近；而语义上不相关的对象，它们的向量距离会更远。
2. 关键度量：我们通常使用以下数学方法来衡量向量间的“距离”或“相似度”：
    1. 余弦相似度 (Cosine Similarity) ：计算两个向量夹角的余弦值。值越接近 1，代表方向越一致，语义越相似。这是最常用的度量方式。
    2. 点积 (Dot Product) ：计算两个向量的乘积和。在向量归一化后，点积等价于余弦相似度。
    3. 欧氏距离 (Euclidean Distance) ：计算两个向量在空间中的直线距离。距离越小，语义越相似。

### 
## 二、多模态嵌入
目的是将不同类型的数据（如图像和文本）映射到同一个共享的向量空间。

```python
import torch
from visual_bge.visual_bge.modeling import Visualized_BGE

model = Visualized_BGE(model_name_bge="BAAI/bge-base-en-v1.5",
                      model_weight="../../models/bge/Visualized_base_en_v1.5.pth")
model.eval()

with torch.no_grad():
    text_emb = model.encode(text="datawhale开源组织的logo")
    img_emb_1 = model.encode(image="../../data/C3/imgs/datawhale01.png")
    multi_emb_1 = model.encode(image="../../data/C3/imgs/datawhale01.png", text="datawhale开源组织的logo")
    img_emb_2 = model.encode(image="../../data/C3/imgs/datawhale02.png")
    multi_emb_2 = model.encode(image="../../data/C3/imgs/datawhale02.png", text="datawhale开源组织的logo")

# 计算相似度   实际是测算距离
sim_1 = img_emb_1 @ img_emb_2.T
sim_2 = img_emb_1 @ multi_emb_1.T
sim_3 = text_emb @ multi_emb_1.T
sim_4 = multi_emb_1 @ multi_emb_2.T

print("=== 相似度计算结果 ===")
print(f"纯图像 vs 纯图像: {sim_1}")
print(f"图文结合1 vs 纯图像: {sim_2}")
print(f"图文结合1 vs 纯文本: {sim_3}")
print(f"图文结合1 vs 图文结合2: {sim_4}")

# 向量信息分析
print("\n=== 嵌入向量信息 ===")
print(f"多模态向量维度: {multi_emb_1.shape}")
print(f"图像向量维度: {img_emb_1.shape}")
print(f"多模态向量示例 (前10个元素): {multi_emb_1[0][:10]}")
print(f"图像向量示例 (前10个元素):   {img_emb_1[0][:10]}")
```

`Visualized_BGE` 是通过将图像token嵌入集成到BGE文本嵌入框架中构建的通用多模态嵌入模型，具备处理超越纯文本的多模态数据的灵活性。

+ `**model_name_bge**`：指定模型的基础文本底座。
+ `**model_weight**`：这是加载视觉多模态部分的特定权重文件。

多模态编码能力: Visual BGE提供了编码多模态数据的多样性，支持纯文本、纯图像或图文组合的格式：

+ 纯文本编码: 保持原始BGE模型的强大文本嵌入能力。
+ 纯图像编码: 使用基于EVA-CLIP的视觉编码器处理图像。
+ 图文联合编码: 将图像和文本特征融合到统一的向量空间。

### 练习：
尝试将代码中的部分文本替换，

1. datawhale的logo替换成blue whale     结果：图文结合1与纯文本的相似度降为0.5510
2. 替换成blue                                           结果：相似度进一步降低至0.4302

## 三、向量数据库
向量数据库通常采用四层架构，通过存储层、索引层、查询层和服务层的协同工作来实现高效相似性搜索，其中存储层负责存储向量数据和元数据，优化存储效率并支持分布式存储；索引层维护索引算法（HNSW、LSH、PQ等），负责索引的创建与优化，并支持索引调整；查询层处理查询请求，支持混合查询并实现查询优化；服务层管理客户端连接，提供监控和日志能力，并实现安全管理。

主要技术手段包括：

+ 基于树的方法：如 Annoy 使用的随机投影树，通过树形结构实现对数复杂度的搜索
+ 基于哈希的方法：如 LSH（局部敏感哈希），通过哈希函数将相似向量映射到同一“桶”
+ 基于图的方法：如 HNSW（分层可导航小世界图），通过多层邻近图结构实现快速搜索
+ 基于量化的方法：如 Faiss 的 IVF 和 PQ，通过聚类和量化压缩向量

下面的代码演示了使用 LangChain 和 FAISS 完成一个完整的“创建 -> 保存 -> 加载 -> 查询”流程。

```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 1. 示例文本和嵌入模型
texts = [
    "张三是法外狂徒",
    "FAISS是一个用于高效相似性搜索和密集向量聚类的库。",
    "LangChain是一个用于开发由语言模型驱动的应用程序的框架。"
]
docs = [Document(page_content=t) for t in texts]
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 2. 创建向量存储并保存到本地
vectorstore = FAISS.from_documents(docs, embeddings)

local_faiss_path = "./faiss_index_store"
vectorstore.save_local(local_faiss_path)

print(f"FAISS index has been saved to {local_faiss_path}")

# 3. 加载索引并执行查询
# 加载时需指定相同的嵌入模型，并允许反序列化
loaded_vectorstore = FAISS.load_local(
    local_faiss_path,
    embeddings,
    allow_dangerous_deserialization=True
)

# 执行相似性搜索
query = "FAISS是做什么的？"
results = loaded_vectorstore.similarity_search(query, k=1)

print(f"\n查询: '{query}'")
print("相似度最高的文档:")
for doc in results:
    print(f"- {doc.page_content}")
```

### 练习：
新建一个代码文件实现对LlamaIndex存储数据的加载和相似性搜索。

具体步骤：1、指定嵌入模型  2、加载数据，恢复索引 3、加载索引，相似性搜索，执行查询

```python
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. 配置全局嵌入模型 
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 2. 指定之前保存数据的文件夹路径
persist_path = "./llamaindex_index_store"

# 3. 加载存储上下文 并从中恢复索引
# 注意：这里需要确保上面从 llama_index.core 导入了 StorageContext 和 load_index_from_storage
storage_context = StorageContext.from_defaults(persist_dir=persist_path)
loaded_index = load_index_from_storage(storage_context)

# 4. 加载索引并执行查询 (LlamaIndex 的专属写法)
# 把加载好的索引变成一个“检索器”，similarity_top_k=1 相当于原来的 k=1
retriever = loaded_index.as_retriever(similarity_top_k=1)

# 执行相似性搜索
query = "LlamaIndex是做什么用的？" 
results = retriever.retrieve(query)

print(f"\n查询: '{query}'")
print(" 相似度最高的文档:")

# LlamaIndex 返回的是 node 对象，文本存在 node.text 里，而不是 page_content
for node in results:
    print(f"- 内容: {node.text}")
    print(f"  相似度得分: {node.score:.4f}")
```

## 四、Milvus介绍及多模态检索实践
Milvus检索的核心功能：近似最近邻（ANN）检索

具体过程见all-in-rag代码文档

## 五、索引优化
### 1、上下文扩展
LlamaIndex 提出了一种实用的索引策略——句子窗口检索。该技术巧妙地结合了两种方法的优点：它在检索时聚焦于高度精确的单个句子，在送入LLM生成答案前，又智能地将上下文扩展回一个更宽的“窗口”，从而同时保证检索的准确性和生成的质量。

其工作流程如下：

    1. 索引阶段：在构建索引时，文档被分割成单个句子。每个句子都作为一个独立的“节点（Node）”存入向量数据库。同时，每个句子节点都会在元数据（metadata）中存储其上下文窗口，即该句子原文中的前N个和后N个句子。这个窗口内的文本不会被索引，仅仅是作为元数据存储。
    2. 检索阶段：当用户发起查询时，系统会在所有单一句子节点上执行相似度搜索。因为句子是表达完整语义的最小单位，所以这种方式可以非常精确地定位到与用户问题最相关的核心信息。
    3. 后处理阶段：在检索到最相关的句子节点后，系统会使用一个名为 `MetadataReplacementPostProcessor` 的后处理模块。该模块会读取到检索到的句子节点的元数据，并用元数据中存储的完整上下文窗口来替换节点中原来的单一句子内容。
    4. 生成阶段：最后，这些被替换了内容的、包含丰富上下文的节点被传递给LLM，用于生成最终的答案。

```python
import os
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

# 1. 配置模型
Settings.llm = DeepSeek(model="deepseek-chat", temperature=0.1, api_key=os.getenv("DEEPSEEK_API_KEY"))
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

# 2. 加载文档
documents = SimpleDirectoryReader(
    input_files=["../../data/C3/pdf/IPCC_AR6_WGII_Chapter03.pdf"]
).load_data()

# 3. 创建节点与构建索引
# 3.1 句子窗口索引
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)
sentence_nodes = node_parser.get_nodes_from_documents(documents)
sentence_index = VectorStoreIndex(sentence_nodes)

# 3.2 常规分块索引 (基准)
base_parser = SentenceSplitter(chunk_size=512)
base_nodes = base_parser.get_nodes_from_documents(documents)
base_index = VectorStoreIndex(base_nodes)

# 4. 构建查询引擎
sentence_query_engine = sentence_index.as_query_engine(
    similarity_top_k=2,
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)
base_query_engine = base_index.as_query_engine(similarity_top_k=2)

# 5. 执行查询并对比结果
query = "What are the concerns surrounding the AMOC?"
print(f"查询: {query}\n")

print("--- 句子窗口检索结果 ---")
window_response = sentence_query_engine.query(query)
print(f"回答: {window_response}\n")

print("--- 常规检索结果 ---")
base_response = base_query_engine.query(query)
print(f"回答: {base_response}\n")
```

（1）构建句子窗口索引：这一步利用了 `SentenceWindowNodeParser`。它将文档解析为以单个句子为单位的 `Node`，同时将包含上下文的“窗口”文本（默认为前后各3个句子）存储在每个 `Node` 的元数据中。这一步是实现“为检索精确性而索引小块”思想的关键。

（2）构建查询引擎与后处理：查询引擎的构建是实现“为生成质量而扩展上下文”的关键。

+ 在创建 `sentence_query_engine` 时，配置中加入了一个重要的后处理器 `MetadataReplacementPostProcessor`。
+ 它的作用是：当检索器根据用户查询找到最相关的节点（也就是单个句子）后，这个后处理器会立即介入。
+ 它会从该节点的元数据中读取出预先存储的完整“窗口”文本，并用它替换掉节点中原来的单个句子内容。
+ 这样，最终传递给大语言模型的就不再是孤立的句子，而是包含丰富上下文的完整文本段落，从而确保了生成答案的质量和连贯性。



### 2、结构化索引
原理是在索引文本块的同时，为其附加结构化的元数据（Metadata）。这些元数据可以是任何有助于筛选和定位信息的标签。

基于多表格的递归检索

```python
import os
import pandas as pd
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

# 配置模型
Settings.llm = DeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"))
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

# 1. 加载和预处理数据
excel_file = '../../data/C3/excel/movie.xlsx'
xls = pd.ExcelFile(excel_file)

summary_docs = []
content_docs = []

print("开始加载和处理Excel文件...")
for sheet_name in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet_name)
    
    # 数据清洗
    if '评分人数' in df.columns:
        df['评分人数'] = df['评分人数'].astype(str).str.replace('人评价', '').str.strip()
        df['评分人数'] = pd.to_numeric(df['评分人数'], errors='coerce').fillna(0).astype(int)

    # 创建摘要文档 (用于路由)
    year = sheet_name.replace('年份_', '')
    summary_text = f"这个表格包含了年份为 {year} 的电影信息，包括电影名称、导演、评分、评分人数等。"
    summary_doc = Document(
        text=summary_text,
        metadata={"sheet_name": sheet_name}
    )
    summary_docs.append(summary_doc)
    
    # 创建内容文档 (用于最终问答)
    content_text = df.to_string(index=False)
    content_doc = Document(
        text=content_text,
        metadata={"sheet_name": sheet_name}
    )
    content_docs.append(content_doc)

print("数据加载和处理完成。\n")

# 2. 构建向量索引
# 使用默认的内存SimpleVectorStore，它支持元数据过滤

# 2.1 为摘要创建索引
summary_index = VectorStoreIndex(summary_docs)

# 2.2 为内容创建索引
content_index = VectorStoreIndex(content_docs)

print("摘要索引和内容索引构建完成。\n")

# 3. 定义两步式查询逻辑
def query_safe_recursive(query_str):
    print(f"--- 开始执行查询 ---")
    print(f"查询: {query_str}")
    
    # 第一步：路由 - 在摘要索引中找到最相关的表格
    print("\n第一步：在摘要索引中进行路由...")
    summary_retriever = VectorIndexRetriever(index=summary_index, similarity_top_k=1)
    retrieved_nodes = summary_retriever.retrieve(query_str)
    
    if not retrieved_nodes:
        return "抱歉，未能找到相关的电影年份信息。"
    
    # 获取匹配到的工作表名称
    matched_sheet_name = retrieved_nodes[0].node.metadata['sheet_name']
    print(f"路由结果：匹配到工作表 -> {matched_sheet_name}")
    
    # 第二步：检索 - 在内容索引中根据工作表名称过滤并检索具体内容
    print("\n第二步：在内容索引中检索具体信息...")
    content_retriever = VectorIndexRetriever(
        index=content_index,
        similarity_top_k=1, # 通常只返回最匹配的整个表格即可
        filters=MetadataFilters(
            filters=[ExactMatchFilter(key="sheet_name", value=matched_sheet_name)]
        )
    )
    
    # 创建查询引擎并执行查询
    query_engine = RetrieverQueryEngine.from_args(content_retriever)
    response = query_engine.query(query_str)
    
    print("--- 查询执行结束 ---\n")
    return response

# 4. 执行查询
query = "1994年评分人数最少的电影是哪一部？"
response = query_safe_recursive(query)

print(f"最终回答: {response}")
```

将路由和检索彻底分离。具体步骤如下：

（1）创建两个独立的向量索引：

+ 摘要索引（用于路由）：为每个Excel工作表（例如，“1994年电影数据”）创建一个非常简短的摘要性`Document`，例如：“此文档包含1994年的电影信息”。然后，用所有这些摘要文档构建一个轻量级的向量索引。这个索引的唯一目的就是充当“路由器”。
+ 内容索引（用于问答）：将每个工作表的实际数据（例如，整个表格）转换为一个大的文本`Document`，并为其附加一个关键的元数据标签，如 `{"sheet_name": "年份_1994"}`。然后，用所有这些包含真实内容的文档构建一个向量索引。`text` 属性用于精确检索，而其 `metadata` 中则“隐藏”了用于生成答案的丰富上下文窗口。

（2）执行两步查询：

+ 第一步：路由。当用户提问（例如，“1994年评分人数最少的电影是哪一部？”）时，首先在“摘要索引”中进行检索。由于问题中的“1994年”与“此文档包含1994年的电影信息”这个摘要高度相关，检索器会快速返回其对应的元数据，告诉系统目标是 `年份_1994` 这个工作表。
+ 第二步：检索。拿到 `年份_1994` 这个目标后，系统会在“内容索引”中进行检索，但这次会附加一个元数据过滤器（`MetadataFilter`），强制要求只在 `sheet_name == "年份_1994"` 的文档中进行搜索。这样，LLM就能在正确的、经过筛选的数据范围内找到问题的答案。
