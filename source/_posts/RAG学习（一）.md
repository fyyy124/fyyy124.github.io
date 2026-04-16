参考[https://github.com/datawhalechina/all-in-rag](https://github.com/datawhalechina/all-in-rag)

## 一、RAG简介
###### 1.定义
从本质上讲，RAG（Retrieval-Augmented Generation）是一种旨在解决大语言模型（LLM）“知其然不知其所以然”问题的技术范式。它的核心是将模型内部学到的“参数化知识”（模型权重中固化的、模糊的“记忆”），与来自外部知识库的“非参数化知识”（精准、可随时更新的外部数据）相结合。其运作逻辑就是在 LLM 生成文本前，先通过检索机制从外部知识库中动态获取相关信息，并将这些“参考资料”融入生成过程，从而提升输出的准确性和时效性 [1](https://github.com/fyyy124/all-in-rag/blob/main/docs/chapter1/01_RAG_intro.md#user-content-fn-1-c7ccf47d2427065c7d1812d677a1f549) [2](https://github.com/fyyy124/all-in-rag/blob/main/docs/chapter1/01_RAG_intro.md#user-content-fn-2-c7ccf47d2427065c7d1812d677a1f549) [3](https://github.com/fyyy124/all-in-rag/blob/main/docs/chapter1/01_RAG_intro.md#user-content-fn-3-c7ccf47d2427065c7d1812d677a1f549)。

💡 一句话总结：RAG 就是让 LLM 学会了“开卷考试”，它既能利用自己学到的知识，也能随时查阅外部资料。

###### 2.技术原理
“参数化知识”和“非参数化知识”的结合

###### （1）检索阶段：寻找“非参数化知识”
+ 知识向量化：嵌入模型（Embedding Model） 充当了“连接器”的角色。它将外部知识库编码为向量索引（Index），存入向量数据库。
+ 语义召回：当用户发起查询时，检索模块利用同样的嵌入模型将问题向量化，并通过相似度搜索（Similarity Search），从海量数据中精准锁定与问题最相关的文档片段。

###### （2）生成阶段：融合两种知识
+ 上下文整合：生成模块接收检索阶段送来的相关文档片段以及用户的原始问题。
+ 指令引导生成：该模块会遵循预设的 Prompt 指令，将上下文与问题有效整合，并引导 LLM（如 DeepSeek）进行可控的、有理有据的文本生成。

### 


## 二、RAG构建基本流程
### 四步构建最小可行系统（MVP）
以文档提供的代码示例为例

###### （1）数据准备与清洗：
这是系统的地基。我们需要将 PDF、Word 等多源异构数据标准化，并采用合理的分块策略（如按语义段落切分而非固定字符数），避免信息在切割中支离破碎。

加载原始文档，然后对文本分块。

```python
# 加载本地markdown文件
loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()

# 文本分块 长文档被分割成较小的文本块，便于后续的嵌入和搜索
text_splitter = RecursiveCharacterTextSplitter()
chunks = text_splitter.split_documents(docs)
```

###### （2）索引构建：
将切分好的文本通过嵌入模型转化为向量，并存入数据库。可以在此阶段关联元数据（如来源、页码），这对后续的精确引用很有帮助。

```python
# 中文嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
  
# 构建向量存储 将分割后的文本块 (texts) 通过初始化好的嵌入模型转换为向量表示
vectorstore.add_documents(chunks)
```

这个过程结束后构建了一个可供查询的知识索引。

###### （3）检索策略优化：
不要依赖单一的向量搜索。可以采用混合检索（向量+关键词）等方式来提升召回率，并引入重排序模型对检索结果进行二次精选，确保 LLM 看到的都是精华。

在此代码中，直接这样查询

```python
# 用户查询
question = "文中举了哪些例子？"

# 在向量存储中查询相关文档 k=3即查找最相关的3个文档
retrieved_docs = vectorstore.similarity_search(question, k=3)

#准备上下文: 将检索到的多个文本块的页面内容合并成一个单一的字符串，
#形成最终的上下文信息 (docs_content) 供大语言模型参考。
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
```

###### （4）生成与提示工程：
最后，设计一套清晰的 Prompt 模板，引导 LLM 基于检索到的上下文回答用户问题，并明确要求模型“不知道就说不知道”，防止幻觉。

最后一步是将检索到的上下文与用户问题结合，利用大语言模型（LLM）生成答案：

构建提示词模板: 使用`ChatPromptTemplate.from_template`创建一个结构化的提示模板。此模板指导LLM根据提供的上下文 (`context`) 回答用户的问题 (`question`)，并明确指出在信息不足时应如何回应。

```python
# 提示词模板
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:"""
                                          )
```



## 三、练习
###### 1、LangChain代码最终得到的输出携带了各种参数，查询相关资料尝试把这些参数过滤掉得到`content`里的具体回答。
```python
response = llm.invoke(prompt.format(question=question, context=docs_content))
#只提取文字部分
answer = response.content
print(answer)
```

###### 2、修改Langchain代码中`RecursiveCharacterTextSplitter()`的参数`chunk_size`和`chunk_overlap`，观察输出结果有什么变化。
| **<font style="color:rgb(31, 31, 31);">实验组别</font>** | **<font style="color:rgb(31, 31, 31);">Chunk Size</font>** | **<font style="color:rgb(31, 31, 31);">Chunk Overlap</font>** | **<font style="color:rgb(31, 31, 31);">检索与输出表现 (现象)</font>** |
| --- | --- | --- | --- |
| <font style="color:rgb(31, 31, 31);">实验 A</font> | **<font style="color:rgb(31, 31, 31);">2000</font>** | <font style="color:rgb(31, 31, 31);">100 (5%)</font> | **<font style="color:rgb(31, 31, 31);">扁平流水账，广度大：</font>**<br/><font style="color:rgb(31, 31, 31);">按顺序罗列了8个毫无分类的例子。跨度极大，包含了从文章开头到结尾的例子，但丢失了很多细小案例。</font> |
| <font style="color:rgb(31, 31, 31);">实验 B</font><br/><font style="color:rgb(31, 31, 31);">   </font><br/><font style="color:rgb(31, 31, 31);"></font> | **<font style="color:rgb(31, 31, 31);">1000</font>** | <font style="color:rgb(31, 31, 31);">100 (10%)</font> | **<font style="color:rgb(31, 31, 31);">结构化分类，抠细节：</font>**<br/><font style="color:rgb(31, 31, 31);">大模型主动将例子归纳为3大类；挖掘出了基准组漏掉的“小车代码”等隐藏细节，但丢失了文章结尾的例子。</font> |
| <font style="color:rgb(31, 31, 31);">实验 C</font><br/><font style="color:rgb(31, 31, 31);">   </font><br/><font style="color:rgb(31, 31, 31);"></font> | <font style="color:rgb(31, 31, 31);">1000</font> | **<font style="color:rgb(31, 31, 31);">200</font>**<font style="color:rgb(31, 31, 31);"> (20%)</font> | **<font style="color:rgb(31, 31, 31);">结果无变化：</font>**<br/><font style="color:rgb(31, 31, 31);">输出的大类和具体例子，与实验 B几乎</font>完全一模一样。 |
| <font style="color:rgb(31, 31, 31);">实验 D</font><br/><font style="color:rgb(31, 31, 31);">   </font><br/><font style="color:rgb(31, 31, 31);"></font> | <font style="color:rgb(31, 31, 31);">1000</font> | **<font style="color:rgb(31, 31, 31);">50</font>**<font style="color:rgb(31, 31, 31);"> (5%)</font> | **<font style="color:rgb(31, 31, 31);">内容大洗牌：</font>**<br/><font style="color:rgb(31, 31, 31);">“小车代码”例子离奇消失，反而凭空找出了极其完整的“探索与利用（挖油/餐馆）”新大类。</font> |


**总结**：在 RAG 系统的文本切分中，Chunk Size是决定大模型“视野”的核心主导因素，调大它能保留完整的宏观逻辑但容易让细节在长文本中被稀释漏检；调小它能精准命中细节并促使大模型做出优秀的结构化归纳，但跨度有限。相比之下，Chunk Overlap仅仅防止断句，将其设定在块大小的 10%~15% 即可完美运转。



######  3.LlamaIndex  代码加注释
根据RAG四个步骤分别注释，该框架便利性大大提升，代码量明显减少。

```python
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
# 引入底层切分器，用于演示步骤1
from llama_index.core.node_parser import SentenceSplitter 
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

# ==========================================
# 前置配置：全局模型绑定
# ==========================================
# 配置 LLM（大语言模型），用于步骤4的生成
Settings.llm = OpenAILike(
    model="glm-4.7-flash-free",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_base="https://aihubmix.com/v1",
    is_chat_model=True
)

# 配置嵌入模型，用于步骤2的向量化
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")


# ==========================================
# （1）数据准备与清洗：标准化与合理分块
# ==========================================
# 加载原始文档：SimpleDirectoryReader 会自动读取并提取文本
docs = SimpleDirectoryReader(input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]).load_data()


# ==========================================
# （2）索引构建：转为向量并存入数据库
# ==========================================
# 这一行代码是黑盒，它内部自动完成了两件事：
# 1. 拿着上面切好的文本块（Nodes），调用 BGE 模型计算向量。
# 2. 将文本块、元数据（如文档来源、页码信息）和向量坐标，一同存入默认的内存数据库。
index = VectorStoreIndex.from_documents(docs)


# ==========================================
# （3）检索策略优化：提升召回率与二次精选
# ==========================================
# as_query_engine() 初始化了检索器。
query_engine = index.as_query_engine()


# ==========================================
# （4）生成与提示工程：引导回答并防止幻觉
# ==========================================
# LlamaIndex 默认已经内置了一套极其完善的 Prompt 模板。
# 它的底层模板中已经包含了类似“请根据提供的上下文回答，如果不知道请说不知道”的防幻觉指令。
print(query_engine.get_prompts())

print(query_engine.query("文中举了哪些例子?"))
```
