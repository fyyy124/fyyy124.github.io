---
title: RAG学习（二）
date: 2026-04-22 18:46:30
tags:
---

参考整理源自[https://github.com/datawhalechina/all-in-rag](https://github.com/datawhalechina/all-in-rag)

# 数据加载
## 一、文档加载器
### 主要功能
R文档加载器负责将各种格式的非结构化文档（如PDF、Word、Markdown、HTML等）转换为程序可以处理的结构化数据。数据加载的质量会直接影响后续的索引构建、检索效果和最终的生成质量。

文档加载器在 RAG 的数据管道中一般需要完成三个核心任务，一是解析不同格式的原始文档，将 PDF、Word、Markdown 等内容提取为可处理的纯文本，二是在解析过程中同时抽取文档来源、页码、作者等关键信息作为元数据，三是把文本和元数据整理成统一的数据结构，方便后续进行切分、向量化和入库，其整体流程与传统数据工程中的抽取、转换、加载相似，目标都是把杂乱的原始文档清洗并对齐为适合检索和建模的标准化语料。

## 二、Unstructured文档处理库
Unstructured [1](https://github.com/fyyy124/all-in-rag/blob/main/docs/chapter2/04_data_load.md#user-content-fn-1-664e331fbca981296b7fba835cbe0786)是一个专业的文档处理库，专门设计用于RAG和AI微调场景的非结构化数据预处理。提供了统一的接口来处理多种文档格式，是目前应用较广泛的文档加载解决方案之一。Unstructured 在格式支持和内容解析方面具有明显优势，它一方面支持 PDF、Word、Excel、HTML、Markdown 等多种文档格式，并通过统一的 API 接口避免为不同格式分别编写代码，另一方面可以自动识别标题、段落、表格、列表等文档结构，同时保留相应的元数据信息。

## 三、代码示例
```markdown

from unstructured.partition.auto import partition

# PDF文件路径
pdf_path = "../../data/C2/pdf/rag.pdf"

# 使用Unstructured加载并解析PDF文档
elements = partition(
    filename=pdf_path,
    content_type="application/pdf"
)

# 打印解析结果
print(f"解析完成: {len(elements)} 个元素, {sum(len(str(e)) for e in elements)} 字符")

# 统计元素类型
from collections import Counter
types = Counter(e.category for e in elements)
print(f"元素类型: {dict(types)}")

# 显示所有元素
print("\n所有元素:")
for i, element in enumerate(elements, 1):
    print(f"Element {i} ({element.category}):")
    print(element)
    print("=" * 60)
```

partition 函数参数解析：

+ `filename`: 文档文件路径，支持本地文件路径；
+ `content_type`: 可选参数，指定MIME类型（如"application/pdf"），可绕过自动文件类型检测；
+ `file`: 可选参数，文件对象，与 filename 二选一使用；
+ `url`: 可选参数，远程文档 URL，支持直接处理网络文档；
+ `include_page_breaks`: 布尔值，是否在输出中包含页面分隔符；
+ `strategy`: 处理策略，可选 "auto"、"fast"、"hi_res" 等；
+ `encoding`: 文本编码格式，默认自动检测。

`partition`函数使用自动文件类型检测，内部会根据文件类型路由到对应的专用函数（如PDF文件会调用`partition_pdf`）。如果需要更专业的PDF处理，可以直接使用`from unstructured.partition.pdf import partition_pdf`，它提供更多PDF特有的参数选项，如OCR语言设置、图像提取、表格结构推理等高级功能，同时性能更优。

**练习**：使用`partition_pdf`替换当前`partition`函数并分别尝试用`hi_res`和`ocr_only`进行解析，观察输出结果有何变化。

各种不同模式对比

| **<font style="color:rgb(31, 31, 31);">解析模式</font>** | **<font style="color:rgb(31, 31, 31);">核心原理</font>** | **<font style="color:rgb(31, 31, 31);">结果表现 </font>** | **<font style="color:rgb(31, 31, 31);">适用场景</font>** |
| --- | --- | --- | --- |
| **<font style="color:rgb(31, 31, 31);">默认 (Auto)</font>** | <font style="color:rgb(31, 31, 31);">直接提取底层文本</font> | **<font style="color:rgb(31, 31, 31);">较快，但碎片化。</font>**<font style="color:rgb(31, 31, 31);"> 段落被切碎，标签分类完全错误，丢失表格。</font> | <font style="color:rgb(31, 31, 31);">格式极其规范的纯文本 PDF（如标准 Word 导出的公文）。</font> |
| **<font style="color:rgb(31, 31, 31);">纯 OCR (</font>**`**<font style="color:rgb(68, 71, 70);">ocr_only</font>**`<br/>**<font style="color:rgb(31, 31, 31);">)</font>** | <font style="color:rgb(31, 31, 31);">把整页当图硬扫</font> | **<font style="color:rgb(31, 31, 31);">极差，乱码连篇。</font>**<font style="color:rgb(31, 31, 31);"> 被网页分栏排版干扰，连正常的中文都无法识别。</font> | <font style="color:rgb(31, 31, 31);">全是扫描件、没有复杂排版（单栏到底）的旧文档。</font> |
| **<font style="color:rgb(31, 31, 31);">高精度 (</font>**`**<font style="color:rgb(68, 71, 70);">hi_res</font>**`<br/>**<font style="color:rgb(31, 31, 31);">)</font>** | <font style="color:rgb(31, 31, 31);">AI 版面检测 + 局部 OCR</font> | **<font style="color:rgb(31, 31, 31);">完美，结构清晰。</font>**<font style="color:rgb(31, 31, 31);"> 成功识别表格、图片说明，正文段落连贯无断句。</font> | **<font style="color:rgb(31, 31, 31);">复杂排版文档，以及所有准备用于大模型 RAG 的高质量语料抽取。</font>** |


# 文本分块
文本分块（Text Chunking）是构建 RAG 流程的关键步骤。它的原理是将加载后的长篇文档，切分成更小、更易于处理的单元。这些被切分出的文本块，是后续向量检索和模型处理的基本单位。

## 基础分块策略
### 1、固定大小分块
根据LangChain源码，这种方法的工作原理分为两个主要阶段：

（1）按段落分割：`CharacterTextSplitter` 采用默认分隔符 `"\n\n"`，使用正则表达式将文本按段落进行分割，通过 `_split_text_with_regex` 函数处理。

（2）智能合并：调用继承自父类的 `_merge_splits` 方法，将分割后的段落依次合并。该方法会监控累积长度，当超过 `chunk_size` 时形成新块，并通过重叠机制（`chunk_overlap`）保持上下文连续性，同时在必要时发出超长块的警告。

```python
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# 1. 文档加载
loader = TextLoader("../../data/C2/txt/蜂医.txt", encoding="utf-8")
docs = loader.load()

# 2. 初始化固定大小分块器
text_splitter = CharacterTextSplitter(
    chunk_size=200,    # 每个块的大小
    chunk_overlap=10   # 块之间的重叠大小
)

# 3. 执行分块
chunks = text_splitter.split_documents(docs)

# 4. 打印结果
print(f"文本被切分为 {len(chunks)} 个块。\n")
print("--- 前5个块内容示例 ---")
for i, chunk in enumerate(chunks[:5]):
    print("=" * 60)
    # chunk 是一个 Document 对象，需要访问它的 .page_content 属性来获取文本
    print(f'块 {i+1} (长度: {len(chunk.page_content)}): "{chunk.page_content}"')
```

这种方法的主要优势在于实现简单、处理速度快且计算开销小。劣势在于可能会在语义边界处切断文本，影响内容的完整性和连贯性。



### 2、递归字符分块
`RecursiveCharacterTextSplitter`算法流程： 

（1）寻找有效分隔符: 从分隔符列表中从前到后遍历，找到第一个在当前文本中存在的分隔符。如果都不存在，使用最后一个分隔符（通常是空字符串 `""`）。

（2）切分与分类处理: 使用选定的分隔符切分文本，然后遍历所有片段：

    - 如果片段不超过块大小: 暂存到 `_good_splits` 中，准备合并
    - 如果片段超过块大小:
        * 首先，将暂存的合格片段通过 `_merge_splits` 合并成块
        * 然后，检查是否还有剩余分隔符：
            + 有剩余分隔符: 递归调用 `_split_text` 继续分割
            + 无剩余分隔符: 直接保留为超长块

（3）最终处理: 将剩余的暂存片段合并成最后的块

### 3、语义分块
这种方法不依赖于固定的字符数或预设的分隔符，而是尝试根据文本的语义内涵来切分。其核心是：在语义主题发生显著变化的地方进行切分。这使得每个分块都具有高度的内部语义一致性。LangChain 提供了 `langchain_experimental.text_splitter.SemanticChunker` 来实现这一功能。

`SemanticChunker` 的工作流程可以概括为以下几个步骤：

（1）句子分割 (Sentence Splitting)：首先，使用标准的句子分割规则（例如，基于句号、问号、感叹号）将输入文本拆分成一个句子列表。

（2）上下文感知嵌入 (Context-Aware Embedding)：这是 `SemanticChunker` 的一个关键设计。该分块器不是对每个句子独立进行嵌入，而是通过 `buffer_size` 参数（默认为1）来捕捉上下文信息。对于列表中的每一个句子，这种方法会将其与前后各 `buffer_size` 个句子组合起来，然后对这个临时的、更长的组合文本进行嵌入。这样，每个句子最终得到的嵌入向量就融入了其上下文的语义。

（3）计算语义距离 (Distance Calculation)：计算每对相邻句子的嵌入向量之间的余弦距离。这个距离值量化了两个句子之间的语义差异——距离越大，表示语义关联越弱，跳跃越明显。

（4）识别断点 (Breakpoint Identification)：`SemanticChunker` 会分析所有计算出的距离值，并根据一个统计方法（默认为 `percentile`）来确定一个动态阈值。例如，它可能会将所有距离中第95百分位的值作为切分阈值。所有距离大于此阈值的点，都被识别为语义上的“断点”。

（5）合并成块 (Merging into Chunks)：最后，根据识别出的所有断点位置，将原始的句子序列进行切分，并将每个切分后的部分内的所有句子合并起来，形成一个最终的、语义连贯的文本块。

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 初始化 SemanticChunker
text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile" # 也可以是 "standard_deviation", "interquartile", "gradient"
)

loader = TextLoader("../../data/C2/txt/蜂医.txt", encoding="utf-8")
documents = loader.load()

docs = text_splitter.split_documents(documents)

print(f"文本被切分为 {len(docs)} 个块。\n")
print("--- 前2个块内容示例 ---")
for i, chunk in enumerate(docs[:2]):
    print("=" * 60)
    print(f'块 {i+1} (长度: {len(chunk.page_content)}):\n"{chunk.page_content}"')
```

### 4、基于文档结构分块
针对结构清晰的 Markdown 文档，利用其标题层级进行分块是一种高效且保留了丰富语义的方法。LangChain 提供了 `MarkdownHeaderTextSplitter` 来处理。

+ 实现原理: 该分块器的主要逻辑是“先按标题分组，再按需细分”。
    1. 定义分割规则: 用户首先需要提供一个标题层级的映射关系，例如 `[ ("#", "Header 1"), ("##", "Header 2") ]`，告诉分块器 `#` 是一级标题，`##` 是二级标题。
    2. 内容聚合: 分块器会遍历整个文档，将每个标题下的所有内容（直到下一个同级或更高级别的标题出现前）聚合在一起。每个聚合后的内容块都会被赋予一个包含其完整标题路径的元数据。

### LlamaIndex:面向节点的解析与转换
[LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/) 将数据处理流程抽象为对“节点（Node）”的操作。文档被加载后，首先会被解析成一系列的“节点”，分块只是节点转换（Transformation）中的一环。

LlamaIndex 的分块体系有以下特点：

（1）丰富的节点解析器 (Node Parser): LlamaIndex 提供了大量针对特定数据格式和方法的节点解析器，可以大致分为几类：

+ 结构感知型: 如 `MarkdownNodeParser`, `JSONNodeParser`, `CodeSplitter` 等，能理解并根据源文件的结构（如Markdown标题、代码函数）进行切分。
+ 语义感知型:
    - `SemanticSplitterNodeParser`: 与 LangChain 的 `SemanticChunker` 类似，这种解析器使用嵌入模型来检测句子之间的语义“断点”，在语义连续性明显减弱的地方切开，从而让每个 chunk 内部尽量连贯。
    - `SentenceWindowNodeParser`: 这是一种巧妙的方法。该方法将文档切分成单个的句子，但在每个句子节点（Node）的元数据中，会存储其前后相邻的N个句子（即“窗口”）。这使得在检索时，可以先用单个句子的嵌入进行精确匹配，然后将包含上下文“窗口”的完整文本送给LLM，极大地提升了上下文的质量。
+ 常规型: 如 `TokenTextSplitter`, `SentenceSplitter` 等，提供基于Token数量或句子边界的常规切分方法。

（2）灵活的转换流水线: 用户可以构建一个灵活的流水线，例如先用 `MarkdownNodeParser` 按章节切分文档，再对每个章节节点应用 `SentenceSplitter` 进行更细粒度的句子级切分。每个节点都携带丰富的元数据，记录着其来源和上下文关系。

（3）良好的互操作性: LlamaIndex 提供了 `LangchainNodeParser`，可以方便地将任何 LangChain 的 `TextSplitter` 封装成 LlamaIndex 的节点解析器，无缝集成到其处理流程中。
