---
title: langChain学习
date: 2026-04-13 16:53:20
tags:
---
## 一、LangChain 是什么？
LangChain 是一个AI应用开发框架，核心作用是「串联大模型、工具、数据」，帮你快速实现复杂AI任务（比如智能问答、多步骤任务执行、知识库问答），不用重复写底层代码，提升开发效率。

简单说：它就是AI应用的「脚手架」，把大模型当成“大脑”，把工具（搜索、文件读取等）当成“手脚”，用LangChain就能轻松让“大脑”指挥“手脚”干活。


## 二、核心概念
LangChain核心就5个组件，掌握这5个，就能应对80%的入门场景：

+ **LLM**：大模型入口，比如OpenAI、通义千问，是AI的“大脑”。
+ **Chain**：任务流水线，把多个步骤（比如“提问→调用大模型→输出结果”）串联起来，自动执行。
+ **Agent**：智能决策者，能根据任务自动判断“该做什么、该调用哪个工具”，比Chain更灵活。
+ **Memory**：记忆功能，让AI记住对话历史（比如多轮聊天时，记得上一轮说的话）。
+ **Tool**：工具集，比如联网搜索、读取本地文件，是AI的“手脚”，帮AI突破自身知识边界。


## 三、3个实操Demo
### Demo1：基础LLM调用（最入门，验证环境）
功能：调用大模型，直接回答问题，相当于“简单版ChatGPT”。

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # 注意修改点 1：千问是对话模型，改用 ChatOpenAI

# 1. 加载环境变量（密钥）
load_dotenv()

# 2. 初始化大模型（大脑）
llm = ChatOpenAI(
    # 明确传入通义千问的 API Key
    api_key=os.getenv("DASHSCOPE_API_KEY"), 
    
    # 这是最核心的一句，把请求地址从美国 OpenAI 换成阿里云
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", 
    
    # 换成通义千问的模型代号（qwen-turbo 速度快且极其便宜）
    model="qwen-turbo",  
    
    temperature=0.7  # 创造性，0-1之间，越小越严谨
)

# 3. 调用大模型回答问题
question = "LangChain是什么？用一句话简单说明"
result = llm.invoke(question)

# 4. 打印结果
# 注意修改点 5：ChatOpenAI 返回的是一个包含各种信息的对象，加上 .content 才能只提取文字
print("回答：", result.content)
```

### Demo2核心代码：Chain串联任务（入门核心）
功能：把“提问→处理问题→调用大模型→整理结果”串联起来，实现简单的任务自动化。

```python

# 定义提问模板（让大模型按固定格式输出）
prompt = PromptTemplate(
    input_variables=["question"],  # 传入的变量（问题）
    template="请用简洁的语言回答以下问题，不超过50字：{question}"  # 模板内容
)

# 创建Chain（串联模板和大模型）
chain = LLMChain(llm=llm, prompt=prompt)

# 执行Chain，传入问题
result = chain.invoke({"question": "LangChain核心作用是什么？"})
```

核心亮点：通过PromptTemplate固定输出格式，Chain自动完成“填充模板→调用大模型→返回结果”，比直接调用LLM更灵活。

### Demo3：Agent工具调用（进阶，体现LangChain价值）
功能：让AI自动判断是否需要调用工具

```python
# 定义工具
@tool
def calculator(expression: str) -> str:
    """
    用于计算数学表达式的工具
    参数：expression 是数学字符串，例如 '3**5' 或 '10+20*3'
    返回：计算结果
    """
    return str(eval(expression))

# 工具列表
tools = [calculator]

#  创建 Agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能助手，可以使用工具计算数学题"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

#   运行
result = agent_executor.invoke({
    "input": "3的5次方等于多少？"
})

```

```python
@tool
def 函数名(参数: 参数类型) -> str:
    """
    工具作用（必须写）
    参数：参数说明
    """
    # 你的逻辑
    return "结果字符串"
```

## 四、新手避坑指南
+ 1. 密钥不要硬编码：一定要用.env文件管理，避免泄露。
+ 2. 模型选择：可以用免费的模型先体验。
+ 3. verbose=True：调试时一定要开启，能快速定位问题（比如Agent没调用工具、Chain执行失败）。
+ 4. 工具调用：不同工具需要对应的API Key（比如Serper搜索），按提示申请即可，大多有免费额度。

## 五、核心学习资源
+ 官方文档：https://python.langchain.com/docs/get_started/introduction （最权威，查用法）
+ 官方示例：https://github.com/langchain-ai/langchain/tree/master/examples （现成代码，直接改）

## 总结
LangChain入门不用贪多，先掌握「LLM+Chain+Agent」三个核心，跑通上面3个Demo，就具备了动手做简单AI应用的能力。后续再逐步学习Memory（记忆）、RAG（知识库），就能实现更复杂的功能（比如企业知识库问答、多轮对话机器人）。
