---
title: LangGraph学习
date: 2026-04-15 15:00:57
tags:
---

:::warning
🔔核心定位：基于LangChain生态，主打「图结构+状态管理」，专为复杂流程AI任务设计。
:::

## 一、LangGraph 是什么？
LangGraph 是 LangChain 生态下的进阶框架，核心是「用图结构（节点+边）定义AI任务流程」，支持循环执行、分支判断、状态管理，解决LangChain中普通Chain无法处理的复杂多步骤任务。

简单说：LangChain的Chain是「线性流水线」，LangGraph是「可循环、可分支的流程图」，比如让AI完成“任务拆解→执行→检查→修正”的闭环，只有LangGraph能轻松实现。

**核心优势：**相比普通Chain，它能处理多轮循环、条件判断，记住任务执行状态，适合复杂流程（比如多步骤Agent、业务审批、循环纠错）。

## 二、核心概念
LangGraph的核心就3个概念，比LangChain更简单，聚焦「图结构」和「状态管理」：

+ **节点（Node）：**图中的“步骤”，可以是一个函数、一个Agent、一个任务（比如“任务拆解”“执行任务”“检查结果”），是图的最小执行单元。
+ **边（Edge）：**图中的“连接线”，定义节点之间的执行顺序（比如“先拆解任务，再执行任务”），支持条件分支（比如“结果合格→结束，不合格→重新执行”）。
+ **状态（State）：**图的“记忆”，存储整个任务流程中的所有数据（比如任务内容、执行结果、中间步骤），所有节点可共享、修改状态，实现多步骤上下文联动。

**补充：**LangGraph的核心逻辑——用节点定义“做什么”，用边定义“怎么做（顺序/分支）”，用状态记录“做了什么”，实现复杂流程的自动化闭环。



## 三、代码示例
```python
import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# 1. 加载环境变量
load_dotenv()

# 2. 初始化大模型（通义千问）
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"), 
    base_url="[https://dashscope.aliyuncs.com/compatible-mode/v1](https://dashscope.aliyuncs.com/compatible-mode/v1)", 
    model="qwen-turbo",  
    temperature=0.7 
)

# 3. 定义状态（不变）
class State(TypedDict):
    question: str
    answer: str
    refined_answer: str

# 4. 节点1： 回答问题
def answer_node(state: State) -> State:
    question = state["question"]
    answer = llm.invoke(f"简介回答：{question}，不超过50字")

    return {
        "question": question,
        "answer": answer.content,
        "refined_answer": ""
    }

# 5. 节点2： 优化回答
def refine_node(state: State) -> State:
    answer = state["answer"]
    refined = llm.invoke(f"整理以下回答，保持原意，更简洁：{answer}")

    return {
        **state,
        "refined_answer": refined.content  
    }

# 6. 构建流程图
graph = StateGraph(State)
graph.add_node("answer", answer_node)
graph.add_node("refine", refine_node)
graph.add_edge(START, "answer")
graph.add_edge("answer", "refine")  
graph.add_edge("refine", END)

app = graph.compile()

# 运行
initial_state = {"question": "LangGraph和LangChain的核心区别是什么？"}
result = app.invoke(initial_state)

print("原始回答：", result["answer"])
print("整理后回答：", result["refined_answer"])

```

+ 1. 状态定义：必须用TypedDict明确状态字段，所有节点的输入输出都要围绕状态，避免状态缺失导致报错。
+ 2. 边的连接：线性边用add_edge，条件分支用add_conditional_edges，注意分支判断的返回值要和边的映射对应（比如Demo2中返回“合格”对应结束）。
+ 3. 循环终止：循环流程一定要设置终止条件（比如Demo3的索引判断），否则会出现无限循环。
+ 4. 复用LangChain资源：LangGraph可直接复用LangChain的LLM、Agent、Tool，无需重新配置，比如把Demo中的llm换成LangChain的Agent即可实现更复杂任务。
+ 5. 调试技巧：可在节点函数中打印状态（print(state)），快速定位节点执行异常、状态更新错误。




## 总结
LangGraph入门核心是「图结构+状态管理」，不用死记理论，先跑通3个Demo，理解节点、边、状态的关系即可。它是LangChain的进阶补充，适合处理复杂多步骤、循环、分支任务，后续可联动LangChain的Agent、Tool，实现更强大的AI应用（比如多步骤任务Agent、业务流程自动化）。
