---
title: LangGraph实现一个简单得ReAct问答智能体Demo
date: 2026-04-15 22:12:58
tags:
---

<font style="color:rgb(0, 0, 0);background-color:rgba(255, 255, 255, 0.5);">根据LangGraph的基本构建思路，结合官方文档，加入简单的搜索工具，完成了一个简单的智能回答demo。以下为重点实现步骤。</font>

#### 一、模型配置
使用千问api，初始化大模型

```plain
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"), 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", 
    model="qwen-max",  
    temperature=0.7 
)
```



#### 二、定义状态（State)
 LangGraph 的核心是“状态流”。我们定义一个 `messages` 列表来存储对话历史。  

```python
class State(TypedDict) :
    # 状态是一个字典，字典的键为messages,值是一个装有对话消息的列表（习惯叫消息列表）。
    # 使用了add_messages函数来追加新消息到列表中，而不是覆盖它。
    messages: Annotated[list, add_messages]
```

#### 三、工具集定义
 这里集成了 DuckDuckGo 联网搜索工具  。

```python
search_web = DuckDuckGoSearchRun(
    name="duckduckgo_search", 
    description="搜索互联网获取最新信息。适用于需要实时数据的场景。"
   )
    
tools = [search_web ]
# 将工具绑定到 LLM，使模型具备调用工具的能力
llm_with_tools = llm.bind_tools(tools)
```



#### 四、定义chatbot聊天节点
```python
def chatbot(state: State) :
    # 必须使用 llm_with_tools，否则模型不知道自己有工具可用
    return {"messages": [llm_with_tools.invoke(state["messages"])]}
```



#### 五、创建图，向图中添加节点和边
 这是实现 **ReAct 循环** 的核心：通过条件边判断是否需要进入工具节点。  

```python
graph_builder = StateGraph(State)
# 添加推理节点和执行节点
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))

# 设置逻辑连线
graph_builder.add_edge(START, "chatbot")
# 条件路由：如果模型决定调用工具，则前往 tools 节点，否则结束
graph_builder.add_conditional_edges("chatbot", tools_condition)
# 执行完工具后，将结果返回给 chatbot 重新推理
graph_builder.add_edge("tools", "chatbot")
```

#### 六、持久化记忆与运行  
 引入 `MemorySaver`，通过 `thread_id` 标识不同的对话，实现跨轮次记忆。  

```python
# 编译图并开启记忆功能
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# 只需要设定一个唯一的 thread_id，LangGraph 就会自动帮我们把对话存起来
config = {"configurable": {"thread_id": "my_first_chat"}}

while True:
    try:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
            
        result = graph.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config # 传入配置以保持记忆
        )
        
        # result["messages"] 里包含了所有历史消息，[-1] 就是 AI 的最新回复
        print("Assistant:", result["messages"][-1].content)
        
    except Exception as e:
        print(f"发生错误: {e}")
        break
```

#### 七、输出示例
```plain
User: 2026年是什么年  
Assistant: 2026年是农历的丙午年，也就是马年。这一年从公历2026年2月17日（农历正月初一）开始，至2027年2月5日结束。根据中国的传统历法，这是六十年一遇的"火马年"或"红马年"。在传统文化中，马象征着奔腾、活力、进取与自由，因此丙午马年常被寄予一往无前、事业腾飞的美好寓意。对于生肖属马的人来说，2026年将是他们的本命年。

User: 马的成语有什么  
Assistant: 关于马的成语有很多，这里列举一些常见的：

1. 一马当先 - 比喻在工作、学习等方面走在最前面。
2. 马到成功 - 形容事情进展顺利，很快就取得了成功。
3. 千军万马 - 形容兵马众多，声势浩大。
4. 快马加鞭 - 比喻加快速度或加紧进行某项工作。
5. 万马奔腾 - 形容气势磅礴，场面壮观。
6. 老马识途 - 比喻有经验的人对事情比较熟悉。
7. 走马观花 - 比喻粗略地观察事物。
8. 马不停蹄 - 形容连续不断地工作或赶路。
9. 龙马精神 - 形容人精力充沛，活力四射。
10. 指鹿为马 - 比喻故意颠倒黑白，混淆是非。

这些成语都以马为主题，各自表达了不同的含义和用法。如果你需要更多关于马的成语或者是特定成语的故事背景，请告诉我！

User: q  
Goodbye!
```

#### 