import ollama

response = ollama.embed(
    model="qllama/bge-reranker-v2-m3:q8_0",
    input=[
        "周一几点开门",
        "本店周一至周五上午九点开始营业",
        "本店提供外卖服务"
    ]
)

print(response)