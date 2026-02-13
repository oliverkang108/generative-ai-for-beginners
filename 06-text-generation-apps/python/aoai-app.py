# 导入必要的库
from openai import OpenAI
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化客户端，指向本地 Ollama 服务
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # 任意字符串，Ollama 不验证密钥
)

# 定义一个函数，用于发送消息并获取流式回复
def get_stream_response(user_input):
    try:
        # 调用聊天补全 API，开启流式输出
        stream = client.chat.completions.create(
            model="qwen2.5:3b",  # 换成你本地 Ollama 里的千问模型名
            messages=[
                {"role": "system", "content": "你是一个友好的AI助手。"},
                {"role": "user", "content": user_input}
            ],
            stream=True  # 开启流式输出
        )

        print("助手: ", end="", flush=True)
        # 迭代处理每个返回的 chunk
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()  # 换行
    except Exception as e:
        print(f"\n发生错误: {str(e)}")

# 主程序入口
if __name__ == "__main__":
    print("本地千问助手已启动（流式输出），输入 'quit' 退出。")
    while True:
        user_input = input("你: ")
        if user_input.lower() == "quit":
            print("助手: 再见！")
            break
        get_stream_response(user_input)