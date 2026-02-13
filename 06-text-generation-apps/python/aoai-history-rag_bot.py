from openai import OpenAI
import os
from dotenv import load_dotenv
import numpy as np

# 加载环境变量
load_dotenv()

# 初始化 OpenAI 客户端（指向本地 Ollama）
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # 任意字符串，Ollama 不验证
)

# 初始化对话历史（默认角色：Python编程顾问）
conversation_history = [
    {"role": "system",
     "content": "你是一位资深的Python编程顾问，精通Python语法、最佳实践和性能优化。你的回答要：1. 专业、准确，提供可运行的代码示例；2. 解释代码背后的原理，而不仅仅是给出答案；3. 当用户提问不清晰时，主动追问细节；4. 记住之前的对话上下文，提供连贯的帮助。现在开始你的工作吧！"}
]

# ========== 核心配置 ==========
MAX_HISTORY_ROUNDS = 10
MAX_TOTAL_CHARS = 10000
RAG_ENABLED = False  # RAG 模式开关
KNOWLEDGE_BASE = []  # 存储知识库片段和向量


# =======================================

# --- RAG 核心函数 ---

def load_knowledge_base(file_path):
    """加载本地txt文件到知识库"""
    global KNOWLEDGE_BASE
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在：{file_path}")
        return False

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 简单切分：按段落切分（可优化为更智能的切分）
    chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]

    # 向量化每个片段
    KNOWLEDGE_BASE = []
    for chunk in chunks:
        # 调用 Ollama 的 embedding 接口生成向量
        response = client.embeddings.create(
            model="nomic-embed-text",  # 轻量 embedding 模型，需先运行 `ollama pull nomic-embed-text`
            input=chunk
        )
        embedding = np.array(response.data[0].embedding)
        KNOWLEDGE_BASE.append({"text": chunk, "embedding": embedding})

    print(f"✅ 已加载知识库：共 {len(KNOWLEDGE_BASE)} 个片段")
    return True


def retrieve_relevant_context(query, top_k=3):
    """根据问题检索最相关的知识库片段"""
    if not KNOWLEDGE_BASE:
        return "⚠️ 知识库为空，请先使用 'load' 命令加载文件。"

    # 生成问题的向量
    response = client.embeddings.create(
        model="nomic-embed-text",
        input=query
    )
    query_embedding = np.array(response.data[0].embedding)

    # 计算相似度（余弦相似度）
    similarities = []
    for item in KNOWLEDGE_BASE:
        similarity = np.dot(query_embedding, item["embedding"]) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(item["embedding"]))
        similarities.append((similarity, item["text"]))

    # 按相似度排序，取前 top_k 个
    similarities.sort(reverse=True, key=lambda x: x[0])
    top_chunks = [chunk for _, chunk in similarities[:top_k]]

    # 拼接成上下文
    context = "\n---\n".join(top_chunks)
    return f"以下是从知识库中检索到的相关信息：\n{context}"


# --- 原有记忆与截断函数 ---

def truncate_by_chars(history, max_chars):
    total_chars = 0
    truncated_history = [history[0]]
    for msg in reversed(history[1:]):
        msg_char_count = len(msg["content"])
        if total_chars + msg_char_count > max_chars:
            break
        truncated_history.insert(1, msg)
        total_chars += msg_char_count
    return truncated_history


def truncate_by_rounds(history, max_rounds):
    max_total_messages = 1 + 2 * max_rounds
    if len(history) > max_total_messages:
        history = [history[0]] + history[-max_total_messages + 1:]
    return history


def get_stream_response_with_history(user_input):
    global conversation_history, RAG_ENABLED

    # 如果开启了 RAG，先检索相关信息
    if RAG_ENABLED:
        context = retrieve_relevant_context(user_input)
        user_input_with_context = f"基于以下知识库信息回答问题：\n{context}\n\n用户问题：{user_input}"
    else:
        user_input_with_context = user_input

    conversation_history.append({"role": "user", "content": user_input_with_context})
    conversation_history = truncate_by_rounds(conversation_history, MAX_HISTORY_ROUNDS)
    conversation_history = truncate_by_chars(conversation_history, MAX_TOTAL_CHARS)

    try:
        stream = client.chat.completions.create(
            model="qwen2.5:3b",
            messages=conversation_history,
            stream=True
        )

        print("助手: ", end="", flush=True)
        assistant_reply = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                assistant_reply += content
        print()

        conversation_history.append({"role": "assistant", "content": assistant_reply})
        conversation_history = truncate_by_rounds(conversation_history, MAX_HISTORY_ROUNDS)
        conversation_history = truncate_by_chars(conversation_history, MAX_TOTAL_CHARS)

    except Exception as e:
        print(f"\n发生错误: {str(e)}")


# --- 主程序 ---

if __name__ == "__main__":
    print(f"🔥 带记忆的本地千问助手（RAG 知识库版）")
    print(f"💡 限制：最多{MAX_HISTORY_ROUNDS}轮对话 | 总字符数≤{MAX_TOTAL_CHARS}")
    print("📌 指令：quit=退出 | clear=清空记忆 | show=查看记忆 | role=切换角色 | load=加载知识库 | rag=开启/关闭RAG\n")

    while True:
        user_input = input("你: ")

        if user_input.lower() == "quit":
            print("助手: 再见！")
            break

        if user_input.lower() == "clear":
            conversation_history = [{"role": "system",
                                     "content": "你是一位资深的Python编程顾问，精通Python语法、最佳实践和性能优化。你的回答要：1. 专业、准确，提供可运行的代码示例；2. 解释代码背后的原理，而不仅仅是给出答案；3. 当用户提问不清晰时，主动追问细节；4. 记住之前的对话上下文，提供连贯的帮助。现在开始你的工作吧！"}]
            print("助手: 已清空所有对话记忆～恢复默认角色：Python编程顾问")
            continue

        if user_input.lower() == "show":
            print("\n===== 当前对话记忆 =====")
            for i, msg in enumerate(conversation_history):
                print(f"{i}. [{msg['role']}] {msg['content'][:50]}..." if len(
                    msg['content']) > 50 else f"{i}. [{msg['role']}] {msg['content']}")
            print(
                f"总条数：{len(conversation_history)} | 总字符数：{sum(len(msg['content']) for msg in conversation_history)}")
            print("========================\n")
            continue

        if user_input.lower() == "role":
            print("\n请选择角色：")
            print("1. Python编程顾问")
            print("2. 文档解释器（帮你读代码/文档）")
            print("3. 面试辅导师（帮你准备Python面试）")
            choice = input("输入角色编号：")


            role_prompts = {
                "1": "你是一位资深的Python编程顾问，精通Python语法、最佳实践和性能优化。你的回答要：1. 专业、准确，提供可运行的代码示例；2. 解释代码背后的原理，而不仅仅是给出答案；3. 当用户提问不清晰时，主动追问细节；4. 记住之前的对话上下文，提供连贯的帮助。现在开始你的工作吧！",
                "2": "你是专业的文档解释器，擅长把复杂的技术文档、代码逻辑用通俗的语言讲清楚。你的回答要：1. 先理解文档/代码的核心逻辑；2. 分步骤解释，避免专业术语堆砌；3. 针对用户的疑问精准解答，不偏离主题；4. 记住之前的对话上下文，保持解释的连贯性。现在开始你的工作吧！",
                "3": "你是Python面试辅导师，专注于帮助用户准备Python相关的技术面试。你的回答要：1. 梳理面试高频考点，结合实际场景讲解；2. 模拟面试提问，给出答题思路和加分点；3. 指出用户回答中的不足并给出改进建议；4. 记住之前的对话上下文，提供连贯的辅导。现在开始你的工作吧！"
            }

            if choice in role_prompts:
                conversation_history = [{"role": "system", "content": role_prompts[choice]}]
                role_names = {"1": "Python编程顾问", "2": "文档解释器", "3": "面试辅导师"}
                print(f"✅ 已切换为：{role_names[choice]}（已清空旧对话历史）")
            else:
                print("❌ 无效选择，保持当前角色")
            continue

        if user_input.lower() == "load":
            file_path = input("请输入要加载的 .txt 文件路径：")
            if load_knowledge_base(file_path):
                print("💡 现在可以输入 'rag' 开启 RAG 模式，提问时将基于知识库回答。")
            continue

        if user_input.lower() == "rag":
            RAG_ENABLED = not RAG_ENABLED
            status = "开启" if RAG_ENABLED else "关闭"
            print(f"✅ RAG 模式已{status}")
            continue

        get_stream_response_with_history(user_input)