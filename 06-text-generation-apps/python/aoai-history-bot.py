from openai import OpenAI
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化 OpenAI 客户端（指向本地 Ollama）
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # 任意字符串，Ollama 不验证
)

# 初始化对话历史（Python编程顾问默认角色）
conversation_history = [
    {"role": "system",
     "content": "你是一位资深的Python编程顾问，精通Python语法、最佳实践和性能优化。你的回答要：1. 专业、准确，提供可运行的代码示例；2. 解释代码背后的原理，而不仅仅是给出答案；3. 当用户提问不清晰时，主动追问细节；4. 记住之前的对话上下文，提供连贯的帮助。现在开始你的工作吧！"}
]

# ========== 核心配置：双重限制 ==========
MAX_HISTORY_ROUNDS = 10  # 最多保留10轮对话
MAX_TOTAL_CHARS = 10000  # 历史记录总字符数上限


# =======================================

# 辅助函数：按字符数截断历史
def truncate_by_chars(history, max_chars):
    total_chars = 0
    truncated_history = [history[0]]  # 始终保留系统提示词
    # 从后往前遍历（保留最新的对话）
    for msg in reversed(history[1:]):
        msg_char_count = len(msg["content"])
        # 如果加当前消息会超字符限制，就停止
        if total_chars + msg_char_count > max_chars:
            break
        # 插入到系统提示词后面（保持对话顺序）
        truncated_history.insert(1, msg)
        total_chars += msg_char_count
    return truncated_history


# 辅助函数：按轮数截断历史
def truncate_by_rounds(history, max_rounds):
    # 系统提示词(1条) + 每轮2条（用户+助手） → 总条数上限
    max_total_messages = 1 + 2 * max_rounds
    if len(history) > max_total_messages:
        # 保留系统提示词，只取最后 max_total_messages-1 条对话
        history = [history[0]] + history[-max_total_messages + 1:]
    return history


# 流式获取带双重限制的回复
def get_stream_response_with_history(user_input):
    global conversation_history

    # 1. 把用户新输入添加到对话历史
    conversation_history.append({"role": "user", "content": user_input})

    # ========== 第一步：按轮数截断 ==========
    conversation_history = truncate_by_rounds(conversation_history, MAX_HISTORY_ROUNDS)

    # ========== 第二步：按字符数二次截断 ==========
    conversation_history = truncate_by_chars(conversation_history, MAX_TOTAL_CHARS)

    try:
        # 2. 调用模型，传入截断后的完整历史
        stream = client.chat.completions.create(
            model="qwen2.5:3b",
            messages=conversation_history,
            stream=True
        )

        print("助手: ", end="", flush=True)
        assistant_reply = ""  # 临时保存AI的完整回复
        # 3. 流式输出回复
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                assistant_reply += content
        print()  # 换行

        # 4. 把AI的回复添加到历史
        conversation_history.append({"role": "assistant", "content": assistant_reply})

        # ========== 再次双重截断（防止添加回复后超限） ==========
        conversation_history = truncate_by_rounds(conversation_history, MAX_HISTORY_ROUNDS)
        conversation_history = truncate_by_chars(conversation_history, MAX_TOTAL_CHARS)

    except Exception as e:
        print(f"\n发生错误: {str(e)}")


# 主程序
if __name__ == "__main__":
    print(f"🔥 带记忆的本地千问助手（专业角色版）")
    print(f"💡 限制：最多{MAX_HISTORY_ROUNDS}轮对话 | 总字符数≤{MAX_TOTAL_CHARS}")
    print("📌 指令：quit=退出 | clear=清空记忆 | show=查看当前记忆 | role=切换角色\n")

    while True:
        user_input = input("你: ")

        # 退出指令
        if user_input.lower() == "quit":
            print("助手: 再见！")
            break

        # 清空记忆指令
        if user_input.lower() == "clear":
            # 清空后恢复默认角色（Python编程顾问）
            conversation_history = [
                {"role": "system",
                 "content": "你是一位资深的Python编程顾问，精通Python语法、最佳实践和性能优化。你的回答要：1. 专业、准确，提供可运行的代码示例；2. 解释代码背后的原理，而不仅仅是给出答案；3. 当用户提问不清晰时，主动追问细节；4. 记住之前的对话上下文，提供连贯的帮助。现在开始你的工作吧！"}]
            print("助手: 已清空所有对话记忆～恢复默认角色：Python编程顾问")
            continue

        # 查看记忆指令
        if user_input.lower() == "show":
            print("\n===== 当前对话记忆 =====")
            for i, msg in enumerate(conversation_history):
                print(f"{i}. [{msg['role']}] {msg['content'][:50]}..." if len(
                    msg['content']) > 50 else f"{i}. [{msg['role']}] {msg['content']}")
            print(
                f"总条数：{len(conversation_history)} | 总字符数：{sum(len(msg['content']) for msg in conversation_history)}")
            print("========================\n")
            continue

        # 切换角色指令（核心修复）
        if user_input.lower() == "role":
            print("\n请选择角色：")
            print("1. Python编程顾问")
            print("2. 文档解释器（帮你读代码/文档）")
            print("3. 面试辅导师（帮你准备Python面试）")
            choice = input("输入角色编号：")

            # 定义各角色的完整系统提示词
            role_prompts = {
                "1": "你是一位资深的Python编程顾问，精通Python语法、最佳实践和性能优化。你的回答要：1. 专业、准确，提供可运行的代码示例；2. 解释代码背后的原理，而不仅仅是给出答案；3. 当用户提问不清晰时，主动追问细节；4. 记住之前的对话上下文，提供连贯的帮助。现在开始你的工作吧！",
                "2": "你是专业的文档解释器，擅长把复杂的技术文档、代码逻辑用通俗的语言讲清楚。你的回答要：1. 先理解文档/代码的核心逻辑；2. 分步骤解释，避免专业术语堆砌；3. 针对用户的疑问精准解答，不偏离主题；4. 记住之前的对话上下文，保持解释的连贯性。现在开始你的工作吧！",
                "3": "你是Python面试辅导师，专注于帮助用户准备Python相关的技术面试。你的回答要：1. 梳理面试高频考点，结合实际场景讲解；2. 模拟面试提问，给出答题思路和加分点；3. 指出用户回答中的不足并给出改进建议；4. 记住之前的对话上下文，提供连贯的辅导。现在开始你的工作吧！"
            }

            if choice in role_prompts:
                # 关键：切换角色时清空旧对话历史，只保留新的系统提示词
                conversation_history = [{"role": "system", "content": role_prompts[choice]}]
                role_names = {"1": "Python编程顾问", "2": "文档解释器", "3": "面试辅导师"}
                print(f"✅ 已切换为：{role_names[choice]}（已清空旧对话历史）")
            else:
                print("❌ 无效选择，保持当前角色")
            continue

        # 正常对话
        get_stream_response_with_history(user_input)