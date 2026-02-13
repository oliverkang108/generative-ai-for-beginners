from openai import OpenAI
import os
from dotenv import load_dotenv
import numpy as np
import time

# 加载环境变量
load_dotenv()

# 初始化 OpenAI 客户端（指向本地 Ollama）
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # 任意字符串，Ollama 不验证
)

# 定义通用系统提示词（包含RAG规则）
BASE_SYSTEM_PROMPTS = {
    "python_consultant": "你是一位资深的Python编程顾问，精通Python语法、最佳实践和性能优化。你的回答必须严格遵守以下规则：1. 如果开启了RAG模式，必须100%基于检索到的知识库内容回答，禁止使用任何知识库外的示例、概念或表述；2. 必须完整列出知识库中提到的所有相关内容，不能遗漏；3. 示例代码必须100%复用知识库中的原文，不能修改或自创；4. 专业、准确，解释代码背后的原理；5. 记住之前的对话上下文，提供连贯的帮助。现在开始你的工作吧！",
    "doc_interpreter": "你是专业的文档解释器，擅长把复杂的技术文档、代码逻辑用通俗的语言讲清楚。你的回答必须严格遵守以下规则：1. 如果开启了RAG模式，必须100%基于检索到的知识库内容回答，禁止使用任何知识库外的信息；2. 分步骤解释，避免专业术语堆砌；3. 针对用户的疑问精准解答，不偏离主题；4. 记住之前的对话上下文，保持解释的连贯性。现在开始你的工作吧！",
    "interview_coach": "你是Python面试辅导师，专注于帮助用户准备Python相关的技术面试。你的回答必须严格遵守以下规则：1. 如果开启了RAG模式，必须100%基于检索到的知识库内容回答，禁止使用任何知识库外的信息；2. 梳理面试高频考点，结合知识库内容讲解；3. 指出用户回答中的不足并给出改进建议；4. 记住之前的对话上下文，提供连贯的辅导。现在开始你的工作吧！"
}

# 初始化对话历史（默认角色：Python编程顾问）
conversation_history = [
    {"role": "system", "content": BASE_SYSTEM_PROMPTS["python_consultant"]}
]

# ========== 核心配置 ==========
MAX_HISTORY_ROUNDS = 10
MAX_TOTAL_CHARS = 10000
RAG_ENABLED = False  # RAG 模式开关
KNOWLEDGE_BASE = []  # 存储知识库片段和向量
MAX_CHUNK_LENGTH = 500  # 单知识库片段最大字符数


# =======================================

# --- RAG 核心函数 ---
def smart_split_text(content, max_length=MAX_CHUNK_LENGTH):
    """智能文本切分：按段落+长度限制"""
    # 先按空行切分段落
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    chunks = []

    for para in paragraphs:
        # 段落长度合适，直接加入
        if len(para) <= max_length:
            chunks.append(para)
        else:
            # 长段落按句子切分（中文按句号，英文按句点）
            sentences = []
            temp = ""
            for char in para:
                temp += char
                if char in ["。", ".", "！", "？"] and len(temp) > 50:
                    sentences.append(temp.strip())
                    temp = ""
            if temp:
                sentences.append(temp.strip())
            # 合并句子，不超过最大长度
            current_chunk = ""
            for sent in sentences:
                if len(current_chunk + sent) <= max_length:
                    current_chunk += sent + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sent + " "
            if current_chunk:
                chunks.append(current_chunk.strip())
    return chunks


def load_knowledge_base(file_path):
    """加载本地txt文件到知识库（带进度+异常处理）"""
    global KNOWLEDGE_BASE
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在：{file_path}")
        return False

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        # 兼容GBK编码文件
        with open(file_path, "r", encoding="gbk") as f:
            content = f.read()
    except Exception as e:
        print(f"❌ 读取文件失败：{str(e)}")
        return False

    if not content:
        print("❌ 文件内容为空")
        return False

    # 智能切分文本
    chunks = smart_split_text(content)
    if not chunks:
        print("❌ 文本切分后无有效内容")
        return False

    # 向量化每个片段（显示进度）- 依然使用 nomic-embed-text
    KNOWLEDGE_BASE = []
    total_chunks = len(chunks)
    print(f"📄 开始向量化 {total_chunks} 个知识库片段（使用 nomic-embed-text 模型）...")

    for i, chunk in enumerate(chunks, 1):
        try:
            # 调用 Ollama 的 embedding 接口生成向量
            response = client.embeddings.create(
                model="nomic-embed-text",  # 核心：保留 nomic-embed-text 模型
                input=chunk
            )
            embedding = np.array(response.data[0].embedding)
            # 过滤空向量
            if len(embedding) == 0 or np.all(embedding == 0):
                print(f"⚠️ 跳过空向量片段 {i}/{total_chunks}")
                continue
            KNOWLEDGE_BASE.append({
                "text": chunk,
                "embedding": embedding,
                "chunk_id": i
            })
            print(f"✅ 完成 {i}/{total_chunks} 个片段向量化", end="\r")
            time.sleep(0.1)  # 避免请求过快
        except Exception as e:
            print(f"\n⚠️ 片段 {i}/{total_chunks} 向量化失败：{str(e)}")
            continue

    print(f"\n✅ 已加载知识库：共 {len(KNOWLEDGE_BASE)} 个有效片段（原始 {total_chunks} 个）")
    return True


def retrieve_relevant_context(query, top_k=1):
    """根据问题检索最相关的知识库片段（带相似度得分+异常处理）"""
    if not KNOWLEDGE_BASE:
        return "⚠️ 知识库为空，请先使用 'load' 命令加载文件。"

    try:
        # 生成问题的向量 - 依然使用 nomic-embed-text
        response = client.embeddings.create(
            model="nomic-embed-text",
            input=query
        )
        query_embedding = np.array(response.data[0].embedding)
        # 检查问题向量是否有效
        if len(query_embedding) == 0 or np.all(query_embedding == 0):
            return "⚠️ 问题向量化失败，请重新提问。"
    except Exception as e:
        return f"⚠️ 生成问题向量失败：{str(e)}"

    # 计算相似度（余弦相似度）
    similarities = []
    for item in KNOWLEDGE_BASE:
        try:
            similarity = np.dot(query_embedding, item["embedding"]) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(item["embedding"]))
            similarities.append((similarity, item["text"], item["chunk_id"]))
        except Exception as e:
            print(f"⚠️ 计算片段 {item['chunk_id']} 相似度失败：{str(e)}")
            continue

    if not similarities:
        return "⚠️ 无有效相似度结果，无法检索。"

    # 按相似度排序，取前 top_k 个
    similarities.sort(reverse=True, key=lambda x: x[0])
    top_results = similarities[:top_k]

    # 拼接成上下文（带相似度得分，方便调试）
    context_parts = []
    for sim, text, chunk_id in top_results:
        context_parts.append(f"【片段 {chunk_id} | 相似度 {sim:.4f}】\n{text}")

    context = "\n\n".join(context_parts)
    # 结构化强制指令
    return f"""【重要规则】
1. 仅使用以下知识库内容回答，禁止添加任何外部知识、自创示例；
2. 必须完整列出知识库中提到的所有相关内容，不能遗漏；
3. 示例代码必须100%复用原文，不能修改；
4. 回答结构：先总结核心内容，再逐一贴出知识库中的定义+示例。

【知识库内容】
{context}"""


# --- 原有记忆与截断函数（优化逻辑）---
def truncate_by_chars(history, max_chars):
    total_chars = 0
    truncated_history = [history[0]]  # 保留系统提示词
    # 从后往前遍历，保留最新对话
    for msg in reversed(history[1:]):
        msg_char_count = len(msg["content"])
        if total_chars + msg_char_count > max_chars:
            break
        truncated_history.insert(1, msg)
        total_chars += msg_char_count
    return truncated_history


def truncate_by_rounds(history, max_rounds):
    # 系统提示词(1条) + 每轮2条（用户+助手）
    max_total_messages = 1 + 2 * max_rounds
    if len(history) > max_total_messages:
        history = [history[0]] + history[-max_total_messages + 1:]
    return history


def get_stream_response_with_history(user_input):
    global conversation_history, RAG_ENABLED

    # 如果开启了 RAG，先检索相关信息
    if RAG_ENABLED:
        context = retrieve_relevant_context(user_input)
        user_input_with_context = f"""【用户问题】：{user_input}
{context}"""
    else:
        user_input_with_context = user_input

    # 添加用户输入到历史
    conversation_history.append({"role": "user", "content": user_input_with_context})
    # 双重截断
    conversation_history = truncate_by_rounds(conversation_history, MAX_HISTORY_ROUNDS)
    conversation_history = truncate_by_chars(conversation_history, MAX_TOTAL_CHARS)

    try:
        # 生成回复（仅保留 OpenAI SDK 兼容的参数，修复报错）
        stream = client.chat.completions.create(
            model="qwen2.5:3b",
            messages=conversation_history,
            stream=True,
            temperature=0.0,  # 0=完全确定性输出，避免自创内容
            top_p=1.0  # 只选概率最高的token
            # 已删除：max_new_tokens、seed、stop 等不兼容参数
        )

        print("助手: ", end="", flush=True)
        assistant_reply = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                assistant_reply += content
        print()

        # 添加助手回复到历史
        conversation_history.append({"role": "assistant", "content": assistant_reply})
        # 再次截断
        conversation_history = truncate_by_rounds(conversation_history, MAX_HISTORY_ROUNDS)
        conversation_history = truncate_by_chars(conversation_history, MAX_TOTAL_CHARS)

    except Exception as e:
        print(f"\n❌ 生成回复失败: {str(e)}")


# --- 主程序（优化用户体验）---
def show_knowledge_base():
    """查看已加载的知识库片段"""
    if not KNOWLEDGE_BASE:
        print("⚠️ 暂无加载的知识库")
        return
    print("\n===== 当前知识库片段 =====")
    for i, item in enumerate(KNOWLEDGE_BASE, 1):
        preview = item["text"][:100] + "..." if len(item["text"]) > 100 else item["text"]
        print(f"{i}. 片段 {item['chunk_id']}：{preview}")
    print(f"总有效片段数：{len(KNOWLEDGE_BASE)}")
    print("========================\n")


if __name__ == "__main__":
    print(f"🔥 带记忆的本地千问助手（RAG 知识库增强版）")
    print(f"💡 限制：最多{MAX_HISTORY_ROUNDS}轮对话 | 总字符数≤{MAX_TOTAL_CHARS}")
    print(f"🔧 Embedding 模型：nomic-embed-text | 对话模型：qwen2.5:3b")
    print(
        "📌 指令：quit=退出 | clear=清空记忆 | show=查看记忆 | role=切换角色 | load=加载知识库 | rag=开启/关闭RAG | show_kb=查看知识库\n")

    while True:
        user_input = input("你: ").strip()

        if user_input.lower() == "quit":
            print("助手: 再见！")
            break

        if user_input.lower() == "clear":
            # 清空后恢复默认角色（保留RAG规则）
            conversation_history = [{"role": "system", "content": BASE_SYSTEM_PROMPTS["python_consultant"]}]
            print("助手: 已清空所有对话记忆～恢复默认角色：Python编程顾问")
            continue

        if user_input.lower() == "show":
            print("\n===== 当前对话记忆 =====")
            for i, msg in enumerate(conversation_history):
                preview = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
                print(f"{i}. [{msg['role']}] {preview}")
            print(
                f"总条数：{len(conversation_history)} | 总字符数：{sum(len(msg['content']) for msg in conversation_history)}")
            print("========================\n")
            continue

        if user_input.lower() == "show_kb":
            show_knowledge_base()
            continue

        if user_input.lower() == "role":
            print("\n请选择角色：")
            print("1. Python编程顾问")
            print("2. 文档解释器（帮你读代码/文档）")
            print("3. 面试辅导师（帮你准备Python面试）")
            choice = input("输入角色编号：").strip()

            role_map = {
                "1": ("python_consultant", "Python编程顾问"),
                "2": ("doc_interpreter", "文档解释器"),
                "3": ("interview_coach", "面试辅导师")
            }

            if choice in role_map:
                prompt_key, role_name = role_map[choice]
                conversation_history = [{"role": "system", "content": BASE_SYSTEM_PROMPTS[prompt_key]}]
                print(f"✅ 已切换为：{role_name}（已清空旧对话历史）")
            else:
                print("❌ 无效选择，保持当前角色")
            continue

        if user_input.lower() == "load":
            file_path = input("请输入要加载的 .txt 文件路径：").strip()
            if load_knowledge_base(file_path):
                print("💡 现在可以输入 'rag' 开启 RAG 模式，提问时将基于知识库回答。")
            continue

        if user_input.lower() == "rag":
            RAG_ENABLED = not RAG_ENABLED
            status = "开启" if RAG_ENABLED else "关闭"
            print(f"✅ RAG 模式已{status}")
            continue

        if not user_input:
            print("助手: 请输入有效内容～")
            continue

        get_stream_response_with_history(user_input)