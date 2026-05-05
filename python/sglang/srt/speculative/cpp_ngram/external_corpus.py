import json
from collections.abc import Iterator
from pathlib import Path

# Must match SuffixAutomaton::kSeparatorToken in suffix_automaton.h.
# 分隔符 token，与 C++ 后缀自动机（SAM）中的 kSeparatorToken 保持一致
# 使用 int32 最小值，确保不与正常词表 ID 冲突
SEPARATOR_TOKEN = -(2**31)

# Default chunk size for streaming tokenized documents into the SAM.
# 每次流式传入 SAM 的 token 块大小，避免一次性传入过大导致内存压力
DEFAULT_CHUNK_SIZE = 4096


def iter_external_corpus_chunks(
    path: str, tokenizer, max_tokens: int, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> Iterator[list[int]]:
    """Chunk documents and yield fixed-size token chunks from a JSONL corpus file."""
    # 将语料路径转为 Path 对象，便于文件操作
    corpus_path = Path(path)
    # 校验语料文件是否存在
    if not corpus_path.is_file():
        raise ValueError(f"External ngram corpus path does not exist: {path}")
    # 必须提供 tokenizer 才能对文本进行编码
    if tokenizer is None:
        raise ValueError("A tokenizer is required to load an external ngram corpus.")
    # 最大 token 数必须为正整数
    if max_tokens <= 0:
        raise ValueError("External ngram corpus max tokens must be positive.")

    # 已累计处理的 token 总数
    total_tokens = 0
    # 标记是否已处理过上一篇文档，用于决定是否插入分隔符
    has_previous_doc = False
    with corpus_path.open("r", encoding="utf-8") as f:
        # 逐行读取 JSONL 格式的语料文件
        for line_no, line in enumerate(f, start=1):
            # 跳过空行
            if not line.strip():
                continue

            # 解析 JSON 行，获取文档文本
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in external ngram corpus at line {line_no}: {e.msg}"
                ) from e

            # 每条记录必须是字符串类型（文档文本）
            if not isinstance(record, str):
                raise ValueError(
                    "Invalid external ngram corpus record at line "
                    f"{line_no}: expected a JSON string."
                )

            # 对文档文本进行 tokenization，不添加特殊 token
            token_ids = list(tokenizer.encode(record, add_special_tokens=False))
            # 跳过空文档
            if not token_ids:
                continue

            # 计算插入分隔符所需额外 token 数（首篇文档不需要分隔符）
            separator_cost = 1 if has_previous_doc else 0
            next_total_tokens = total_tokens + separator_cost + len(token_ids)
            # 检查是否超过最大 token 配额
            if next_total_tokens > max_tokens:
                raise ValueError(
                    "External ngram corpus exceeds the configured token limit "
                    f"({max_tokens}) at line {line_no} after loading "
                    f"{total_tokens} tokens."
                )
            total_tokens = next_total_tokens

            # 若不是第一篇文档，在 token 序列头部插入分隔符，区分文档边界
            if has_previous_doc:
                token_ids = [SEPARATOR_TOKEN] + token_ids
            # 将 token 序列按 chunk_size 切块后逐块 yield，支持流式写入 SAM
            for i in range(0, len(token_ids), chunk_size):
                yield token_ids[i : i + chunk_size]
            has_previous_doc = True
