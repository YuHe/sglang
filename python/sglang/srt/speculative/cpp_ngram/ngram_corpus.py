# -*- coding: utf-8 -*-

import logging
from collections.abc import Iterable, Sequence
from typing import Dict, List, Tuple

import numpy as np

# 通过 JIT 编译加载 C++ N-gram 语料库类（后缀自动机实现）
from sglang.jit_kernel.ngram_corpus import get_ngram_corpus_cls

logger = logging.getLogger(__name__)


# NgramCorpus: N-gram 语料库的 Python 封装
# 底层由 C++ 后缀自动机（SAM）实现，支持有状态的 N-gram 匹配和外部语料加载
class NgramCorpus:
    def __init__(
        self,
        max_trie_depth=18,      # N-gram 匹配的最大 trie 深度（查找窗口长度）
        min_bfs_breadth=1,      # BFS 搜索最小宽度（候选分支的最少数量）
        max_bfs_breadth=8,      # BFS 搜索最大宽度（候选分支的最多数量）
        draft_token_num=8,      # 每次生成的草稿 token 数量
        match_type="BFS",       # 匹配算法类型（BFS 或其他）
        capacity=1000000,       # SAM 节点容量（影响可存储的 N-gram 数量）
        external_sam_budget=0,  # 外部语料在 SAM 中的节点预算
        external_corpus_max_tokens=10000000,  # 外部语料最大 token 数
    ) -> None:
        # 动态加载 C++ 编译的语料库类
        cls = get_ngram_corpus_cls()
        # 创建底层 C++ 对象
        self._obj = cls(
            capacity=capacity,
            max_trie_depth=max_trie_depth,
            min_bfs_breadth=min_bfs_breadth,
            max_bfs_breadth=max_bfs_breadth,
            draft_token_num=draft_token_num,
            match_type=match_type,
            external_sam_budget=external_sam_budget,
            external_corpus_max_tokens=external_corpus_max_tokens,
        )
        # 默认树掩码（单 token 场景）
        self.default_mask = np.ones((1, 1), dtype=np.int64)
        self.draft_token_num = draft_token_num
        self.external_corpus_max_tokens = external_corpus_max_tokens
        # 请求 ID 到内部状态 ID 的映射（用于有状态匹配）
        self._req_id_to_state_id: Dict[str, int] = {}
        # 下一个可分配的状态 ID（单调递增）
        self._next_state_id: int = 0
        # 外部语料 ID 到已加载 token 数的映射
        self._corpus_token_counts: Dict[str, int] = {}
        # 所有外部语料已加载的总 token 数
        self._total_loaded_tokens: int = 0

    def _get_state_id(self, req_id: str) -> int:
        # 获取或创建请求对应的内部状态 ID（用于跨步有状态匹配）
        sid = self._req_id_to_state_id.get(req_id)
        if sid is None:
            # 为新请求分配一个唯一的状态 ID
            sid = self._next_state_id
            self._next_state_id += 1
            self._req_id_to_state_id[req_id] = sid
        return sid

    def batch_put(self, batch_tokens: List[List[int]]):
        # 将一批 token 序列批量插入 SAM（更新语料库）
        self._obj.insert(batch_tokens)

    def synchronize(self):
        # 等待 SAM 上的异步操作完成，确保后续查询读到最新状态
        self._obj.synchronize()  # type: ignore

    @property
    def remaining_token_budget(self) -> int:
        # 计算外部语料剩余可用 token 配额
        return self.external_corpus_max_tokens - self._total_loaded_tokens

    def load_external_corpus_named(
        self, corpus_id: str, chunks: Iterable[Sequence[int]]
    ) -> int:
        # 将外部语料以命名方式加载到 SAM
        if corpus_id in self._corpus_token_counts:
            raise ValueError(
                f"External corpus '{corpus_id}' already exists. Remove it before "
                f"adding a new corpus with the same id."
            )
        # Note(kpham-sgl): remaining_token_budget is stale (e.g if there are removes
        # during the load), which makes the budget more conservative than it should be.
        # This is acceptable because otherwise load_external_corpus_named would need to check the budget after each chunk,
        # which would be inefficient.
        # 调用底层 C++ 对象流式加载 token 块，返回加载到的 token 数
        _, loaded_token_count = self._obj.load_external_corpus_named(
            corpus_id, chunks, self.remaining_token_budget
        )
        return loaded_token_count

    # Commit corpus bookkeeping after successful load. Call only at background thread join.
    # (or after synchronous load_external_corpus_named returns)
    def commit_external_corpus_load(
        self, corpus_id: str, loaded_token_count: int
    ) -> None:
        # 提交外部语料加载的账本（记录 token 数，更新总计）
        # 必须在后台线程 join 后调用，确保加载操作已完成
        self._corpus_token_counts[corpus_id] = loaded_token_count
        self._total_loaded_tokens += loaded_token_count

    def remove_external_corpus(self, corpus_id: str) -> None:
        # 从 SAM 中移除指定外部语料，并更新账本
        self._obj.remove_corpus(corpus_id)
        old_count = self._corpus_token_counts.pop(corpus_id, 0)
        self._total_loaded_tokens -= old_count

    def list_external_corpora(self) -> Dict[str, int]:
        # 列出所有已加载的外部语料及其 token 数量
        return self._obj.list_corpora()

    def reset(self):
        # 重置整个语料库（清空 SAM 和状态映射），用于清理缓存
        self._obj.reset()  # type: ignore
        self._req_id_to_state_id.clear()
        self._next_state_id = 0

    def batch_get(
        self,
        req_ids: List[str],
        batch_tokens: List[List[int]],
        total_lens: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        # 批量有状态匹配：为每个请求从语料库中查找最佳 N-gram 草稿序列
        # 返回草稿 token 数组和对应的树掩码（numpy 格式）
        state_ids = [self._get_state_id(rid) for rid in req_ids]
        return self._obj.match_stateful(state_ids, batch_tokens, total_lens)

    def erase_match_state(self, req_ids: List[str]):
        # 清理已完成请求的匹配状态，释放 SAM 中的对应状态节点
        state_ids = []
        for rid in req_ids:
            sid = self._req_id_to_state_id.pop(rid, None)
            if sid is not None:
                state_ids.append(sid)
        if state_ids:
            self._obj.erase_states(state_ids)

    def leaf_paths_from_mask(
        self, tokens: List[int], tree_mask: List[List[int]]
    ) -> List[List[int]]:
        """
        Find all leaf paths according to the binary tree_mask (i.e., paths that are not prefixes of any other path).

        Args:
            mask   : List[List[int]]   # nxn binary matrix
            tokens : List[int]         # token list corresponding to columns

        Returns:
            List[List[int]]            # token lists of only the leaf paths, preserving their order of appearance
        """
        # 将每行掩码转为 token 索引集合，便于子集判断
        row_sets = [
            (i, {idx for idx, v in enumerate(row) if v == 1})
            for i, row in enumerate(tree_mask)
        ]
        leaf_sets = []
        leaf_rows = []

        # 从后向前遍历，筛选出不是任何其他路径前缀的叶子路径
        for i, cur_set in reversed(row_sets):
            if any(cur_set <= kept for kept in leaf_sets):
                continue
            leaf_sets.append(cur_set)
            leaf_rows.append(i)

        # 恢复原始顺序
        leaf_rows.reverse()
        result = []
        for r in leaf_rows:
            # 提取叶子路径对应的 token 序列
            path = [tokens[col] for col in range(len(tokens)) if tree_mask[r][col] == 1]
            result.append(path)

        return result

    def debug_result(
        self, decoding_ids: np.ndarray, decoding_masks: np.ndarray, tokenizer=None
    ):
        # 调试工具：打印草稿结果和叶子路径（支持 tokenizer 解码）
        decoding_ids = decoding_ids.reshape(-1, self.draft_token_num)
        decoding_masks = decoding_masks.reshape(
            -1, self.draft_token_num, self.draft_token_num
        )
        logger.info(f"\n{decoding_ids=}\n{decoding_masks=}")
        for i in range(decoding_ids.shape[0]):
            leaf_paths = self.leaf_paths_from_mask(
                decoding_ids[i].tolist(), decoding_masks[i].tolist()
            )
            if tokenizer is None:
                logger.info(f"draft path {i}: {leaf_paths}")
            else:
                logger.info(f"result {i}:")
                for leaf_path in leaf_paths:
                    logger.info(
                        f"draft path {i}: {leaf_path} -> {tokenizer.decode(leaf_path, ensure_ascii=False)}"
                    )


# main function
if __name__ == "__main__":
    format = f"%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(
        level=logging.DEBUG,
        format=format,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    # 测试：插入两个 token 序列后查询 N-gram 匹配结果
    token_ids = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 2, 3, 44, 55, 66, 77, 88, 99, 100],
    ]
    corpus = NgramCorpus(max_trie_depth=12, draft_token_num=8)
    corpus.batch_put(token_ids)

    corpus.synchronize()
    queries = [[1, 2, 3], [3, 44], [3, 6, 999]]
    decoding_ids, decoding_masks = corpus.batch_get(
        req_ids=[f"query-{i}" for i in range(len(queries))],
        batch_tokens=queries,
        total_lens=[len(q) for q in queries],
    )

    corpus.debug_result(decoding_ids, decoding_masks)
