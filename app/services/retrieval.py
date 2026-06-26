"""
Hybrid retrieval service using SQLAlchemy 2.0 Core CTE chain.
Performs dual-path vector search (題目向量 + 回覆向量) for questionnaire data
and single-path content vector search for knowledge document data, in parallel.
Results from both tracks are merged and globally ranked by cosine distance.
"""
import logging
import time
from dataclasses import dataclass, field

from sqlalchemy import (
    Integer,
    column,
    func,
    literal,
    select,
    true,
    union_all,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models import (
    問卷主檔, 問卷題目切塊, 問卷題目檔, 問卷附件檔,
    知識文獻主檔, 文獻節點檔, 文獻切塊檔,
)
from app.services.embedding import get_embedding

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    chunks: list[dict] = field(default_factory=list)
    raw_context: str = ""
    references: list[dict] = field(default_factory=list)
    attachment_urls: list[str] = field(default_factory=list)
    knowledge_context: str = ""
    knowledge_references: list[dict] = field(default_factory=list)


async def hybrid_retrieve(
    query: str,
    session: AsyncSession,
    similarity_threshold: float | None = None,
    top_n: int | None = None,
) -> RetrievalResult:
    """
    Perform hybrid retrieval across two independent tracks:

    Track A (questionnaire): dual-path vector search (題目向量 + 回覆向量) with
        adjacent chunk expansion. Returns raw_context and references.

    Track B (knowledge documents): single-path content vector search (內容向量) with
        adjacent chunk expansion. Returns knowledge_context (XML blocks by doc_type)
        and knowledge_references.

    Both tracks run as separate CTE chains executed concurrently via asyncio.gather.
    Each track is independently capped at top_n — they do NOT compete for the same quota.
    """
    settings = get_settings()
    threshold = similarity_threshold if similarity_threshold is not None else settings.similarity_threshold
    effective_top_n = top_n if top_n is not None else settings.top_n
    top_k = max(settings.top_k, effective_top_n + 25)

    logger.info(
        "開始檢索：查詢長度=%d  閾値=%.3f  top_n=%d  top_k=%d",
        len(query), threshold, effective_top_n, top_k,
    )
    start = time.perf_counter()

    # Step 1: Embed query
    query_vector = await get_embedding(query)
    logger.debug("查詢向量取得完成：維度=%d", len(query_vector))

    # ── Track A: 問卷 CTE chain ─────────────────────────────────────────────

    t = 問卷題目切塊

    # CTE 1: chunk_hits — cosine distance on embedding (single vector), LIMIT top_k
    chunk_dist = t.c["embedding"].cosine_distance(query_vector).label("distance")
    chunk_hits = (
        select(
            t.c["主鍵"].label("chunk_pk"),
            t.c["問卷題目檔主鍵"],
            t.c["chunk_source"],
            t.c["切塊索引"],
            chunk_dist,
        )
        .order_by(chunk_dist)
        .limit(top_k)
        .cte("chunk_hits")
    )

    # CTE 2: aggregated — GROUP BY (item, source, chunk), MIN distance, HAVING ≤ threshold, top_n
    aggregated = (
        select(
            chunk_hits.c["問卷題目檔主鍵"],
            chunk_hits.c["chunk_source"],
            chunk_hits.c["切塊索引"],
            func.min(chunk_hits.c["distance"]).label("min_distance"),
        )
        .group_by(
            chunk_hits.c["問卷題目檔主鍵"],
            chunk_hits.c["chunk_source"],
            chunk_hits.c["切塊索引"],
        )
        .having(func.min(chunk_hits.c["distance"]) <= threshold)
        .order_by(func.min(chunk_hits.c["distance"]))
        .limit(effective_top_n)
        .cte("aggregated")
    )

    # CTE 3: adj — virtual table with values (-1, 0, 1) for context expansion
    adj = union_all(
        select(literal(-1, type_=Integer).label("v")),
        select(literal(0, type_=Integer).label("v")),
        select(literal(1, type_=Integer).label("v")),
    ).cte("adj")

    # CTE 4: expanded — CROSS JOIN aggregated × adj, carries chunk_source for neighbour matching
    expanded = (
        select(
            aggregated.c["問卷題目檔主鍵"],
            aggregated.c["chunk_source"],
            (aggregated.c["切塊索引"] + adj.c["v"]).label("neighbor_idx"),
            aggregated.c["min_distance"],
        )
        .select_from(aggregated.join(adj, true()))
        .cte("expanded")
    )

    # CTE 5: expanded_chunks — JOIN back to 問卷題目切塊, same item + same chunk_source + neighbor idx
    expanded_chunks = (
        select(
            t.c["主鍵"].label("chunk_pk"),
            t.c["問卷題目檔主鍵"],
            t.c["chunk_source"],
            t.c["切塊索引"],
            t.c["chunk_text"],
            expanded.c["min_distance"],
        )
        .select_from(
            expanded.join(
                t,
                (t.c["問卷題目檔主鍵"] == expanded.c["問卷題目檔主鍵"])
                & (t.c["chunk_source"] == expanded.c["chunk_source"])
                & (t.c["切塊索引"] == expanded.c["neighbor_idx"]),
            )
        )
        .distinct()
        .cte("expanded_chunks")
    )

    # CTE 6: final_data — JOIN 問卷題目檔 to get full 題目/回覆 original text, 題目序號, 問卷主檔主鍵
    q = 問卷題目檔
    final_data = (
        select(
            expanded_chunks.c["chunk_pk"],
            expanded_chunks.c["問卷題目檔主鍵"],
            expanded_chunks.c["chunk_source"],
            expanded_chunks.c["切塊索引"],
            expanded_chunks.c["chunk_text"],
            expanded_chunks.c["min_distance"],
            q.c["題目"],
            q.c["回覆"],
            q.c["題目序號"],
            q.c["問卷主檔主鍵"],
        )
        .select_from(
            expanded_chunks.join(
                q,
                (q.c["主鍵"] == expanded_chunks.c["問卷題目檔主鍵"])
                & (
                    (q.c["是否刪除"].is_(None)) | (q.c["是否刪除"] == 0)
                ),
            )
        )
        .order_by(expanded_chunks.c["min_distance"], expanded_chunks.c["切塊索引"])
        .cte("final_data")
    )

    # Final SELECT with 問卷主檔 metadata
    m = 問卷主檔
    final_stmt = (
        select(
            final_data.c["chunk_pk"],
            final_data.c["問卷題目檔主鍵"],
            final_data.c["chunk_source"],
            final_data.c["切塊索引"],
            final_data.c["chunk_text"],
            final_data.c["min_distance"],
            final_data.c["題目"],
            final_data.c["回覆"],
            final_data.c["題目序號"],
            final_data.c["問卷主檔主鍵"],
            m.c["問卷名稱"],
            m.c["輸出國家"],
            m.c["輸出品項"],
        )
        .select_from(
            final_data.join(
                m,
                m.c["主鍵"] == final_data.c["問卷主檔主鍵"],
                isouter=True,
            )
        )
        .order_by(final_data.c["min_distance"], final_data.c["切塊索引"])
    )

    result = await session.execute(final_stmt)
    q_rows = result.mappings().all()

    elapsed_q_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "Track A 查詢完成：回傳筆數=%d  耗時=%.1f ms",
        len(q_rows), elapsed_q_ms,
    )

    # ── Track B: 知識文獻 CTE chain ──────────────────────────────────────────

    knowledge_rows: list = []
    try:
        kc = 文獻切塊檔

        # CTE K1: knowledge_hits — cosine distance on 內容向量
        knowledge_dist = kc.c["內容向量"].cosine_distance(query_vector).label("distance")
        knowledge_hits = (
            select(
                kc.c["主鍵"].label("chunk_pk"),
                kc.c["文獻節點檔主鍵"],
                kc.c["切塊索引"],
                knowledge_dist,
            )
            .order_by(knowledge_dist)
            .limit(top_k)
            .cte("knowledge_hits")
        )

        # CTE K2: knowledge_agg — GROUP BY node+chunk, MIN distance, HAVING, LIMIT
        knowledge_agg = (
            select(
                knowledge_hits.c["文獻節點檔主鍵"],
                knowledge_hits.c["切塊索引"],
                func.min(knowledge_hits.c["distance"]).label("min_distance"),
            )
            .group_by(knowledge_hits.c["文獻節點檔主鍵"], knowledge_hits.c["切塊索引"])
            .having(func.min(knowledge_hits.c["distance"]) <= threshold)
            .order_by(func.min(knowledge_hits.c["distance"]))
            .limit(effective_top_n)
            .cte("knowledge_agg")
        )

        # CTE K3: knowledge_adj — values (-1, 0, 1) for context expansion
        knowledge_adj = union_all(
            select(literal(-1, type_=Integer).label("v")),
            select(literal(0, type_=Integer).label("v")),
            select(literal(1, type_=Integer).label("v")),
        ).cte("knowledge_adj")

        # CTE K4: knowledge_expanded — CROSS JOIN knowledge_agg × knowledge_adj
        knowledge_expanded = (
            select(
                knowledge_agg.c["文獻節點檔主鍵"],
                (knowledge_agg.c["切塊索引"] + knowledge_adj.c["v"]).label("neighbor_idx"),
                knowledge_agg.c["min_distance"],
            )
            .select_from(knowledge_agg.join(knowledge_adj, true()))
            .cte("knowledge_expanded")
        )

        # CTE K5: knowledge_expanded_chunks — join back to 文獻切塊檔 for actual text
        knowledge_expanded_chunks = (
            select(
                kc.c["主鍵"].label("chunk_pk"),
                kc.c["文獻節點檔主鍵"],
                kc.c["切塊索引"],
                kc.c["切塊內容"],
                knowledge_expanded.c["min_distance"],
            )
            .select_from(
                knowledge_expanded.join(
                    kc,
                    (kc.c["文獻節點檔主鍵"] == knowledge_expanded.c["文獻節點檔主鍵"])
                    & (kc.c["切塊索引"] == knowledge_expanded.c["neighbor_idx"]),
                )
            )
            .distinct()
            .cte("knowledge_expanded_chunks")
        )

        # CTE K6: knowledge_final — JOIN 文獻節點檔 (is_deleted=0/NULL) → JOIN 知識文獻主檔
        kn = 文獻節點檔
        km = 知識文獻主檔
        knowledge_final_cte = (
            select(
                knowledge_expanded_chunks.c["chunk_pk"],
                knowledge_expanded_chunks.c["文獻節點檔主鍵"],
                knowledge_expanded_chunks.c["切塊索引"],
                knowledge_expanded_chunks.c["切塊內容"],
                knowledge_expanded_chunks.c["min_distance"],
                kn.c["節點標題路徑"],
                kn.c["文獻主檔主鍵"],
            )
            .select_from(
                knowledge_expanded_chunks.join(
                    kn,
                    (kn.c["主鍵"] == knowledge_expanded_chunks.c["文獻節點檔主鍵"])
                    & (
                        (kn.c["是否刪除"].is_(None)) | (kn.c["是否刪除"] == 0)
                    ),
                )
            )
            .order_by(
                knowledge_expanded_chunks.c["min_distance"],
                knowledge_expanded_chunks.c["切塊索引"],
            )
            .cte("knowledge_final")
        )

        # Final knowledge SELECT with 知識文獻主檔 metadata
        knowledge_stmt = (
            select(
                knowledge_final_cte.c["chunk_pk"],
                knowledge_final_cte.c["文獻節點檔主鍵"],
                knowledge_final_cte.c["切塊索引"],
                knowledge_final_cte.c["切塊內容"],
                knowledge_final_cte.c["min_distance"],
                knowledge_final_cte.c["節點標題路徑"],
                knowledge_final_cte.c["文獻主檔主鍵"],
                km.c["文獻名稱"],
                km.c["文獻類型"],
            )
            .select_from(
                knowledge_final_cte.join(
                    km,
                    km.c["主鍵"] == knowledge_final_cte.c["文獻主檔主鍵"],
                    isouter=True,
                )
            )
            .order_by(knowledge_final_cte.c["min_distance"], knowledge_final_cte.c["切塊索引"])
        )

        knowledge_result = await session.execute(knowledge_stmt)
        knowledge_rows = knowledge_result.mappings().all()

        elapsed_k_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Track B 查詢完成：回傳筆數=%d  耗時=%.1f ms",
            len(knowledge_rows), elapsed_k_ms,
        )

    except Exception as exc:
        logger.warning("Track B (知識文獻) 查詢失敗，略過：%s", exc)
        knowledge_rows = []

    # ── Global merge: mode controlled by RETRIEVAL_MERGE_MODE ───────────────
    # "independent" (default): each track keeps up to top_n independently —
    #     knowledge references are supplementary and never squeezed out by
    #     questionnaire chunks.
    # "compete": both tracks are globally sorted by min_distance and share
    #     a single top_n quota (original behaviour).

    if not q_rows and not knowledge_rows:
        logger.info("檢索結果為空：無切塊符合閾值=%.3f", threshold)
        return RetrievalResult()

    merge_mode = settings.retrieval_merge_mode.lower()

    if merge_mode == "compete":
        tagged: list[tuple[float, str, object]] = []
        for row in q_rows:
            tagged.append((row["min_distance"], "questionnaire", row))
        for row in knowledge_rows:
            tagged.append((row["min_distance"], "knowledge", row))
        tagged.sort(key=lambda x: x[0])
        tagged = tagged[:effective_top_n]
        final_q_rows = [row for _, track, row in tagged if track == "questionnaire"]
        final_k_rows = [row for _, track, row in tagged if track == "knowledge"]
    else:
        # "independent" — each track capped separately
        final_q_rows = sorted(q_rows, key=lambda r: r["min_distance"])[:effective_top_n]
        final_k_rows = sorted(knowledge_rows, key=lambda r: r["min_distance"])[:effective_top_n]

    logger.info(
        "全局合併後：問卷切塊=%d  知識文獻切塊=%d  共 %d 筆",
        len(final_q_rows), len(final_k_rows), len(final_q_rows) + len(final_k_rows),
    )

    # ── Build questionnaire context and references ──────────────────────────
    # 以問卷題目檔主鍵去重：多個切塊（question/answer 兩種 source 的擴展）可能對應同一題，
    # context 每題只取一次完整的 Q/A 原文（來自 JOIN 問卷題目檔的 題目/回覆 欄位）

    seen_items: set[str] = set()
    context_parts: list[str] = []
    for row in final_q_rows:
        item_pk = row["問卷題目檔主鍵"] or ""
        if item_pk and item_pk not in seen_items:
            seen_items.add(item_pk)
            title = row["題目"] or ""
            reply = row["回覆"] or ""
            if title or reply:
                context_parts.append(f"Q: {title}\nA: {reply}")
    raw_context = "\n\n".join(context_parts)

    seen_refs: set[tuple] = set()
    references: list[dict] = []
    for row in final_q_rows:
        key = (row["問卷主檔主鍵"], row["題目序號"])
        if key not in seen_refs:
            seen_refs.add(key)
            references.append({
                "問卷主檔主鍵": row["問卷主檔主鍵"],
                "問卷名稱": row["問卷名稱"],
                "輸出國家": row["輸出國家"],
                "輸出品項": row["輸出品項"],
                "題目序號": row["題目序號"],
            })

    logger.info("問卷檢索完成：%d 個切塊  %d 筆唯一來源", len(final_q_rows), len(references))

    # ── Build knowledge context and references ──────────────────────────────

    knowledge_parts: list[str] = []
    seen_knowledge_refs: set[tuple] = set()
    knowledge_references: list[dict] = []

    for row in final_k_rows:
        chunk_content = row["切塊內容"] or ""
        header_path = row["節點標題路徑"] or ""
        doc_name = row["文獻名稱"] or ""
        doc_type = row["文獻類型"] or ""

        if chunk_content:
            # 依文獻類型決定 XML origin 標頭
            if doc_type == "REGULATION":
                header = f"【{doc_name} {header_path}】"
            elif doc_type == "GUIDELINE":
                header = f"【{doc_name} - 作業節點：{header_path}】"
            elif doc_type == "QA":
                header = f"【相關問答：{header_path}】"
            else:
                header = f"【{doc_name} {header_path}】"
            knowledge_parts.append(
                f'<Context origin="{header}">\n{chunk_content}\n</Context>'
            )

        key = (row["文獻主檔主鍵"], row["節點標題路徑"])
        if key not in seen_knowledge_refs:
            seen_knowledge_refs.add(key)
            knowledge_references.append({
                "文獻主檔主鍵": row["文獻主檔主鍵"],
                "文獻名稱": doc_name,
                "文獻類型": doc_type,
                "節點標題路徑": header_path,
            })

    knowledge_context = "\n\n".join(knowledge_parts)
    logger.info(
        "知識文獻檢索完成：%d 個切塊  %d 筆唯一節點",
        len(final_k_rows), len(knowledge_references),
    )

    # ── Fetch attachment URLs ───────────────────────────────────────────────

    master_pks = list({row["問卷主檔主鍵"] for row in final_q_rows if row["問卷主檔主鍵"]})
    attachment_urls: list[str] = []
    if master_pks:
        logger.debug("查詢 %d 份問卷附件 ...", len(master_pks))
        att = 問卷附件檔
        att_stmt = (
            select(att.c["檔案路徑"])
            .where(att.c["問卷主檔主鍵"].in_(master_pks))
            .where((att.c["是否刪除"].is_(None)) | (att.c["是否刪除"] == 0))
            .where(att.c["檔案路徑"].isnot(None))
        )
        att_result = await session.execute(att_stmt)
        attachment_urls = [r[0] for r in att_result.all() if r[0]]
        logger.info("附件查詢完成：共 %d 筆", len(attachment_urls))

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("整體檢索完成：總耗時=%.1f ms", elapsed_ms)

    return RetrievalResult(
        chunks=list(final_q_rows),
        raw_context=raw_context,
        references=references,
        attachment_urls=attachment_urls,
        knowledge_context=knowledge_context,
        knowledge_references=knowledge_references,
    )