import argparse
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.config import ELASTICSEARCH_CHUNK_INDEX
from common.doc_store.es_conn_pool import ES_CONN
from run_pdf_to_md_chunked import LLM_MODEL, LLM_URL

DEFAULT_KB_ID = "2d2736b2e8f64bd094ee4d3eed76b146"
DEFAULT_OUTPUT = ROOT_DIR / "pdf" / f"rag_eval_kb_{DEFAULT_KB_ID}_200.jsonl"


def fetch_chunks(kb_id: str, size: int, seed: int) -> List[Dict[str, Any]]:
    es = ES_CONN.get_conn()
    doc_body = {
        "size": 0,
        "query": {
            "bool": {
                "filter": [
                    {"term": {"kb_id": kb_id}},
                    {"exists": {"field": "chunk_text"}},
                ]
            }
        },
        "aggs": {
            "docs": {
                "terms": {
                    "field": "doc_id",
                    "size": 10000,
                    "order": {"_key": "asc"},
                }
            }
        },
    }
    doc_res = es.search(index=ELASTICSEARCH_CHUNK_INDEX, body=doc_body)
    buckets = doc_res.get("aggregations", {}).get("docs", {}).get("buckets", [])
    doc_ids = [bucket["key"] for bucket in buckets]
    if not doc_ids:
        return []

    rng = random.Random(seed)
    rng.shuffle(doc_ids)

    base_quota = max(size // len(doc_ids), 1)
    remainder = size % len(doc_ids)
    chunks = []
    seen_es_ids = set()

    def _append_hits(hits: List[Dict[str, Any]]) -> None:
        for hit in hits:
            source = hit.get("_source", {}) or {}
            text = (source.get("chunk_text") or "").strip()
            es_id = hit.get("_id")
            if not text or es_id in seen_es_ids:
                continue
            source["_es_id"] = es_id
            chunks.append(source)
            seen_es_ids.add(es_id)

    for idx, doc_id in enumerate(doc_ids):
        quota = base_quota + (1 if idx < remainder else 0)
        body = {
            "size": quota,
            "query": {
                "function_score": {
                    "query": {
                        "bool": {
                            "filter": [
                                {"term": {"kb_id": kb_id}},
                                {"term": {"doc_id": doc_id}},
                                {"exists": {"field": "chunk_text"}},
                            ]
                        }
                    },
                    "random_score": {"seed": seed + idx, "field": "_seq_no"},
                }
            },
            "_source": {"excludes": ["embedding"]},
        }
        res = es.search(index=ELASTICSEARCH_CHUNK_INDEX, body=body)
        _append_hits(res.get("hits", {}).get("hits", []))
        if len(chunks) >= size:
            return chunks[:size]

    if len(chunks) >= size:
        return chunks[:size]

    body = {
        "size": size - len(chunks),
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"kb_id": kb_id}},
                            {"exists": {"field": "chunk_text"}},
                        ]
                    }
                },
                "random_score": {"seed": seed, "field": "_seq_no"},
            }
        },
        "_source": {"excludes": ["embedding"]},
    }
    res = es.search(index=ELASTICSEARCH_CHUNK_INDEX, body=body)
    _append_hits(res.get("hits", {}).get("hits", []))
    return chunks[:size]


def call_llm(prompt: str, max_tokens: int = 1024, retries: int = 2) -> str:
    headers = {"Content-Type": "application/json", "Authorization": "Bearer "}
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
    }
    last_error = None
    for _ in range(retries + 1):
        try:
            resp = requests.post(LLM_URL, headers=headers, json=payload, timeout=900)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"LLM request failed: {last_error}")


def parse_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def fallback_qa_from_text(text: str) -> Dict[str, Any]:
    question = ""
    answer = ""
    question_match = re.search(r'"?question"?\s*[:：]\s*"?(.+?)(?:"?\s*[,，]\s*"?answer"?\s*[:：]|$)', text, re.DOTALL)
    answer_match = re.search(r'"?answer"?\s*[:：]\s*"?(.+?)"?\s*}?$', text, re.DOTALL)
    if question_match:
        question = question_match.group(1).strip().strip('",，')
    if answer_match:
        answer = answer_match.group(1).strip().strip('",，')
    return {"question": question, "answer": answer}


def build_prompt(chunk: Dict[str, Any]) -> str:
    text = (chunk.get("chunk_text") or "").strip()
    title_parts = [
        chunk.get("doc_title") or "",
        chunk.get("chapter_title") or "",
        chunk.get("section_title") or "",
    ]
    title = " / ".join([part for part in title_parts if part])
    return f"""你需要根据给定 chunk 构造一条 RAG 评测测试集样本。

要求：
1. question 必须能够仅根据该 chunk 回答。
2. answer 必须忠实依据 chunk，不要引入外部知识。
3. question 尽量具体，避免泛泛提问。
4. 只返回 JSON 对象，不要 Markdown，不要解释。
5. JSON 格式固定为：{{"question":"...","answer":"..."}}

chunk 标题：{title}

chunk 内容：
{text[:6000]}
"""


def source_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "es_id": chunk.get("_es_id"),
        "chunk_id": chunk.get("chunk_id"),
        "kb_id": chunk.get("kb_id"),
        "doc_id": chunk.get("doc_id"),
        "fd_id": chunk.get("fd_id"),
        "chapter_id": chunk.get("chapter_id"),
        "doc_title": chunk.get("doc_title"),
        "chapter_title": chunk.get("chapter_title"),
        "section_title": chunk.get("section_title"),
        "page_num_int": chunk.get("page_num_int"),
        "chunk_text": chunk.get("chunk_text"),
    }


def build_record(chunk: Dict[str, Any], index: int, generate_qa: bool) -> Dict[str, Any]:
    context = (chunk.get("chunk_text") or "").strip()
    if generate_qa:
        response_text = call_llm(build_prompt(chunk))
        try:
            qa = parse_json_object(response_text)
        except Exception:
            qa = fallback_qa_from_text(response_text)
        question = (qa.get("question") or "").strip()
        answer = (qa.get("answer") or "").strip()
        if not question or not answer:
            title_parts = [
                chunk.get("doc_title") or "",
                chunk.get("chapter_title") or "",
                chunk.get("section_title") or "",
            ]
            title = " / ".join([part for part in title_parts if part]) or "该片段"
            question = question or f"请根据材料概括“{title}”这一 chunk 的主要内容。"
            answer = answer or context
    else:
        title_parts = [
            chunk.get("doc_title") or "",
            chunk.get("chapter_title") or "",
            chunk.get("section_title") or "",
        ]
        title = " / ".join([part for part in title_parts if part]) or "该片段"
        question = f"请根据材料概括“{title}”这一 chunk 的主要内容。"
        answer = context
    return {
        "id": f"kb2d2736b2_sample_{index:04d}",
        "question": question,
        "answer": answer,
        "ground_truth": answer,
        "contexts": [context],
        "source_chunk": source_chunk(chunk),
    }


def write_jsonl(records: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(records: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(),
        "count": len(records),
        "data": records,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb-id", default=DEFAULT_KB_ID)
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--format", choices=["jsonl", "json"], default="jsonl")
    parser.add_argument("--no-qa", action="store_true")
    args = parser.parse_args()

    chunks = fetch_chunks(args.kb_id, args.size, args.seed)
    if len(chunks) < args.size:
        print(f"[WARN] only fetched {len(chunks)} chunks for kb_id={args.kb_id}")

    random.Random(args.seed).shuffle(chunks)
    selected = chunks[:args.size]
    records = []
    for idx, chunk in enumerate(selected, 1):
        print(f"[{idx}/{len(selected)}] building sample from chunk_id={chunk.get('chunk_id')}", flush=True)
        records.append(build_record(chunk, idx, generate_qa=not args.no_qa))

    output_path = Path(args.output)
    if args.format == "jsonl":
        write_jsonl(records, output_path)
    else:
        write_json(records, output_path)
    print(f"[DONE] wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
