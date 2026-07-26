"""Build the RAG corpus from TQuAD.

Three things happen here, and the second one is the non-obvious one:

1. Paragraph -> chunk. TQuAD is SQuAD-format, so every question already points
   at exactly one paragraph. Using the paragraph as the chunk means the
   retrieval ground truth comes for free -- no hand labelling.

2. Every chunk is padded to a multiple of BLOCK_SIZE tokens. vLLM hashes its
   KV cache in fixed-size token blocks; if chunk boundaries do not line up with
   block boundaries, a shared chunk still produces a different block hash and
   the prefix chain breaks anyway. Padding costs a few wasted tokens per chunk
   and is what makes canonical chunk ordering actually pay off.

3. Chunks are embedded on CPU. One forward pass per chunk, no autoregression,
   so this is slow-but-bounded: it runs once, offline, and never touches the
   request path.

Usage:
    python build_corpus.py --tquad train-v0.1.json --out ./corpus
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

# Sentence-ish boundaries. Deliberately conservative: splitting mid-sentence
# would hurt retrieval more than an oversized chunk does.
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    chunk_id: int
    title: str
    title_idx: int
    para_idx: int
    sub_idx: int
    text: str
    text_padded: str
    n_tokens: int
    n_tokens_padded: int
    char_start: int   # offset of this sub-chunk within its source paragraph
    char_end: int


def make_chunk_id(title_idx: int, para_idx: int, sub_idx: int) -> int:
    """Deterministic and order-preserving.

    Canonical ordering sorts by this id, so it must be stable across runs --
    otherwise the same two queries produce different prefixes on different days
    and the cache results are not comparable.
    """
    return title_idx * 100_000 + para_idx * 100 + sub_idx


def split_paragraph(text: str, tok, max_tokens: int) -> list[tuple[str, int, int]]:
    """Split an over-long paragraph at sentence boundaries.

    Returns (text, char_start, char_end) triples. TQuAD's longest paragraph is
    ~11k characters, far past any sane chunk size, so this is not optional.
    """
    if len(tok.encode(text)) <= max_tokens:
        return [(text, 0, len(text))]

    pieces: list[tuple[str, int, int]] = []
    buf, buf_start, cursor = "", 0, 0

    for sent in _SENT_RE.split(text):
        if not sent:
            continue
        start = text.find(sent, cursor)
        if start < 0:
            start = cursor
        end = start + len(sent)
        cursor = end

        candidate = (buf + " " + sent).strip() if buf else sent
        if buf and len(tok.encode(candidate)) > max_tokens:
            pieces.append((buf, buf_start, start))
            buf, buf_start = sent, start
        else:
            if not buf:
                buf_start = start
            buf = candidate

    if buf:
        pieces.append((buf, buf_start, len(text)))
    return pieces


def pad_to_block(text: str, tok, block_size: int) -> tuple[str, int, int]:
    """Pad text so its token count is a multiple of block_size.

    Works in token space rather than by appending characters: appending
    whitespace is unreliable because tokenizers merge runs of whitespace, so
    the token count can stop moving while the string keeps growing.
    """
    ids = tok.encode(text, add_special_tokens=False)
    n = len(ids)
    remainder = n % block_size
    if remainder == 0:
        return text, n, n

    pad_needed = block_size - remainder
    pad_id = tok.encode(" .", add_special_tokens=False)[-1]
    padded_ids = ids + [pad_id] * pad_needed
    padded_text = tok.decode(padded_ids)

    # decode/encode does not always round-trip exactly; verify rather than trust
    actual = len(tok.encode(padded_text, add_special_tokens=False))
    return padded_text, n, actual


def build(args) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.serving_model)
    raw = json.loads(Path(args.tquad).read_text(encoding="utf-8"))["data"]

    chunks: list[Chunk] = []
    qa_rows: list[dict] = []
    misaligned = 0

    for title_idx, article in enumerate(raw):
        title = article["title"]
        for para_idx, para in enumerate(article["paragraphs"]):
            context = para["context"]
            pieces = split_paragraph(context, tok, args.max_chunk_tokens)

            local: list[Chunk] = []
            for sub_idx, (piece, c0, c1) in enumerate(pieces):
                padded, n_tok, n_pad = pad_to_block(piece, tok, args.block_size)
                if n_pad % args.block_size != 0:
                    misaligned += 1
                ch = Chunk(
                    chunk_id=make_chunk_id(title_idx, para_idx, sub_idx),
                    title=title,
                    title_idx=title_idx,
                    para_idx=para_idx,
                    sub_idx=sub_idx,
                    text=piece,
                    text_padded=padded,
                    n_tokens=n_tok,
                    n_tokens_padded=n_pad,
                    char_start=c0,
                    char_end=c1,
                )
                local.append(ch)
                chunks.append(ch)

            # Map each question onto the sub-chunk that actually contains its
            # answer span. This is the retrieval ground truth -- it is written
            # to disk for analysis only and must never reach the router.
            for qa in para.get("qas", []):
                answers = qa.get("answers") or []
                start = int(answers[0]["answer_start"]) if answers else 0
                owner = local[0]
                for ch in local:
                    if ch.char_start <= start < ch.char_end:
                        owner = ch
                        break
                qa_rows.append({
                    "qa_id": qa["id"],
                    "question": qa["question"],
                    "answer": answers[0]["text"] if answers else None,
                    "chunk_id": owner.chunk_id,
                    "title_idx": title_idx,
                    "title": title,
                })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(asdict(ch), ensure_ascii=False) + "\n")
    with (out / "qa.jsonl").open("w", encoding="utf-8") as f:
        for row in qa_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total_padded = sum(c.n_tokens_padded for c in chunks)
    total_raw = sum(c.n_tokens for c in chunks)
    print(f"chunks           : {len(chunks)}")
    print(f"questions        : {len(qa_rows)}")
    print(f"corpus tokens    : {total_padded} (padding overhead "
          f"{100 * (total_padded - total_raw) / max(total_raw, 1):.1f}%)")
    print(f"misaligned chunks: {misaligned}")
    if misaligned:
        print("  ^ decode/encode did not round-trip for these; inspect before trusting cache numbers")

    if args.skip_embed:
        print("embedding skipped (--skip-embed)")
        return

    embed(chunks, out, args)


def embed(chunks: list[Chunk], out: Path, args) -> None:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    # CPU is fine here: one forward pass per chunk, no autoregressive loop, and
    # the GPU is already fully committed to vLLM.
    model = SentenceTransformer(args.embed_model, device=args.device)

    # e5-family models require these prefixes; omitting them measurably degrades
    # retrieval. Queries must use "query: " at trace time.
    passages = [f"passage: {c.text}" for c in chunks]
    vecs = model.encode(
        passages,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # normalised -> cosine similarity is a dot product
        convert_to_numpy=True,
    ).astype("float32")

    np.save(out / "embeddings.npy", vecs)
    (out / "embed_meta.json").write_text(json.dumps({
        "model": args.embed_model,
        "dim": int(vecs.shape[1]),
        "count": int(vecs.shape[0]),
        "normalized": True,
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    }, indent=2))
    print(f"embeddings       : {vecs.shape} -> {out / 'embeddings.npy'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tquad", required=True, help="path to train-v0.1.json")
    p.add_argument("--out", default="./corpus")
    p.add_argument("--serving-model", default="Qwen/Qwen2.5-7B-Instruct",
                   help="tokenizer used for block alignment -- must match the served model")
    p.add_argument("--embed-model", default="intfloat/multilingual-e5-base")
    p.add_argument("--block-size", type=int, default=16, help="vLLM KV cache block size")
    p.add_argument("--max-chunk-tokens", type=int, default=384)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--skip-embed", action="store_true")
    build(p.parse_args())


if __name__ == "__main__":
    main()
