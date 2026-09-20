import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import (
    ChunkingStrategyComparator,
    FixedSizeChunker,
    RecursiveChunker,
    SentenceChunker,
    compute_similarity,
)
from src.embeddings import MockEmbedder, _mock_embed
from src.models import Document
from src.store import EmbeddingStore

# 1. Baseline analysis on 2 documents
comp = ChunkingStrategyComparator()
doc_paths = [
    Path('data/ecommerce/shopee-buyer-return-refund.md'),
    Path('data/ecommerce/shopee-seller-dispute-resolution.md'),
]

print("=== 1. BASELINE CHUNKING COMPARISON ===")
for p in doc_paths:
    body = p.read_text(encoding='utf-8').split('---')[2].strip()
    res = comp.compare(body, chunk_size=300)
    print(f"\nDocument: {p.name}")
    for k, v in res.items():
        print(f"  {k:15}: count={v['count']:2}, avg_len={v['avg_length']:.1f}")

# 2. Similarity predictions test on 5 sentence pairs
print("\n=== 2. SIMILARITY PREDICTION EXPERIMENT ===")
embedder = MockEmbedder()
pairs = [
    (
        "Người mua có quyền yêu cầu trả hàng trong vòng 15 ngày.",
        "Khách hàng được phép khiếu nại đổi trả sản phẩm trong hai tuần.",
        "cao",
    ),
    (
        "Người bán có 48 giờ để phản hồi yêu cầu hoàn tiền.",
        "Thời hạn shop phản hồi khiếu nại trả hàng là 2 ngày lịch.",
        "cao",
    ),
    (
        "Sản phẩm bảo hành chính hãng từ 7 đến 14 ngày làm việc.",
        "Thời gian trung tâm xử lý bảo hành thiết bị kéo dài từ một đến hai tuần.",
        "cao",
    ),
    (
        "Quy cách đóng gói hàng dễ vỡ bắt buộc quấn 3 lớp bọt khí.",
        "Nghiêm cấm người bán đăng tải sản phẩm vũ khí quân dụng và chất nổ.",
        "thấp",
    ),
    (
        "Mức bồi thường tối đa 100% giá trị đơn hàng có bảo hiểm.",
        "Hướng dẫn người mua lập tài khoản cá nhân trên sàn thương mại.",
        "thấp",
    ),
]

for i, (s1, s2, pred) in enumerate(pairs, 1):
    v1 = embedder(s1)
    v2 = embedder(s2)
    sim = compute_similarity(v1, v2)
    print(f"Pair {i}: Pred={pred:4} | Sim={sim:+.4f} | S1='{s1[:35]}...' | S2='{s2[:35]}...'")

# 3. Benchmark queries run on Ecommerce docs
print("\n=== 3. BENCHMARK QUERIES ON STORE ===")
store = EmbeddingStore(collection_name="ecommerce_benchmark", embedding_fn=_mock_embed)
all_docs = []
for p in sorted(Path('data/ecommerce').glob('*.md')):
    raw = p.read_text(encoding='utf-8')
    parts = raw.split('---')
    fm = dict(re.findall(r'^(\w+):\s*(.+)$', parts[1], re.M))
    body = parts[2].strip()
    # Let's chunk by heading or use RecursiveChunker
    chunker = RecursiveChunker(chunk_size=400)
    chunks = chunker.chunk(body)
    for idx, c in enumerate(chunks):
        meta = dict(fm)
        meta['chunk_idx'] = idx
        all_docs.append(Document(id=f"{fm['doc_id']}_c{idx}", content=c, metadata=meta))

store.add_documents(all_docs)
print(f"Total stored chunks: {store.get_collection_size()}")

queries = [
    ("Thời hạn tối đa để người mua gửi yêu cầu trả hàng và hoàn tiền Shopee Mall là bao lâu?", None),
    ("Người bán có bao nhiêu thời gian để phản hồi khi người mua yêu cầu trả hàng?", {"audience": "seller"}),
    ("Thời gian xử lý bảo hành tiêu chuẩn sản phẩm tại Tiki là bao nhiêu ngày?", None),
    ("Quy chuẩn đóng gói thùng carton mấy lớp cho hàng nặng trên 5kg trên Lazada?", {"audience": "seller"}),
    ("Mức bồi thường tổn thất cho đơn hàng vận chuyển không có bảo hiểm là bao nhiêu?", None),
]

for q, f in queries:
    print(f"\nQuery: {q}")
    print(f"Filter: {f}")
    if f:
        hits = store.search_with_filter(q, top_k=3, metadata_filter=f)
    else:
        hits = store.search(q, top_k=3)
    for rank, h in enumerate(hits, 1):
        print(f"  Top-{rank} [score={h['score']:+.3f}] doc={h['metadata'].get('doc_id')}: {h['content'][:80]}...")
