import csv
import re
import sys
from pathlib import Path

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

D = Path('data/ecommerce')
REQ = ['doc_id', 'title', 'source_url', 'retrieved_at', 'document_version', 'audience']
mds = sorted(D.glob('*.md'))
rows = list(csv.DictReader(open(D / 'sources.csv', encoding='utf-8')))
ids, auds = [], {}

print("=" * 60)
print("KIỂM TRA CHECKPOINT 2 — METADATA VÀ DATA INVENTORY")
print("=" * 60)

for p in mds:
    content = p.read_text(encoding='utf-8')
    parts = content.split('---')
    if len(parts) >= 3:
        fm = dict(re.findall(r'^(\w+):\s*(.+)$', parts[1], re.M))
    else:
        fm = {}
    doc_id = fm.get('doc_id', '')
    ids.append(doc_id)
    aud = fm.get('audience', '')
    auds[aud] = auds.get(aud, 0) + 1
    
    status = "OK" if all(k in fm for k in REQ) and doc_id == p.stem else "THIEU METADATA"
    print(f"{p.name:45} {status}")

print("-" * 60)
print(f"so file : {len(mds)} (can 5-10) -> {'DAT' if 5 <= len(mds) <= 10 else 'CHUA DAT'}")
is_khop = sorted(r['doc_id'] for r in rows) == sorted(ids)
print(f"csv     : {'khop' if is_khop else 'LECH'}")
print(f"audience: {auds} -> {'DAT (>= 2 gia tri)' if len(auds) >= 2 else 'CHUA DAT'}")
print("=" * 60)

if 5 <= len(mds) <= 10 and is_khop and len(auds) >= 2:
    print(">>> CHÚC MỪNG: BỘ DỮ LIỆU ĐẠT CHUẨN 100% CHECKPOINT 2! <<<")
else:
    print(">>> CẦN KIỂM TRA LẠI CÁC MỤC CHƯA ĐẠT TRÊN <<<")
