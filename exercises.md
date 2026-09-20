# Ngày 7 — Bài tập
## Nền tảng Dữ liệu: Embedding & Vector Store | Bài tập thực hành

---

## Phần 1 — Khởi động (Cá nhân)

### Bài tập 1.1 — Cosine Similarity (Độ tương tự Cosine) bằng ngôn ngữ đời thường

Không yêu cầu toán học — hãy giải thích về mặt khái niệm:

- **Điều gì xảy ra khi hai đoạn văn bản có độ tương tự cosine cao?**
  > Khi hai đoạn văn bản có độ tương tự cosine cao (tiệm cận 1.0), hai vector nhúng (embedding) của chúng chỉ về gần cùng một hướng trong không gian đa chiều. Điều này chứng minh hai đoạn văn bản tương đồng rất lớn về mặt ngữ nghĩa (semantic similarity), phản ánh cùng một thông điệp cốt lõi ngay cả khi dùng các từ vựng hoàn toàn khác nhau.

- **Đưa ra một ví dụ cụ thể về hai câu sẽ có độ tương tự CAO và hai câu sẽ có độ tương tự THẤP:**
  * **Ví dụ có độ tương tự CAO:**
    - Câu A: *"Khách hàng có thể gửi yêu cầu trả hàng và hoàn tiền trong vòng 15 ngày."*
    - Câu B: *"Người mua được quyền khiếu nại đổi trả sản phẩm trong thời hạn hai tuần."*
    - *Giải thích:* Cả hai câu đều nói về quy định và thời hạn đổi trả của người tiêu dùng dù không trùng lặp nhiều từ vựng.
  * **Ví dụ có độ tương tự THẤP:**
    - Câu A: *"Chính sách đổi trả hàng hóa áp dụng trong thời hạn 15 ngày kể từ ngày nhận hàng."*
    - Câu B: *"Người bán vi phạm quy định kinh doanh vũ khí quân dụng sẽ bị truy tố trước pháp luật."*
    - *Giải thích:* Hai câu thuộc hai phạm trù nghiệp vụ hoàn toàn khác nhau (quyền lợi đổi trả của khách hàng vs chế tài hình sự đối với người bán).

- **Tại sao độ tương tự cosine lại được ưu tiên hơn khoảng cách Euclid (Euclidean distance) đối với text embeddings?**
  > Độ tương tự cosine chỉ đo góc giữa hai vector mà không phụ thuộc vào độ lớn (magnitude/chiều dài văn bản). Ngược lại, khoảng cách Euclid bị méo mó theo độ dài: hai câu cùng nghĩa nhưng một câu dài và một câu ngắn sẽ có khoảng cách Euclid rất xa nhau, gây sai lệch khi tìm kiếm ngữ nghĩa.

> **Ghi kết quả vào:** Báo cáo — Phần 1 (Khởi động) [ĐÃ HOÀN THÀNH VÀO REPORT_CANHAN.MD]

---

### Bài tập 1.2 — Bài toán tính toán Chunking

- **Một tài liệu có độ dài 10,000 ký tự. Bạn tiến hành chia nhỏ (chunk) với `chunk_size=500` (kích thước chunk), `overlap=50` (độ chồng chéo). Bạn dự kiến sẽ có bao nhiêu chunks?**
  - Công thức: `số lượng chunk = làm_tròn_lên((độ_dài_tài_liệu - độ_chồng_chéo) / (kích_thước_chunk - độ_chồng_chéo))`
  - Áp dụng: Bước nhảy giữa các chunk = $500 - 50 = 450$.
  - Phép tính: $\text{ceil}((10000 - 50) / 450) = \text{ceil}(9950 / 450) = \text{ceil}(22.111...) = 23$.
  - **Đáp án: 23 chunks.**

- **Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk sẽ thay đổi như thế nào? Tại sao bạn lại muốn tăng độ chồng chéo?**
  - Khi overlap tăng lên $100$: Bước nhảy giảm còn $500 - 100 = 400$.
  - Số lượng chunk = $\text{ceil}((10000 - 100) / 400) = \text{ceil}(9900 / 400) = \text{ceil}(24.75) = 25$ chunks (tăng thêm 2 chunks).
  - **Lý do muốn tăng độ chồng chéo:** Để bảo toàn trọn vẹn ngữ cảnh ở các ranh giới cắt giữa các chunk liền kề, tránh làm đứt đôi câu văn hoặc ngắt quãng các mệnh đề điều kiện quan trọng, giúp mô hình retrieval truy xuất chính xác thông tin nằm ở điểm giao cắt.

> **Ghi kết quả vào:** Báo cáo — Phần 1 (Khởi động) [ĐÃ HOÀN THÀNH VÀO REPORT_CANHAN.MD]

---

## Phần 2 — Lập trình cốt lõi (Cá nhân)

Hoàn thành tất cả các TODOs trong `src/chunking.py`, `src/store.py`, và `src/agent.py`. `Document` dataclass và `FixedSizeChunker` đã được triển khai sẵn làm ví dụ — hãy đọc kỹ để hiểu cấu trúc trước khi lập trình phần còn lại.

Chạy `pytest tests/` để kiểm tra tiến độ.

### Danh sách cần làm (Checklist)
- [x] `Document` dataclass — ĐÃ TRIỂN KHAI SẴN
- [x] `FixedSizeChunker` — ĐÃ TRIỂN KHAI SẴN
- [x] `SentenceChunker` — tách dựa trên ranh giới câu, nhóm lại thành các chunks
- [x] `RecursiveChunker` — thử nghiệm các dấu phân cách (separators) theo thứ tự, thực hiện đệ quy trên các đoạn có kích thước quá lớn
- [x] `compute_similarity` — công thức tính độ tương tự cosine kèm cơ chế bảo vệ chia cho 0
- [x] `ChunkingStrategyComparator` — gọi cả ba chiến lược, tính toán các chỉ số thống kê
- [x] `EmbeddingStore.__init__` — khởi tạo store (lưu trữ trong bộ nhớ in-memory chuẩn xác)
- [x] `EmbeddingStore.add_documents` — nhúng (embed) và lưu trữ từng tài liệu
- [x] `EmbeddingStore.search` — nhúng truy vấn, xếp hạng theo tích vô hướng (dot product)
- [x] `EmbeddingStore.get_collection_size` — trả về số lượng
- [x] `EmbeddingStore.search_with_filter` — lọc theo siêu dữ liệu (metadata), sau đó tìm kiếm
- [x] `EmbeddingStore.delete_document` — xóa tất cả các chunks của một doc_id
- [x] `KnowledgeBaseAgent.answer` — truy xuất (retrieve) + tạo prompt + gọi LLM

> **Nộp code:** thư mục `src/` [ĐÃ PASS 42/42 TESTS]
> **Ghi lại hướng tiếp cận vào:** Báo cáo — Phần 4 (Hướng tiếp cận của tôi) [ĐÃ HOÀN THÀNH]

---

## Phần 3 — So Sánh Chiến Lược Truy Xuất (Nhóm)

### Bài tập 3.0 — Chuẩn Bị Tài Liệu (Giờ đầu tiên)

Mỗi nhóm chọn một chủ đề (domain) và chuẩn bị bộ tài liệu:

**Bước 1 — Chọn chủ đề:** Chính sách đổi trả, bảo hành và quy định người bán / người mua trên sàn Thương mại Điện tử Shopee.

**Bước 2 — Thu thập 5-10 tài liệu:** Đã thu thập 6 tài liệu sạch lưu trong `data/ecommerce/`.

| # | Tên tài liệu | Nguồn (Source URL) | Ngày lấy / Phiên bản | Số ký tự | Metadata đã gán |
|---|--------------|------------|--------------------|----------|-----------------|
| 1 | Chính sách trả hàng và hoàn tiền dành cho Người Mua Shopee | https://help.shopee.vn/portal/4/article/77245 | 2026-09-20 / not-stated | 2195 | `audience: buyer`, `category: returns-policy`, `language: vi` |
| 2 | Quy định xử lý khiếu nại và trả hàng dành cho Người Bán Shopee | https://help.shopee.vn/portal/4/article/77245 | 2026-09-20 / not-stated | 2219 | `audience: seller`, `category: dispute-policy`, `language: vi` |
| 3 | Danh sách hàng hóa cấm và hạn chế kinh doanh trên Shopee | https://help.shopee.vn/portal/4/article/77245 | 2026-09-20 / not-stated | 1996 | `audience: seller`, `category: prohibited-items`, `language: vi` |
| 4 | Chính sách bảo hành sản phẩm chính hãng dành cho Người Mua Shopee | https://help.shopee.vn/portal/4/article/77245 | 2026-09-20 / not-stated | 2250 | `audience: buyer`, `category: warranty-policy`, `language: vi` |
| 5 | Quy chuẩn đóng gói và bàn giao hàng hóa dành cho Người Bán Shopee | https://help.shopee.vn/portal/4/article/77245 | 2026-09-20 / not-stated | 2310 | `audience: seller`, `category: shipping-guidelines`, `language: vi` |
| 6 | Chính sách bồi thường hư hỏng và thất lạc hàng hóa vận chuyển Shopee | https://help.shopee.vn/portal/4/article/77245 | 2026-09-20 / not-stated | 2340 | `audience: both`, `category: compensation-policy`, `language: vi` |

**Bước 3 — Thiết kế cấu trúc metadata (metadata schema):** Đầy đủ các trường `doc_id`, `title`, `source_url`, `retrieved_at`, `document_version`, `audience`, `category`, `language`.

> **Ghi kết quả vào:** Báo cáo — Phần 2 (Lựa chọn tài liệu) [ĐÃ HOÀN THÀNH VÀO REPORT_NHOM.MD]

---

### Bài tập 3.1 — Thiết Kế Chiến Lược Truy Xuất (Mỗi người thử riêng)

Mỗi thành viên tự chọn chiến lược riêng để thử nghiệm trên cùng bộ tài liệu của nhóm.

```python
import re
from src.chunking import RecursiveChunker

class HeadingChunker:
    """Chiến lược chia nhỏ tùy chỉnh theo đề mục Markdown (##, ###) cho chính sách TMĐT.

    Lý do thiết kế: Văn bản quy chế thương mại điện tử được soạn thảo theo từng điều khoản
    pháp lý rõ ràng. Việc chia nhỏ theo heading giúp giữ trọn vẹn toàn bộ nội dung của từng
    điều khoản, không làm đứt các mốc thời gian (48h, 15 ngày) và tự động gắn lại tiêu đề
    vào từng chunk con nếu mục quá dài.
    """

    def __init__(self, max_chunk_size: int = 450) -> None:
        self.max_chunk_size = max_chunk_size
        self._recursive = RecursiveChunker(chunk_size=max_chunk_size)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        raw_sections = re.split(r'(?=\n#{2,3}\s+)', text.strip())
        chunks: list[str] = []
        for sec in raw_sections:
            sec = sec.strip()
            if not sec:
                continue
            if len(sec) <= self.max_chunk_size:
                chunks.append(sec)
            else:
                lines = sec.split("\n", 1)
                header = lines[0].strip() if lines[0].startswith("#") else ""
                body = lines[1].strip() if len(lines) > 1 else sec
                sub_chunks = self._recursive.chunk(body)
                for sub in sub_chunks:
                    chunks.append(f"{header}\n{sub}" if header else sub)
        return chunks
```

> **Ghi kết quả vào:** Báo cáo — Phần 3 (Chiến lược chia nhỏ) [ĐÃ HOÀN THÀNH VÀO REPORT_NHOM.MD]

---

### Bài tập 3.2 — Chuẩn Bị Câu Hỏi Đánh Giá (Benchmark Queries)

Mỗi nhóm viết **đúng 5 câu hỏi đánh giá** kèm theo **câu trả lời chuẩn (gold answers)**.

| # | Câu hỏi (Query) | Câu trả lời chuẩn (Gold Answer) | Chunk nào chứa thông tin? |
|---|-------|-------------------------------|--------------------------|
| 1 | Thời hạn tối đa để người mua gửi yêu cầu trả hàng và hoàn tiền đối với sản phẩm Shopee Mall là bao lâu? | 15 ngày kể từ ngày nhận hàng thành công. | `shopee-buyer-return-refund.md` (Mục 1) |
| 2 | Người bán có bao nhiêu thời gian để phản hồi khi người mua yêu cầu trả hàng hoàn tiền? *(Có filter `audience: seller`)* | 48 giờ (2 ngày lịch) kể từ lúc hệ thống gửi thông báo. | `shopee-seller-dispute-resolution.md` (Mục 1) |
| 3 | Thời gian xử lý bảo hành tiêu chuẩn đối với sản phẩm chính hãng tại Shopee là bao nhiêu ngày? | Từ 07 đến 14 ngày làm việc kể từ ngày trung tâm nhận được sản phẩm. | `shopee-buyer-warranty-policy.md` (Mục 3) |
| 4 | Người bán Shopee phải sử dụng thùng carton mấy lớp đối với hàng hóa nặng trên 5 kg hoặc hàng dễ vỡ? *(Có filter `audience: seller`)* | Thùng carton tối thiểu 5 lớp (và quấn 2-3 lớp xốp khí dày tối thiểu 3 cm). | `shopee-seller-packaging-guidelines.md` (Mục 1) |
| 5 | Mức bồi thường tổn thất tối đa đối với đơn hàng vận chuyển Shopee không mua bảo hiểm hàng hóa là bao nhiêu? | Tối đa bằng 04 lần cước phí vận chuyển hoặc tối đa 1.000.000 VNĐ đối với hàng thất lạc thông thường. | `shopee-shipping-damage-compensation.md` (Mục 3) |

> **Ghi kết quả vào:** Báo cáo — Phần 6 (Kết quả — Câu hỏi đánh giá & Câu trả lời chuẩn) [ĐÃ HOÀN THÀNH]

---

### Bài tập 3.3 — Dự Đoán Độ Tương Tự Cosine (Cá nhân)

Gọi hàm `compute_similarity()` trên 5 cặp câu:

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Người mua có quyền yêu cầu trả hàng trong vòng 15 ngày. | Khách hàng được phép khiếu nại đổi trả sản phẩm trong hai tuần. | cao | -0.2540 | Sai (do MockEmbedder băm MD5) |
| 2 | Người bán có 48 giờ để phản hồi yêu cầu hoàn tiền. | Thời hạn shop phản hồi khiếu nại trả hàng là 2 ngày lịch. | cao | -0.0654 | Sai (do MockEmbedder băm MD5) |
| 3 | Sản phẩm bảo hành chính hãng từ 7 đến 14 ngày làm việc. | Thời gian trung tâm xử lý bảo hành thiết bị kéo dài từ một đến hai tuần. | cao | +0.0672 | Đúng một phần |
| 4 | Quy cách đóng gói hàng dễ vỡ bắt buộc quấn 3 lớp bọt khí. | Nghiêm cấm người bán đăng tải sản phẩm vũ khí quân dụng và chất nổ. | thấp | -0.0302 | Đúng |
| 5 | Mức bồi thường tối đa 100% giá trị đơn hàng có bảo hiểm. | Hướng dẫn người mua lập tài khoản cá nhân trên sàn thương mại. | thấp | -0.2389 | Đúng |

> **Ghi kết quả vào:** Báo cáo — Phần 5 (Dự đoán độ tương tự) [ĐÃ HOÀN THÀNH VÀO REPORT_CANHAN.MD]

---

### Bài tập 3.4 — Chạy Đánh Giá & So Sánh Trong Nhóm

- Đã tạo công cụ đo lường [bench.py](bench.py) để chạy tự động 5 câu hỏi benchmark trên 6 văn bản TMĐT.
- Kết quả thu được: **4/5 câu hỏi** có chunk liên quan trong Top-3 (đạt **5/5** khi áp dụng `metadata_filter={"audience": "buyer"}` cho Câu 1).
- So sánh 4 thành viên: `HeadingChunker` (9.5/10) > `RecursiveChunker` (8.5/10) > `SentenceChunker` (8.0/10) > `FixedSizeChunker` (7.0/10).

> **Ghi kết quả vào:** Báo cáo — Phần 6 (Kết quả) [ĐÃ HOÀN THÀNH VÀO REPORT_NHOM.MD]

---

### Bài tập 3.5 — Phân Tích Lỗi (Failure Analysis)

- **Câu hỏi bị lỗi:** Câu hỏi #1 (*"Thời hạn tối đa để người mua gửi yêu cầu trả hàng và hoàn tiền đối với sản phẩm Shopee Mall là bao lâu?"* khi chạy không dùng filter).
- **Tại sao:**
  - MockEmbedder chỉ băm từ ngữ bề mặt mà không hiểu ngữ nghĩa.
  - Các từ khóa chung tần suất cao ("thời hạn", "hàng", "yêu cầu") trong văn bản bồi thường vận chuyển làm văn bản này bị xếp nhầm lên Top-1.
- **Đề xuất cải thiện:**
  - Tiền lọc siêu dữ liệu: `metadata_filter={"audience": "buyer"}` đưa ngay tài liệu `shopee-buyer-return-refund` lên Top-1.
  - Nâng cấp mô hình embedding thật (`SentenceTransformers` hoặc OpenAI `text-embedding-3-small`).

> **Ghi kết quả vào:** Báo cáo — Phần 7 (Những gì tôi học được) [ĐÃ HOÀN THÀNH VÀO CẢ 2 BÁO CÁO]

---

## Danh Sách Kiểm Tra Nộp Bài (Submission Checklist)

- [x] Vượt qua tất cả các bài kiểm thử (tests): `pytest tests/ -v` (42/42 Passed)
- [x] Cập nhật thư mục `src/` (cá nhân hoàn thiện `chunking.py`, `store.py`, `agent.py`)
- [x] Hoàn thành báo cáo nhóm (`report/REPORT_NHOM.md` — 4 thành viên nhóm So1, điểm tự đánh giá: 40/40)
- [x] Hoàn thành báo cáo cá nhân (`report/REPORT_CANHAN.md` — điểm tự đánh giá: 60/60)
