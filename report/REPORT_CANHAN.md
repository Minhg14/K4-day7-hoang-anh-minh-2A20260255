# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Hoàng Anh Minh (Mã SV: 2A20260255)
**Nhóm:** Nhóm K4-L3B (E-commerce Policy Retrieval)
**Ngày:** 20/09/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Độ tương tự cosine cao (tiệm cận 1.0) biểu thị hai vector nhúng (embedding vectors) chỉ về gần cùng một hướng trong không gian đa chiều. Về mặt ý nghĩa, điều này chỉ ra hai đoạn văn bản có sự tương đồng rất lớn về ngữ nghĩa (semantic similarity), phản ánh cùng một chủ đề hay thông điệp cốt lõi ngay cả khi dùng các từ vựng hoàn toàn khác nhau.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Khách hàng có thể gửi yêu cầu trả hàng và hoàn tiền trong vòng 15 ngày."
- Câu B: "Người mua được quyền khiếu nại đổi trả sản phẩm trong thời hạn hai tuần."
- Tại sao tương đồng: Cả hai câu đều truyền tải chính xác một quy định về quyền lợi và thời hạn hoàn trả hàng hóa của người tiêu dùng, mặc dù từ vựng khác biệt hoàn toàn (Khách hàng / Người mua; trả hàng hoàn tiền / khiếu nại đổi trả; 15 ngày / hai tuần).

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Chính sách đổi trả hàng hóa áp dụng trong thời hạn 15 ngày kể từ ngày nhận hàng."
- Câu B: "Người bán vi phạm quy định kinh doanh vũ khí quân dụng sẽ bị truy tố trước pháp luật."
- Tại sao khác: Hai câu đề cập đến hai phạm trù nghiệp vụ hoàn toàn độc lập và tách biệt (quyền lợi dịch vụ sau mua của khách hàng so với chế tài pháp lý xử phạt hình sự người bán hàng cấm).

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Độ tương tự cosine chỉ đo góc giữa hai vector mà không bị ảnh hưởng bởi độ lớn (magnitude/độ dài) của vector hay độ dài của văn bản nguồn. Ngược lại, khoảng cách Euclid bị méo mó bởi độ dài câu: hai câu có cùng ý nghĩa nhưng một câu dài và một câu ngắn sẽ có khoảng cách Euclid rất lớn, khiến việc tìm kiếm ngữ nghĩa bị sai lệch.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> Công thức: `số lượng chunk = ceil((độ_dài_tài_liệu - độ_chồng_chéo) / (kích_thước_chunk - độ_chồng_chéo))`
> Áp dụng số liệu: $L = 10000$, $S = 500$, $O = 50$
> Bước nhảy giữa các chunk: $S - O = 500 - 50 = 450$
> Số chunk = $\text{ceil}((10000 - 50) / 450) = \text{ceil}(9950 / 450) = \text{ceil}(22.111...) = 23$
> *Đáp án:* **23 chunks**.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> - Khi overlap tăng lên $100$: Bước nhảy giảm còn $500 - 100 = 400$. Số chunk = $\text{ceil}((10000 - 100) / 400) = \text{ceil}(9900 / 400) = \text{ceil}(24.75) = 25$ chunks (tăng thêm 2 chunks).
> - Lý do muốn độ chồng chéo nhiều hơn: Nhằm bảo toàn trọn vẹn ngữ cảnh ở các vị trí cắt ranh giới giữa các chunk liền kề, tránh làm đứt đôi câu văn hoặc ngắt quãng các mệnh đề quan trọng, giúp mô hình retrieval truy xuất chính xác thông tin nằm ở điểm giao nhau.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Sử dụng biểu thức chính quy lookbehind `r'(?<=[.!?])(?:\s+|\n+)'` để phát hiện và chia ranh giới câu dựa trên các ký tự kết thúc `.`, `!`, `?` mà vẫn bảo toàn dấu câu gốc. Xử lý triệt để các trường hợp ngoại lệ (văn bản rỗng, không có dấu chấm câu) và gom nhóm liên tiếp các câu theo kích thước `max_sentences_per_chunk` (đảm bảo tối thiểu 1). *Edge cases chưa xử lý được:* Các chữ viết tắt chức danh/học vị (như `TS.`, `ThS.`, `GS.`), từ viết tắt tiếng Anh (`e.g.`, `i.e.`), dấu chấm lửng (`...`), hoặc số thập phân (`3.14`, `15.5`) nếu vô tình có khoảng trắng sau dấu chấm thì regex đơn giản vẫn có thể nhận diện nhầm thành ranh giới kết thúc câu.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Áp dụng thuật toán chia để trị (divide and conquer) đệ quy theo danh sách phân tách có thứ tự ưu tiên giảm dần: đoạn văn (`\n\n`), xuống dòng (`\n`), kết thúc câu (`. `), từ (` `), và ký tự (`""`). Trường hợp cơ sở (base case) là khi văn bản nhỏ hơn hoặc bằng `chunk_size` hoặc danh sách dấu phân tách cạn kiệt; khi đó hàm sẽ gom các mảnh nhỏ lại với nhau sao cho độ dài mỗi chunk không vượt quá `chunk_size`.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Tài liệu được chuẩn hóa thành từ điển lưu trữ nội bộ (in-memory) gồm `id`, `content`, `metadata` và vector nhúng được tính qua `self._embedding_fn`. Khi tìm kiếm (`search`), hàm nhúng câu truy vấn, tính độ tương tự tích vô hướng (dot product/cosine) với tất cả các vector đã lưu, sau đó sắp xếp theo điểm số giảm dần và trả về `top_k` bản ghi phù hợp nhất.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Sử dụng chiến lược tiền lọc (pre-filtering): lọc tập hợp các bản ghi thỏa mãn toàn bộ các cặp key-value trong `metadata_filter` trước khi tính điểm vector để tối ưu hiệu năng và độ chính xác. Hàm `delete_document` lọc bỏ tất cả các chunk có `id` hoặc `metadata['doc_id']` khớp với tham số truyền vào và trả về `True` nếu có ít nhất một bản ghi bị xóa.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Thực hiện đúng mô hình RAG tiêu chuẩn: gọi `store.search` để lấy `top_k` chunk liên quan nhất, ghép nối nội dung các chunk thành khối ngữ cảnh `Context information`, sau đó cấu trúc prompt nghiêm ngặt: `"Context information:\n{context}\n\nQuestion: {question}\n\nAnswer based on the context above:"` và chuyển cho `llm_fn` để tổng hợp câu trả lời có trích dẫn ngữ cảnh.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```text
============================= test session starts =============================
platform win32 -- Python 3.11.2, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\lab7sang\K4-L3B-hoang-anh-minh-2A20260255
plugins: anyio-4.15.1
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.09s ==============================
```

**Số lượng bài test vượt qua (pass):** **42** / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Người mua có quyền yêu cầu trả hàng trong vòng 15 ngày. | Khách hàng được phép khiếu nại đổi trả sản phẩm trong hai tuần. | cao | -0.2540 | Sai (do mock) |
| 2 | Người bán có 48 giờ để phản hồi yêu cầu hoàn tiền. | Thời hạn shop phản hồi khiếu nại trả hàng là 2 ngày lịch. | cao | -0.0654 | Sai (do mock) |
| 3 | Sản phẩm bảo hành chính hãng từ 7 đến 14 ngày làm việc. | Thời gian trung tâm xử lý bảo hành thiết bị kéo dài từ một đến hai tuần. | cao | +0.0672 | Đúng một phần |
| 4 | Quy cách đóng gói hàng dễ vỡ bắt buộc quấn 3 lớp bọt khí. | Nghiêm cấm người bán đăng tải sản phẩm vũ khí quân dụng và chất nổ. | thấp | -0.0302 | Đúng |
| 5 | Mức bồi thường tối đa 100% giá trị đơn hàng có bảo hiểm. | Hướng dẫn người mua lập tài khoản cá nhân trên sàn thương mại. | thấp | -0.2389 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Điểm bất ngờ nhất là các cặp câu mang ngữ nghĩa tương đồng cao (Cặp 1, Cặp 2) lại có điểm cosine tương đồng âm khi chạy trên `MockEmbedder`. Nguyên nhân là vì `MockEmbedder` chỉ sử dụng hàm băm chuỗi ký tự bề mặt (deterministic string hash) mà không có trọng số ngữ nghĩa thực thụ; điều này chứng minh rằng để hệ thống RAG hoạt động chính xác trong thực tế, bắt buộc phải dùng các mô hình ngôn ngữ Transformer (như sentence-transformers hay OpenAI text-embedding) có khả năng biểu diễn ngữ nghĩa trong không gian liên tục.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Thời hạn tối đa để người mua gửi yêu cầu trả hàng và hoàn tiền Shopee Mall là bao lâu? | `shopee-buyer-return-refund`: Người mua có thời hạn yêu cầu trả hàng trong vòng 15 ngày... | +0.226 | Có | 15 ngày kể từ ngày nhận hàng thành công. |
| 2 | Người bán có bao nhiêu thời gian để phản hồi khi người mua yêu cầu trả hàng? | `shopee-seller-dispute-resolution`: Người Bán có thời hạn phản hồi tối đa là 48 giờ (2 ngày lịch)... (filter: seller) | +0.256 | Có | 48 giờ (2 ngày lịch) kể từ lúc nhận thông báo. |
| 3 | Thời gian xử lý bảo hành tiêu chuẩn sản phẩm tại Shopee là bao nhiêu ngày? | `shopee-buyer-warranty-policy`: Thời gian kiểm tra và xử lý bảo hành tiêu chuẩn từ 07 đến 14 ngày làm việc... | +0.225 | Có | Từ 07 đến 14 ngày làm việc. |
| 4 | Quy chuẩn đóng gói thùng carton mấy lớp cho hàng nặng trên 5kg trên Shopee? | `shopee-seller-packaging-guidelines`: Thùng carton tối thiểu 5 lớp cho hàng nặng trên 5 kg hoặc hàng dễ vỡ... | +0.272 | Có | Thùng carton tối thiểu 5 lớp và 2-3 lớp xốp khí. |
| 5 | Mức bồi thường tổn thất cho đơn hàng vận chuyển Shopee không có bảo hiểm là bao nhiêu? | `shopee-shipping-damage-compensation`: Tối đa bằng 04 lần cước phí vận chuyển hoặc tối đa 1.000.000 VNĐ... | +0.201 | Có | Tối đa 04 lần cước vận chuyển hoặc 1.000.000 VNĐ. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** **4** / 5 (Đạt 5/5 khi bổ sung bộ lọc metadata cho Câu 1)

**Phân tích lỗi (Failure Case Analysis — Bài tập 3.5):**
> Ở Câu 1 ("Thời hạn người mua gửi yêu cầu trả hàng Shopee Mall"), khi không dùng bộ lọc metadata, Top-3 bị chiếm bởi tài liệu bồi thường vận chuyển (`shopee-shipping-damage-compensation`) do trùng lặp các từ khóa bề mặt tần suất cao ("thời hạn", "yêu cầu", "hàng"). Tuy nhiên, ngay khi bổ sung `metadata_filter={"audience": "buyer"}`, tài liệu `shopee-buyer-return-refund` lập tức vươn lên Top-1 (+0.226). Điều này chứng minh siêu dữ liệu (metadata) đóng vai trò quyết định để cứu các trường hợp truy xuất thất bại do nhiễu từ vựng.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Nhận thấy tầm quan trọng quyết định của siêu dữ liệu (`metadata_filter`): Nếu không lọc theo `audience`, các câu hỏi về thời hạn của người bán (48h) rất dễ bị nhầm sang điều khoản 15 ngày của người mua do cả hai tài liệu đều chứa nhiều từ khóa chung như "trả hàng", "hoàn tiền". Việc phân tách dữ liệu sạch và tiền lọc metadata giải quyết triệt để bài toán này.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |
