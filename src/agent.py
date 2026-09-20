from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with numbered chunks [1], [2], [3], sources, and anti-hallucination constraints.
        3. Call the LLM to generate an answer with citations.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # Handle empty store: return notification directly without calling LLM
        if self.store.get_collection_size() == 0:
            return "Không tìm thấy thông tin phù hợp trong cơ sở tri thức (cơ sở tri thức đang rỗng)."

        # 1. Retrieve
        results = self.store.search(question, top_k=top_k)
        if not results:
            return "Không tìm thấy thông tin phù hợp trong cơ sở tri thức."

        # 2. Build prompt with source traceability and anti-hallucination constraints
        context_parts = []
        for i, r in enumerate(results, 1):
            source = (
                r.get("metadata", {}).get("source")
                or r.get("metadata", {}).get("source_url")
                or r.get("metadata", {}).get("doc_id")
                or r.get("id")
                or "Unknown"
            )
            context_parts.append(f"[{i}] (Nguồn: {source}):\n{r['content']}")

        context = "\n\n".join(context_parts)
        prompt = (
            f"Bạn là một trợ lý hỏi đáp tri thức hữu ích. Hãy trả lời câu hỏi dựa CHỈ trên các đoạn ngữ cảnh dưới đây.\n\n"
            f"QUY TẮC BẮT BUỘC:\n"
            f"1. Trích dẫn đúng số hiệu nguồn [N] tương ứng cho từng thông tin trong câu trả lời.\n"
            f"2. Không được bịa đặt hoặc suy đoán thông tin ngoài ngữ cảnh được cung cấp.\n"
            f"3. Nếu ngữ cảnh không chứa thông tin để trả lời câu hỏi, hãy nói rõ rằng không tìm thấy thông tin.\n\n"
            f"NGỮ CẢNH:\n{context}\n\n"
            f"CÂU HỎI: {question}\n\n"
            f"CÂU TRẢ LỜI:"
        )

        # 3. Call LLM
        return self.llm_fn(prompt)
