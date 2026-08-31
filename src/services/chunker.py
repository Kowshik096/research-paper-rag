def split_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        while len(para) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(para[:chunk_size])
            para = para[chunk_size - overlap:]
        if len(current) + len(para) + 1 <= chunk_size:
            current = f"{current} {para}".strip()
        else:
            if current:
                chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    return dedupe_short(chunks)


def sliding_window_chunks(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(text), step):
        piece = text[start : start + chunk_size]
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(text):
            break
    return chunks


def dedupe_short(chunks: list[str], min_len: int = 80) -> list[str]:
    merged: list[str] = []
    for chunk in chunks:
        if merged and len(chunk) < min_len:
            merged[-1] = f"{merged[-1]} {chunk}"
        else:
            merged.append(chunk)
    return merged
