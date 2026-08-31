from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    query: str = Field(examples=["ocean mesoscale eddies"])
    max_results: int = Field(default=5, ge=1, le=50)


class PaperOut(BaseModel):
    arxiv_id: str
    title: str
    authors: str
    published: str
    chunk_count: int = 0


class SearchRequest(BaseModel):
    query: str
    mode: str = Field(default="hybrid", pattern="^(bm25|vector|hybrid)$")
    top_k: int = Field(default=6, ge=1, le=20)


class SearchHit(BaseModel):
    title: str
    section: str
    content: str
    score: float


class AskRequest(BaseModel):
    question: str
    top_k: int = 6


class Citation(BaseModel):
    index: int
    title: str
    section: str
    snippet: str


class GradeOut(BaseModel):
    index: int
    title: str
    grade: str
    reason: str = ""


class CragTrace(BaseModel):
    attempted: bool = False
    rewritten_query: str | None = None
    initial_usable: bool = False
    final_usable: bool = False
    refused: bool = False
    grades: list[GradeOut] = []


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    cached: bool = False
    crag: CragTrace | None = None
