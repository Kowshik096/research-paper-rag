from opensearchpy import OpenSearch

from src.config import get_settings

CHUNKS_INDEX = "chunks"

INDEX_BODY = {
    "settings": {"index": {"knn": True}},
    "mappings": {
        "properties": {
            "paper_id": {"type": "integer"},
            "arxiv_id": {"type": "keyword"},
            "title": {"type": "text"},
            "section": {"type": "keyword"},
            "content": {"type": "text"},
            "embedding": {
                "type": "knn_vector",
                "dimension": 768,
                "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "nmslib"},
            },
        }
    },
}

_client = None


def get_client() -> OpenSearch:
    global _client
    if _client is None:
        _client = OpenSearch(hosts=[get_settings().opensearch_url], use_ssl=False, verify_certs=False)
    return _client


def init_index(index_name: str = CHUNKS_INDEX) -> None:
    client = get_client()
    if not client.indices.exists(index=index_name):
        body = dict(INDEX_BODY)
        body["mappings"] = dict(INDEX_BODY["mappings"])
        props = dict(body["mappings"]["properties"])
        emb = dict(props["embedding"])
        emb["dimension"] = get_settings().embed_dim
        props["embedding"] = emb
        body["mappings"]["properties"] = props
        client.indices.create(index=index_name, body=body)


def delete_index(index_name: str) -> None:
    get_client().indices.delete(index=index_name, ignore=[400, 404])


def index_chunks(docs: list[dict], index_name: str = CHUNKS_INDEX) -> None:
    body = []
    for doc in docs:
        body.append({"index": {"_index": index_name}})
        body.append(doc)
    if body:
        get_client().bulk(body=body, params={"refresh": "true"})


def bm25_search(query: str, top_k: int, index_name: str = CHUNKS_INDEX) -> list[dict]:
    body = {
        "size": top_k,
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["title^2", "content"],
                "operator": "or",
            }
        },
    }
    resp = get_client().search(index=index_name, body=body)
    return [_hit_to_doc(h) for h in resp["hits"]["hits"]]


def vector_search(embedding: list[float], top_k: int, index_name: str = CHUNKS_INDEX) -> list[dict]:
    body = {
        "size": top_k,
        "query": {"knn": {"embedding": {"vector": embedding, "k": top_k}}},
        "_source": {"excludes": ["embedding"]},
    }
    resp = get_client().search(index=index_name, body=body)
    return [_hit_to_doc(h) for h in resp["hits"]["hits"]]


def _hit_to_doc(hit: dict) -> dict:
    source = hit["_source"]
    source["_score"] = hit["_score"]
    source["_id"] = hit["_id"]
    return source
