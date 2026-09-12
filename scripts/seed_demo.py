"""Index the evaluation corpus and its cached vectors into local Elasticsearch.

Uses the same fields and cosine query as the Demo engine; avoids re-embedding.
Never deletes indices. Refuses to overwrite a different dataset's index.
"""
import argparse
import hashlib
import json

import numpy as np
from elasticsearch import Elasticsearch, helpers

from run_demo import ROOT, configure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="full")
    args = parser.parse_args()
    configure()
    import get_es_client
    from webiks_hebrew_ragbot.document import document_definition_factory
    from webiks_hebrew_ragbot.elastic_model import index_from_doc_id

    cache = ROOT / "rag_eval/cache" / args.tag
    raw_paras = (cache / "paras.json").read_bytes()
    paras = json.loads(raw_paras)
    embeddings = np.load(cache / "para_emb.npy", mmap_mode="r")
    cfg = json.loads((cache / "config.json").read_text())
    if cfg["seq"] != 512 or len(paras) != len(embeddings):
        raise ValueError("Cache must use 512 tokens and contain one vector per paragraph")
    definitions = document_definition_factory()
    vector_field = f"{definitions.field_to_embed}_{definitions.model_name}_vectors"
    fingerprint = hashlib.sha256(raw_paras).hexdigest()
    # Bulk helpers clone their client with options(); the upstream EsClient
    # subclass does not implement that constructor contract.
    es = Elasticsearch(_transport=get_es_client.factory().transport, request_timeout=120)
    es.info()
    indices = sorted({index_from_doc_id(int(p["doc_id"])) for p in paras})
    for index in indices:
        if es.indices.exists(index=index):
            meta = es.indices.get_mapping(index=index)[index]["mappings"].get("_meta", {})
            if meta.get("corpus_sha256") != fingerprint:
                raise ValueError(f"Index {index} belongs to a different dataset; choose a new ES_EMBEDDING_INDEX")
        else:
            es.indices.create(index=index, settings={"number_of_shards": 1, "number_of_replicas": 0}, mappings={
                "_meta": {"corpus_sha256": fingerprint},
                "properties": {
                    "doc_id": {"type": "integer"},
                    "content": {"type": "text"}, "title": {"type": "text"}, "link": {"type": "text"},
                    vector_field: {"type": "dense_vector", "dims": embeddings.shape[1], "index": False},
                },
            })

    def actions():
        for i, paragraph in enumerate(paras):
            yield {
                "_index": index_from_doc_id(int(paragraph["doc_id"])), "_id": str(i),
                "_source": {**paragraph, "doc_id": int(paragraph["doc_id"]), vector_field: embeddings[i].tolist()},
            }

    for count, (ok, result) in enumerate(helpers.streaming_bulk(es, actions(), chunk_size=128), 1):
        if not ok:
            raise RuntimeError(result)
        if count % 1024 == 0:
            print(f"Indexed {count}/{len(paras)} paragraphs", flush=True)
    es.indices.refresh(index=",".join(indices))
    actual = es.count(index=",".join(indices))["count"]
    if actual != len(paras):
        raise RuntimeError(f"Expected {len(paras)} paragraphs, found {actual}")
    print(f"Ready: {actual} paragraphs in {len(indices)} indices", flush=True)


if __name__ == "__main__":
    main()
