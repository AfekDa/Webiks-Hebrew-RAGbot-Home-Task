# Run the upgraded backend locally

Install the environment and download the assets described in `UPSTREAM.md`.
Commands below run from the repository root. No OpenAI key is needed; only the
answer generation is mocked. Retrieval uses the real Hebrew embedder and
Elasticsearch cosine search, with the submitted **page scoring** step on top.

## 1. Start Elasticsearch

Use Elasticsearch 8.12.2 bound to localhost. For Docker:

```powershell
docker run --name hebrew-rag-es -e "discovery.type=single-node" -e "xpack.security.enabled=false" -e "ES_JAVA_OPTS=-Xms2g -Xmx2g" -p 127.0.0.1:9200:9200 elasticsearch:8.12.2
```

For this Windows session, the standalone distribution is installed under
`.runtime/elasticsearch-8.12.2/`. Run it in a separate terminal:

```powershell
$env:ES_JAVA_OPTS='-Xms2g -Xmx2g'
& .runtime/elasticsearch-8.12.2/bin/elasticsearch.bat
```

Its local `config/elasticsearch.yml` sets `network.host: 127.0.0.1`,
`discovery.type: single-node`, and `xpack.security.enabled: false`.

This local instance uses absolute disk watermarks of 10 GB (low), 5 GB (high),
and 2 GB (flood stage). The default percentages blocked allocation despite about
45 GB free on this PC. These are persistent settings on this local cluster;
disk protection remains enabled. See the [Elasticsearch 8.12 settings](https://www.elastic.co/guide/en/elasticsearch/reference/8.12/modules-cluster.html#disk-based-shard-allocation).

## 2. Prepare and index the corpus

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_eval.ps1
.venv/Scripts/python scripts/seed_demo.py --tag full
```

Seeding reuses the evaluated embeddings and stores the original text and metadata
with the engine's vector field. It creates dedicated `hebrew_rag_local_*` indices;
it does not delete existing indices. Re-running with the same corpus is safe.
An incompatible existing corpus causes an error.

## 3. Launch and query

```powershell
.venv/Scripts/python scripts/run_demo.py
```

Open http://127.0.0.1:5000/docs for the API. `/health` returns 200.
Submit `POST /search` with JSON `{"query":"your Hebrew question","asked_from":"local-demo"}`.
The response includes retrieved `docs`, a mock `llm_result`, and retrieval timing.

`scripts/run_demo.py` resolves the paths and turns on **page scoring** with the
evaluated weights (title 0.25, second paragraph 0.25, margin gate 0.05). It adds
one small model call per query (embedding the titles of the ≤50 retrieved
candidates) and needs no special hardware. To compare the original behaviour in a
separate launch, set `$env:PAGE_SCORING_ENABLED='false'` before starting.

Change settings by setting environment variables and restarting the process.
The server listens only on localhost.

Stop the backend or Elasticsearch with Ctrl+C in its terminal.

## Targeted tests

```powershell
.venv/Scripts/python -m pytest --import-mode=importlib Webiks-Hebrew-RAGbot/tests/test_page_scoring.py Webiks-Hebrew-RAGbot-Demo/tests/test_gpt_client.py -q
```

These check the page-scoring rule (title lift, second paragraph, margin gate,
ties), that the engine keeps each page's best paragraph while reordering pages,
that page scoring off is byte-for-byte the original behaviour, invalid settings,
and key-free mock generation. The existing integration suite additionally uses
Docker testcontainers.

To confirm the live API uses the improvement, run `scripts/verify_demo.py` as
described in `rag_eval/README.md`.
