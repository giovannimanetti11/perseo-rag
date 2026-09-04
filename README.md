# Perseo RAG

[![CI](https://github.com/giovannimanetti11/perseo-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/giovannimanetti11/perseo-rag/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/giovannimanetti11/perseo-rag?style=flat-square)](LICENSE)
[![Last commit](https://img.shields.io/github/last-commit/giovannimanetti11/perseo-rag?style=flat-square)](https://github.com/giovannimanetti11/perseo-rag/commits/main)
[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-service%20layer-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17-4169E1?style=flat-square&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![pgvector](https://img.shields.io/badge/pgvector-vector%20search-336791?style=flat-square&logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)

Perseo RAG is a retrieval service for grounded question answering over versioned document collections.

The project is built around a small set of constraints: scope isolation, traceable answers, deterministic ingestion, provider-independent generation, and a storage model that remains inspectable without depending on a proprietary vector database.

> The repository is in active development. Public interfaces may change until the first tagged release.

## Goals

Perseo RAG is intended to provide a reusable retrieval layer for applications that need to:

- ingest structured or semi-structured documents;
- preserve document history instead of overwriting source material;
- retrieve evidence with scope boundaries applied before ranking;
- combine semantic and lexical retrieval;
- return answers with source attribution;
- abstain when the available evidence is insufficient.

The core is deliberately application-agnostic. A scope represents the authorization boundary chosen by the integrating application, such as a workspace, organization or user. Product-specific adapters, deployment topology and environment-specific configuration are outside the scope of this repository.

## Design principles

**Isolation first**  
Scope boundaries are part of the data model and retrieval path, not an application-level convention.

**Evidence over fluent output**  
Generation is downstream from retrieval. Answers are expected to remain grounded in retrieved material and expose their sources.

**Versioned source material**  
Documents are immutable at the version level. New source states create new versions, making historical retrieval and audit possible.

**Storage that can be inspected**  
PostgreSQL stores application data, metadata and vector representations. Retrieval can be reasoned about with ordinary database tooling.

**Replaceable providers**  
Embedding and generation providers are behind internal interfaces. The domain layer does not depend on a specific model runtime.

**No implicit network access**  
The core accepts normalized input. It does not fetch arbitrary remote URLs on behalf of callers.

## High-level architecture

```text
Source adapter
     │
     ▼
Normalization
     │
     ▼
Document versioning
     │
     ▼
Chunking
     │
     ├──────────────┐
     ▼              ▼
Embeddings     Text indexing
     │              │
     └──────┬───────┘
            ▼
      Hybrid retrieval
            │
            ▼
        Reranking
            │
            ▼
     Grounded context
            │
            ▼
        Generation
            │
            ▼
    Answer + citations
```

The storage hierarchy is intentionally explicit:

```text
Scope
└── Collection
    └── Document
        └── Version
            └── Chunk
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the architectural boundaries, data model and security invariants.

## Ingestion

Ingestion accepts normalized application data without embedding authorization information in the payload. The active scope is supplied separately from authenticated application context.

A document is identified by its collection, source type and source reference. Each source state is normalized and fingerprinted from its content and metadata:

```text
Document input
      │
      ▼
Normalization
      │
      ▼
Deterministic fingerprint
      │
      ▼
Document upsert
      │
      ▼
New state?
  ├── no  ──► existing version
  └── yes ──► new immutable version
                  │
                  ▼
               chunks
```

Equivalent content produces the same fingerprint regardless of line-ending differences, trailing whitespace or metadata key order. Re-ingesting the same state is idempotent and does not duplicate versions or chunks.

`source_ref` is intentionally opaque to the core. URL canonicalization or other source-specific identity rules belong in source adapters.

## Indexing

Chunk text and derived indexes have separate lifecycles.

Full-text search vectors are generated by PostgreSQL and indexed with GIN. Embeddings are stored in a dedicated table keyed by scope, chunk and provider. Reindexing with a different provider therefore does not mutate document versions or replace another provider's vectors.

Embedding implementations conform to a small provider interface:

```text
chunks
  │
  ▼
EmbeddingProvider
  │
  ▼
validation
  │
  ▼
provider-specific index
```

Provider output is validated for count, dimensionality and finite numeric values before it is persisted.

## Retrieval

Perseo RAG exposes independent lexical and dense retrieval paths.

Lexical search uses PostgreSQL full-text search. User queries are parsed with `websearch_to_tsquery` and ranked with cover-density ranking.

Dense search embeds the query with the same provider identity used for the selected vector index, filters by scope, provider and dimensionality, then orders candidates by cosine distance.

Hybrid retrieval combines the two ranked candidate lists with Reciprocal Rank Fusion:

```text
query
 ├── lexical candidates
 └── dense candidates
          │
          ▼
        RRF
          │
          ▼
    fused evidence
```

Fusion depends on rank positions rather than the raw scores produced by each retrieval method, so lexical and vector score scales do not need to be normalized against one another.

Both candidate lists are scope-constrained before fusion. The fusion layer never receives cross-scope candidates.

## Grounding and citations

Each generated answer is derived from a bounded context assembled from retrieved chunks.

A citation resolves back to a concrete document version and chunk. This keeps the response traceable to the source state that was actually used, even when a document has changed since ingestion.

When retrieval does not provide enough evidence, the expected behavior is to abstain rather than fill gaps from model priors.

## Security model

Security-sensitive behavior is part of the design rather than a deployment afterthought.

The baseline includes:

- scope-constrained retrieval;
- PostgreSQL row-level security as a second isolation boundary;
- bounded input and output sizes;
- no arbitrary URL fetching in the core;
- untrusted treatment of retrieved document content;
- logs that exclude raw document bodies and credentials;
- dedicated cross-scope leakage tests;
- no executable tools exposed to the generation layer.

Operational addresses, credentials, private integration details and production deployment configuration do not belong in this repository.

See [SECURITY.md](SECURITY.md) for vulnerability reporting and repository security expectations.

## Planned stack

| Area | Choice |
| --- | --- |
| Runtime | Python 3.12+ |
| Service layer | FastAPI |
| Database | PostgreSQL 17 |
| Vector search | pgvector |
| Migrations | Alembic |
| Validation | Pydantic |
| Dependency management | uv |
| Test suite | pytest |
| Linting / formatting | Ruff |
| Type checking | mypy |

These choices describe the public core. Provider-specific integrations remain replaceable.

## Development

Requirements:

- Python 3.12 or newer;
- [uv](https://docs.astral.sh/uv/);
- Docker with Compose.

Create the local environment, start PostgreSQL and apply migrations:

```bash
cp .env.example .env
uv sync --all-groups
docker compose up -d --wait postgres
uv run alembic upgrade head
```

The development setup uses separate database identities for schema migrations and application traffic. The application identity is non-privileged and does not bypass row-level security.

Run the service locally:

```bash
uv run uvicorn perseo_rag.api:app --reload
```

Run the same quality checks enforced by CI:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest --cov=perseo_rag --cov-report=term-missing
```

## Development roadmap

- [x] Project bootstrap and quality gates
- [x] Scope, collection and document domain model
- [x] Versioned ingestion pipeline
- [x] Embedding and full-text indexing
- [x] Lexical retrieval
- [x] Dense retrieval
- [x] Hybrid retrieval and rank fusion
- [ ] Grounded generation and citations
- [x] Row-level security policies
- [ ] Retrieval evaluation suite
- [x] Cross-scope security tests
- [ ] First tagged release

## Repository boundaries

This repository contains the reusable core only.

It intentionally does **not** contain:

- production credentials or secrets;
- private customer data;
- deployment-specific hostnames, ports or network topology;
- application-specific source adapters;
- production infrastructure manifests;
- private operational runbooks.

Synthetic fixtures will be used for examples and tests.

## License

Licensed under the [Apache License 2.0](LICENSE).
