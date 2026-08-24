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

## Retrieval

The initial retrieval strategy combines two independent candidate sources:

1. semantic similarity over vector representations;
2. PostgreSQL full-text search over normalized chunk content.

Candidate lists are fused before optional reranking. Scope filtering is applied before candidate selection so documents from another scope never enter the ranking set.

The exact ranking strategy is treated as an implementation detail and will be evaluated against a repeatable test corpus rather than tuned around individual queries.

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
- [ ] Versioned ingestion pipeline
- [ ] Vector and full-text indexing
- [ ] Hybrid retrieval
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
