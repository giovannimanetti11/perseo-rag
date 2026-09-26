# Perseo RAG

[![CI](https://github.com/giovannimanetti11/perseo-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/giovannimanetti11/perseo-rag/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/giovannimanetti11/perseo-rag?style=flat-square)](https://github.com/giovannimanetti11/perseo-rag/releases)
[![License](https://img.shields.io/github/license/giovannimanetti11/perseo-rag?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17-4169E1?style=flat-square&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![pgvector](https://img.shields.io/badge/pgvector-vector%20search-336791?style=flat-square&logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)

**Provider-agnostic RAG core with versioned ingestion, hybrid PostgreSQL/pgvector retrieval, grounded citations, scope isolation, evaluation, and FastAPI orchestration.**

Perseo RAG is a reusable retrieval and grounding service for applications that need traceable question answering over private or shared document collections. The core is deliberately independent from source systems, model runtimes and product-specific authorization models.

Version 1.0.0 is the first stable release of the project.

## What it provides

- deterministic, idempotent document ingestion;
- immutable document versions with an explicit current-version head;
- PostgreSQL full-text search;
- provider-keyed vector embeddings with pgvector;
- dense and lexical retrieval combined with Reciprocal Rank Fusion;
- scope isolation before candidate ranking;
- PostgreSQL Row-Level Security as an independent enforcement layer;
- bounded grounded context with provenance and local citation identifiers;
- evidence-based abstention before generation;
- citation validation after generation;
- provider-independent embedding and generation interfaces;
- bounded retry primitives for transient provider failures;
- synthetic retrieval evaluation with Recall@K, MRR and leakage metrics;
- FastAPI orchestration that keeps authorization scope outside query payloads.

## Design principles

**Isolation first**  
Authorization scope is part of persistence and retrieval. Candidate selection is never global-then-filtered.

**Evidence before generation**  
Generation only receives bounded evidence that has passed scope validation. Insufficient evidence produces an explicit abstention.

**Current-state retrieval, preserved history**  
Every new source state creates an immutable version. A separate document head selects the current version used by default retrieval. Historical versions remain stored for audit and future history-aware use cases.

**Inspectable storage**  
Documents, metadata, lexical indexes and vectors live in PostgreSQL. The system does not require a proprietary vector database.

**Replaceable providers**  
Embedding and generation providers are protocols owned by the core. Provider SDKs and deployment-specific adapters stay outside the domain model.

**No implicit network access**  
The core accepts normalized data and does not fetch arbitrary URLs on behalf of callers.

## Architecture

```text
Source adapter
     │
     ▼
Normalization
     │
     ▼
Versioned ingestion
     │
     ├── immutable versions
     └── current document head
              │
              ▼
          Chunking
          /      \
         ▼        ▼
   Embeddings    FTS
         \        /
          ▼      ▼
       Hybrid retrieval
              │
              ▼
       Grounded context
              │
              ▼
        Evidence policy
         /          \
        ▼            ▼
    abstain       Generation
                      │
                      ▼
             Citation validation
                      │
                      ▼
              Grounded answer
```

The storage hierarchy is explicit:

```text
Scope
└── Collection
    └── Document
        ├── DocumentHead ──► current DocumentVersion
        └── DocumentVersion
            └── Chunk
                └── ChunkEmbedding
```

## Versioned ingestion

A document is identified within a scope by its collection, source type and opaque source reference.

Content and metadata are normalized and fingerprinted. Re-ingesting an equivalent source state resolves to the existing version; a changed state creates a new immutable version and moves the document head.

A sequence such as:

```text
A → B → A
```

produces two versions, not three. The head moves back to the original A version.

The active scope is supplied separately from document input. Source-specific canonicalization belongs to the source adapter.

## Retrieval

### Lexical

PostgreSQL full-text search uses a generated `tsvector` column, a GIN index, `websearch_to_tsquery` for user input and `ts_rank_cd` for ranking.

### Dense

Embeddings are stored separately from chunks and keyed by scope, chunk and provider identity. Dense retrieval uses cosine distance and requires matching provider identity and dimensionality.

Version 1.0 uses exact vector search. ANN index selection is intentionally left to deployment-specific optimization because provider dimensionality and corpus size are not core invariants.

### Hybrid

Lexical and dense rankings are combined with Reciprocal Rank Fusion:

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

Both retrieval branches join through the current document head before ranking, so obsolete versions do not enter the default candidate set.

## Scope isolation

The required security order is:

```text
authenticated host request
        │
        ▼
scope resolution
        │
        ▼
application authorization
        │
        ▼
PostgreSQL RLS
        │
        ▼
scope-constrained candidates
        │
        ▼
ranking
        │
        ▼
context scope validation
```

The runtime database role is separate from the migration role and must not be a superuser or have `BYPASSRLS`.

The HTTP query contract does not accept `scope_id`. Scope resolution belongs to the integrating host application.

## Grounding and generation

Retrieved hits are converted into a structured `GroundedContext`. Each evidence item retains:

- scope ID;
- chunk ID;
- document version ID;
- document ID;
- source reference;
- retrieval score;
- local citation ID.

The context builder enforces character and item budgets, removes duplicate chunks and rejects cross-scope evidence.

Generation is downstream from a deterministic evidence policy:

```text
question + grounded context
          │
          ▼
   sufficient evidence?
      /           \
    no             yes
    │               │
    ▼               ▼
 abstain      provider request
                    │
                    ▼
             text + citation IDs
                    │
                    ▼
             citation validation
```

An answered result must cite at least one evidence item from the active context. Unknown citations, empty output and uncited answers are rejected.

## Provider contracts and resilience

The core defines protocols for embedding and generation providers. Concrete adapters own transport-specific serialization and translate external failures into the shared error taxonomy.

Transient timeout, unavailability and rate-limit errors can use bounded exponential backoff. Protocol errors are not retried automatically.

A provider handling private data must be explicitly approved by the deployment that configures it.

## Application integration

The default FastAPI application exposes infrastructure-safe endpoints only. The query route is registered when the host injects both a query handler and a scope resolver.

A typical integration composes the core like this:

```python
from perseo_rag.api import create_app
from perseo_rag.application import QueryService
from perseo_rag.generation import GenerationService
from perseo_rag.grounding import GroundedContextBuilder
from perseo_rag.retrieval import HybridRetriever

query_service = QueryService(
    HybridRetriever(session_factory),
    GroundedContextBuilder(),
    GenerationService(),
    embedding_provider,
    generation_provider,
)

app = create_app(
    query_handler=query_service,
    scope_resolver=resolve_authenticated_scope,
)
```

`embedding_provider`, `generation_provider` and `resolve_authenticated_scope` are application-owned adapters. They are intentionally not implemented by the reusable core.

## Evaluation

The repository includes deterministic synthetic fixtures under `eval/`.

The baseline metrics are:

- Recall@K;
- Mean Reciprocal Rank;
- cross-scope leakage rate;
- number of cases with leaked evidence.

Cross-scope leakage has a required value of zero. Evaluation fixtures are intended for regression detection, not as a substitute for domain-specific retrieval benchmarks.

## Technology

| Area | Choice |
| --- | --- |
| Runtime | Python 3.12+ |
| Service layer | FastAPI |
| Database | PostgreSQL 17 |
| Vector storage/search | pgvector |
| ORM | SQLAlchemy 2 |
| Migrations | Alembic |
| Validation | Pydantic |
| Dependency management | uv |
| Tests | pytest / pytest-cov |
| Linting / formatting | Ruff |
| Type checking | mypy strict |

## Development

Requirements:

- Python 3.12 or newer;
- [uv](https://docs.astral.sh/uv/);
- Docker with Compose.

Set up the repository:

```bash
cp .env.example .env
uv sync --all-groups
docker compose up -d --wait postgres
uv run alembic upgrade head
```

Run the default service:

```bash
uv run uvicorn perseo_rag.api:app --reload
```

The default app exposes `/health`. Query handling requires host-provided providers and scope resolution as shown above.

Run the quality gates:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest --cov=perseo_rag --cov-report=term-missing
uv build
```

## Version 1.0 boundaries

Version 1.0 intentionally does not include:

- source-system-specific crawlers or fetchers;
- a concrete embedding model;
- a concrete generation model;
- authentication implementation;
- deployment-specific infrastructure;
- arbitrary model tools, shell access or unrestricted HTTP access;
- an ANN index chosen on behalf of every deployment;
- a reranker.

Those concerns can be added through adapters or later evaluated extensions without weakening the core trust boundaries.

## Releases

Latest stable release: [v1.0.0](https://github.com/giovannimanetti11/perseo-rag/releases/tag/v1.0.0).

Perseo RAG follows semantic versioning from `1.0.0`. Tagged releases are published through the repository release workflow only after migrations, formatting, linting, type checks, tests and package build complete successfully.

## Repository boundaries

This repository contains the reusable core and synthetic test fixtures only. It does not contain production credentials, private datasets, customer-specific adapters, private infrastructure or operational runbooks.

## License

Licensed under the [Apache License 2.0](LICENSE).
