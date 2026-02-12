# Plan: Dgraph 25.2.0+ as a First-Class Graphiti Backend

## Context

Graphiti is a temporal knowledge graph framework with pluggable database backends (Neo4j, FalkorDB, Neptune, KuzuDB). Dgraph v25.2.0 is a fully open-source graph database with native HNSW vector search, full-text search, DQL query language, and string namespaces -- making it an excellent addition as a first-class backend.

This plan also addresses Claude Agent SDK (`claude-agent-sdk` pip package) integration, ensuring Graphiti can be used as a tool from Agent SDK-powered agents, not just via raw Anthropic API keys.

**Goals:**
1. Add Dgraph v25.2.0+ as a first-class backend via the `SearchInterface` + `GraphOperationsInterface` pattern
2. Add Claude Agent SDK integration (Graphiti as MCP tool for Agent SDK agents)
3. All changes follow existing patterns exactly for upstream PR acceptance

**Decisions (confirmed by user):**
- HTTP API via `httpx` (native async, no gRPC dependency)
- Single Dgraph namespace with `group_id` filtering (consistent with other backends)
- No KuzuDB changes in this PR (just don't go out of our way to support it)
- Claude Agent SDK = `claude-agent-sdk` pip package (v0.1.33+)

---

## Part 1: Dgraph Driver Implementation

### 1.1 Architecture: Interface-Based Integration

**Approach**: Implement `SearchInterface` + `GraphOperationsInterface`. This completely isolates DQL from the Cypher-based core, following the same pattern Neptune uses.

**Why**: The `execute_query` method takes Cypher strings. Dgraph uses DQL. The interface pattern lets us write native DQL without touching any existing query construction code. Every search function in `search_utils.py` already checks `driver.search_interface` first. Every CRUD operation in `nodes.py`/`edges.py` already checks `driver.graph_operations_interface` first.

**How `execute_query` works for Dgraph**: Instead of Cypher, it accepts DQL query strings and sends them to Dgraph's HTTP API. This is only used by code that explicitly handles `GraphProvider.DGRAPH` cases.

### 1.2 Data Model: Dgraph Schema

Dgraph predicates are global. Types group predicates. Dgraph edges (predicates) cannot hold vector embeddings or full-text indexed properties, so `RELATES_TO` must be modeled as an intermediate node (`RelatesToEdge`), identical to KuzuDB's `RelatesToNode_` pattern. Simple edges (MENTIONS, HAS_MEMBER, etc.) use facets for metadata.

```graphql
# ===== Scalar Predicates =====
graphiti.uuid: string @index(exact) @upsert .
graphiti.name: string @index(exact, fulltext) .
graphiti.group_id: string @index(exact) .
graphiti.summary: string @index(fulltext) .
graphiti.created_at: datetime @index(hour) .
graphiti.expired_at: datetime @index(hour) .
graphiti.valid_at: datetime @index(hour) .
graphiti.invalid_at: datetime @index(hour) .
graphiti.source: string @index(exact) .
graphiti.source_description: string @index(fulltext) .
graphiti.content: string @index(fulltext) .
graphiti.fact: string @index(fulltext) .
graphiti.labels: [string] @index(exact) .
graphiti.entity_edges: [string] .
graphiti.episodes: [string] .
graphiti.attributes: string .  # JSON-serialized dict

# ===== Vector Predicates =====
graphiti.name_embedding: float32vector @index(hnsw(metric:"cosine", exponent:"4")) .
graphiti.fact_embedding: float32vector @index(hnsw(metric:"cosine", exponent:"4")) .

# ===== Edge Predicates (with facets for uuid, group_id, created_at) =====
graphiti.mentions: [uid] @reverse .           # Episodic -> Entity
graphiti.has_member: [uid] @reverse .         # Community -> Entity|Community
graphiti.has_episode: [uid] @reverse .        # Saga -> Episodic
graphiti.next_episode: [uid] @reverse .       # Episodic -> Episodic
graphiti.source_entity: uid @reverse .        # RelatesToEdge -> Entity (source)
graphiti.target_entity: uid @reverse .        # RelatesToEdge -> Entity (target)

# ===== Types =====
type Entity {
    graphiti.uuid
    graphiti.name
    graphiti.group_id
    graphiti.summary
    graphiti.created_at
    graphiti.name_embedding
    graphiti.labels
    graphiti.attributes
}

type Episodic {
    graphiti.uuid
    graphiti.name
    graphiti.group_id
    graphiti.source
    graphiti.source_description
    graphiti.content
    graphiti.created_at
    graphiti.valid_at
    graphiti.entity_edges
    graphiti.mentions
    graphiti.next_episode
}

type Community {
    graphiti.uuid
    graphiti.name
    graphiti.group_id
    graphiti.summary
    graphiti.created_at
    graphiti.name_embedding
    graphiti.has_member
}

type Saga {
    graphiti.uuid
    graphiti.name
    graphiti.group_id
    graphiti.created_at
    graphiti.has_episode
}

type RelatesToEdge {
    graphiti.uuid
    graphiti.name
    graphiti.group_id
    graphiti.fact
    graphiti.fact_embedding
    graphiti.episodes
    graphiti.created_at
    graphiti.expired_at
    graphiti.valid_at
    graphiti.invalid_at
    graphiti.attributes
    graphiti.source_entity
    graphiti.target_entity
}
```

**Design rationale:**
- **`RelatesToEdge` as intermediate node**: Dgraph predicates can't hold vector embeddings or be fulltext-indexed. This mirrors KuzuDB's `RelatesToNode_` pattern, already proven in the codebase.
- **Simple edges use facets**: `MENTIONS`, `HAS_MEMBER`, `HAS_EPISODE`, `NEXT_EPISODE` only need uuid/group_id/created_at. Facets handle these efficiently.
- **Predicate namespacing**: `graphiti.` prefix avoids conflicts in shared Dgraph instances.
- **HNSW cosine, exponent 4**: Good default for typical embedding dimensions (1536). The v25.2.0 per-query `ef` and `distance_threshold` params give runtime tuning.

### 1.3 New Files

#### `graphiti_core/driver/dgraph_driver.py`
The main driver class using `httpx.AsyncClient` for native async HTTP communication with Dgraph.

```python
class DgraphDriver(GraphDriver):
    provider = GraphProvider.DGRAPH

    def __init__(self, url: str = 'http://localhost:8080', api_key: str | None = None,
                 database: str = 'default'):
        self._url = url.rstrip('/')
        self._database = database
        self._client = httpx.AsyncClient(base_url=self._url, headers=...)
        self.search_interface = DgraphSearchInterface()
        self.graph_operations_interface = DgraphGraphOperations()

    async def execute_query(self, dql_query: str, **kwargs) -> tuple[list[dict], None, None]:
        # POST to /query with DQL, parse JSON response
        # Extract variables from kwargs, format as DQL variables
        ...

    async def _mutate(self, mutation: str | dict, commit_now: bool = True) -> dict:
        # POST to /mutate?commitNow=true with NQuad or JSON mutations
        ...

    async def _alter(self, schema: str) -> dict:
        # POST to /alter with schema string
        ...

    async def build_indices_and_constraints(self, delete_existing: bool = False):
        if delete_existing:
            await self._alter('{"drop_all": true}')
        await self._alter(DGRAPH_SCHEMA)  # The schema string from 1.2

    def session(self, database: str | None = None) -> GraphDriverSession:
        return DgraphDriverSession(self)

    async def close(self):
        await self._client.aclose()

    def delete_all_indexes(self):
        return self._alter('{"drop_all": true}')
```

#### `graphiti_core/driver/dgraph_search.py`
All 12 `SearchInterface` methods implemented with native DQL.

Key implementations:

| Method | DQL Pattern |
|--------|-------------|
| `edge_fulltext_search` | `func: alloftext(graphiti.fact, $q)` + `@filter(eq(dgraph.type, "RelatesToEdge"))` |
| `edge_similarity_search` | `func: similar_to(graphiti.fact_embedding, $k, $vec)` + type filter |
| `node_fulltext_search` | `func: anyoftext(graphiti.name, $q) OR alloftext(graphiti.summary, $q)` + Entity filter |
| `node_similarity_search` | `func: similar_to(graphiti.name_embedding, $k, $vec)` + Entity filter |
| `episode_fulltext_search` | `func: alloftext(graphiti.content, $q)` + Episodic filter |
| `edge_bfs_search` | `@recurse(depth: N)` traversing `source_entity`/`target_entity` |
| `node_bfs_search` | `@recurse(depth: N)` traversing `mentions`/`source_entity`/`target_entity` |
| `community_fulltext_search` | `func: alloftext(graphiti.name, $q)` + Community filter |
| `community_similarity_search` | `func: similar_to(graphiti.name_embedding, $k, $vec)` + Community filter |
| `node_distance_reranker` | `shortest(from: uid(center), to: uid(target))` |
| `episode_mentions_reranker` | `count(~graphiti.mentions)` aggregation |

#### `graphiti_core/driver/dgraph_graph_operations.py`
All ~50+ `GraphOperationsInterface` methods using DQL upsert mutations.

**Node upsert** (all node types follow this pattern):
```python
async def node_save(self, node, driver):
    mutation = {
        "query": '{ node as var(func: eq(graphiti.uuid, "%s")) }' % node.uuid,
        "set": {
            "uid": "uid(node)",
            "dgraph.type": "Entity",
            "graphiti.uuid": node.uuid,
            "graphiti.name": node.name,
            "graphiti.group_id": node.group_id,
            "graphiti.summary": node.summary,
            "graphiti.created_at": node.created_at.isoformat(),
            "graphiti.name_embedding": node.name_embedding,
            "graphiti.labels": node.labels,
            "graphiti.attributes": json.dumps(node.attributes),
        }
    }
    await driver._mutate(mutation)
```

**Edge save (MENTIONS via facets)**:
```python
async def episodic_edge_save(self, edge, driver):
    nquads = (
        f'uid(ep) <graphiti.mentions> uid(ent) '
        f'(uuid="{edge.uuid}", group_id="{edge.group_id}", '
        f'created_at="{edge.created_at.isoformat()}") .'
    )
    query = (
        f'{{ ep as var(func: eq(graphiti.uuid, "{edge.source_node_uuid}")) }}\n'
        f'{{ ent as var(func: eq(graphiti.uuid, "{edge.target_node_uuid}")) }}'
    )
    await driver._mutate({"query": query, "set": nquads})
```

**Edge save (RELATES_TO via intermediate node)**:
```python
async def edge_save(self, edge, driver):
    # Same pattern as KuzuDB's RelatesToNode_ -- create intermediate node
    mutation = {
        "query": '...',  # lookup source, target, and existing edge by uuid
        "set": {
            "uid": "uid(edge_node)",
            "dgraph.type": "RelatesToEdge",
            "graphiti.uuid": edge.uuid,
            "graphiti.source_entity": {"uid": "uid(source)"},
            "graphiti.target_entity": {"uid": "uid(target)"},
            "graphiti.fact": edge.fact,
            "graphiti.fact_embedding": edge.fact_embedding,
            # ... all other properties
        }
    }
    await driver._mutate(mutation)
```

### 1.4 Files to Modify

| File | Change |
|------|--------|
| `graphiti_core/driver/driver.py` | Add `DGRAPH = 'dgraph'` to `GraphProvider` enum |
| `graphiti_core/driver/__init__.py` | Export `DgraphDriver` |
| `graphiti_core/nodes.py` | Add `GraphProvider.DGRAPH` to match cases (falls through to `graph_operations_interface` which is checked first) |
| `graphiti_core/edges.py` | Add `GraphProvider.DGRAPH` to match cases + `load_fact_embedding()` DQL query |
| `graphiti_core/graph_queries.py` | Add `DGRAPH` cases returning empty/no-op (all queries go through interface) |
| `graphiti_core/search/search_utils.py` | Add `DGRAPH` to `fulltext_query()` function |
| `graphiti_core/models/nodes/node_db_queries.py` | Add `DGRAPH` cases (interface handles most operations) |
| `graphiti_core/models/edges/edge_db_queries.py` | Add `DGRAPH` cases (interface handles most operations) |
| `pyproject.toml` | Add optional dep: `dgraph = ["httpx>=0.27.0"]`, add to dev deps |

### 1.5 HTTP Client Design

Using `httpx.AsyncClient` for Dgraph's HTTP API:

| Endpoint | Purpose | Method |
|----------|---------|--------|
| `POST /query` | Execute DQL queries | `execute_query()` |
| `POST /mutate?commitNow=true` | Execute mutations (JSON or NQuad) | `_mutate()` |
| `POST /alter` | Schema operations | `_alter()`, `build_indices_and_constraints()` |
| `GET /health` | Health check | `health_check()` |

Response format is JSON. DQL query responses have `data` and `extensions` keys. Mutations return `uids` map for newly created nodes.

**Auth**: Dgraph Cloud uses API tokens via `X-Dgraph-AuthToken` header. Self-hosted can use ACL tokens. Both passed via `api_key` constructor param.

---

## Part 2: Claude Agent SDK Integration

### 2.1 Background

The Claude Agent SDK (`pip install claude-agent-sdk`, v0.1.33+) provides:
- `query()` async function for running agent loops
- `ClaudeSDKClient` for bidirectional conversations
- MCP server integration via `mcp_servers` option in `ClaudeAgentOptions`
- Built-in tools (Read, Write, Bash, etc.)
- Custom tool definitions via `create_sdk_mcp_server()` and `@tool` decorator

**Key API**:
```python
from claude_agent_sdk import query, ClaudeAgentOptions

async for message in query(
    prompt="Search for...",
    options=ClaudeAgentOptions(
        mcp_servers={"graphiti": {"command": "...", "args": [...]}}
    ),
):
    print(message)
```

### 2.2 Integration Approach

Graphiti's MCP server already exposes all Graphiti operations as MCP tools. The Agent SDK can connect to MCP servers directly. The integration has two parts:

**A. Convenience wrapper for Agent SDK users** -- Create a helper that produces the correct `mcp_servers` config for `ClaudeAgentOptions`:

```python
# graphiti_core/agent_sdk.py (NEW)
from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, tool

def create_graphiti_mcp_tools(graphiti_instance):
    """Create an in-process MCP server exposing Graphiti operations
    for use with ClaudeAgentOptions.mcp_servers."""

    @tool("search_memory", "Search the knowledge graph", {...})
    async def search_memory(args):
        results = await graphiti_instance.search(args["query"], ...)
        return {"content": [{"type": "text", "text": format_results(results)}]}

    @tool("add_memory", "Add information to the knowledge graph", {...})
    async def add_memory(args):
        await graphiti_instance.add_episode(...)
        return {"content": [{"type": "text", "text": "Memory stored."}]}

    return create_sdk_mcp_server(
        name="graphiti",
        version="0.27.0",
        tools=[search_memory, add_memory, ...]
    )
```

**B. Documentation + example** showing how to use Graphiti's existing MCP server with the Agent SDK via subprocess:

```python
from claude_agent_sdk import query, ClaudeAgentOptions

async for msg in query(
    prompt="What do you know about the user's preferences?",
    options=ClaudeAgentOptions(
        mcp_servers={
            "graphiti": {
                "command": "uv",
                "args": ["run", "--directory", "/path/to/mcp_server", "python", "main.py"],
            }
        }
    ),
):
    ...
```

### 2.3 Files

| File | Action | Purpose |
|------|--------|---------|
| `graphiti_core/agent_sdk.py` | Create | In-process MCP tool wrapper for Agent SDK |
| `examples/agent_sdk/agent_with_graphiti.py` | Create | Example of Agent SDK using Graphiti |
| `pyproject.toml` | Modify | Add optional dep: `agent-sdk = ["claude-agent-sdk>=0.1.30"]` |

---

## Part 3: Implementation Order

### Phase 1: Foundation
1. Add `DGRAPH` to `GraphProvider` enum in `driver.py`
2. Create `dgraph_driver.py` with httpx-based connection, schema management, `execute_query`, `_mutate`, `_alter`
3. Add `pyproject.toml` optional dependency `dgraph = ["httpx>=0.27.0"]`

### Phase 2: Graph Operations
4. Create `dgraph_graph_operations.py` implementing `GraphOperationsInterface`
5. Implement Entity node CRUD (save, delete, get_by_uuid, get_by_uuids, get_by_group_ids, bulk)
6. Implement Episodic, Community, Saga node CRUD
7. Implement EntityEdge CRUD (RelatesToEdge intermediate node pattern)
8. Implement EpisodicEdge, CommunityEdge, HasEpisodeEdge, NextEpisodeEdge CRUD
9. Implement maintenance: clear_data, community detection, get_mentioned_nodes, etc.

### Phase 3: Search
10. Create `dgraph_search.py` implementing `SearchInterface`
11. Vector similarity search using `similar_to()`
12. Fulltext search using `alloftext`/`anyoftext`
13. BFS traversal using `@recurse`
14. Rerankers: node_distance (using `shortest()`), episode_mentions

### Phase 4: Core Integration
15. Add `GraphProvider.DGRAPH` to match statements in `nodes.py`, `edges.py`
16. Update `graph_queries.py` with DGRAPH no-op cases
17. Update `search_utils.py` fulltext_query for DGRAPH
18. Update `models/nodes/node_db_queries.py` and `models/edges/edge_db_queries.py`

### Phase 5: Agent SDK
19. Create `graphiti_core/agent_sdk.py` with in-process MCP tool wrapper
20. Create `examples/agent_sdk/agent_with_graphiti.py`
21. Add `pyproject.toml` optional dep for `claude-agent-sdk`

### Phase 6: Testing & Docs
22. Create `tests/driver/test_dgraph_driver.py` - unit tests
23. Create `tests/test_dgraph_int.py` - integration tests (mirrors `test_graphiti_int.py`)
24. Create `docker-compose.dgraph.yml` (Dgraph Zero + Alpha)
25. Create `examples/quickstart/quickstart_dgraph.py`
26. Update README.md with Dgraph in supported backends list
27. Run `make check` (format + lint + test) to verify no regressions

---

## Part 4: Complete File Manifest

### New Files (10)

| File | Purpose |
|------|---------|
| `graphiti_core/driver/dgraph_driver.py` | DgraphDriver class + DgraphDriverSession |
| `graphiti_core/driver/dgraph_search.py` | DgraphSearchInterface (12 search methods) |
| `graphiti_core/driver/dgraph_graph_operations.py` | DgraphGraphOperations (~50 CRUD methods) |
| `graphiti_core/agent_sdk.py` | Agent SDK MCP tool wrapper |
| `examples/quickstart/quickstart_dgraph.py` | Dgraph quickstart |
| `examples/agent_sdk/agent_with_graphiti.py` | Agent SDK integration example |
| `docker-compose.dgraph.yml` | Dgraph test containers |
| `tests/driver/test_dgraph_driver.py` | Driver unit tests |
| `tests/driver/test_dgraph_search.py` | Search unit tests |
| `tests/test_dgraph_int.py` | Integration tests |

### Modified Files (10)

| File | Change |
|------|--------|
| `graphiti_core/driver/driver.py` | Add `DGRAPH = 'dgraph'` to `GraphProvider` |
| `graphiti_core/driver/__init__.py` | Export `DgraphDriver` |
| `graphiti_core/nodes.py` | Add DGRAPH to match cases |
| `graphiti_core/edges.py` | Add DGRAPH to match cases + `load_fact_embedding` DQL |
| `graphiti_core/graph_queries.py` | Add DGRAPH no-op cases |
| `graphiti_core/search/search_utils.py` | Add DGRAPH to `fulltext_query()` |
| `graphiti_core/models/nodes/node_db_queries.py` | Add DGRAPH cases |
| `graphiti_core/models/edges/edge_db_queries.py` | Add DGRAPH cases |
| `pyproject.toml` | Add `dgraph` and `agent-sdk` optional deps |
| `README.md` | Add Dgraph to supported backends |

---

## Part 5: Dgraph v25.2.0 Feature Mapping

| Graphiti Need | Dgraph Feature | DQL Pattern |
|---------------|---------------|-------------|
| Vector similarity (nodes) | HNSW index on float32vector | `similar_to(graphiti.name_embedding, k, $vec)` |
| Vector similarity (edges) | HNSW index on RelatesToEdge | `similar_to(graphiti.fact_embedding, k, $vec)` |
| Fulltext search | Built-in fulltext tokenizer | `alloftext(graphiti.fact, "query")` |
| BFS traversal | Recurse queries | `@recurse(depth: N)` |
| Temporal filtering | DateTime index | `@filter(ge(graphiti.created_at, "..."))` |
| Batch upsert | Upsert blocks | Multi-NQuad mutations in single request |
| Edge properties | RelatesToEdge intermediate node | Mirrors KuzuDB's `RelatesToNode_` pattern |
| Edge metadata | Facets on predicates | `(uuid="...", group_id="...", created_at="...")` |
| Community detection | Groupby aggregation | `groupby` + count queries |
| Shortest path | `shortest()` function | `shortest(from: uid(...), to: uid(...))` |
| Per-query vector tuning | ef + distance_threshold (v25.2.0) | `similar_to(..., ef: 100, distance_threshold: 0.5)` |

---

## Part 6: Verification

1. `make format` - passes (ruff formatting)
2. `make lint` - passes (ruff + pyright)
3. `make test` - existing tests pass (no regressions)
4. `pytest tests/driver/test_dgraph_driver.py` - new unit tests pass
5. `pytest tests/driver/test_dgraph_search.py` - search tests pass
6. `pytest tests/test_dgraph_int.py` - integration tests pass with running Dgraph (via docker-compose.dgraph.yml)
7. Quickstart example runs end-to-end with Dgraph backend
8. Agent SDK example demonstrates Graphiti-as-tool pattern
