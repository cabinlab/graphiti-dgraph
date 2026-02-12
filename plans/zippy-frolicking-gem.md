# Plan: Dgraph 25.2.0+ Backend for Graphiti

## Context

Graphiti needs Dgraph v25.2.0+ as a first-class backend. Dgraph uses DQL (not Cypher), so none of the existing ~67 Cypher query templates can be reused. The codebase already does interface-first dispatch at almost every database entry point — `nodes.py` (25 checks), `edges.py`, and `search_utils.py` (12 checks) all check `graph_operations_interface`/`search_interface` first and only fall through to Cypher when the interface isn't set.

However, 4 call sites in `graphiti.py` use hardcoded Cypher that bypasses both interfaces: saga lookup by name, previous-episode-in-saga queries (×2), and episode mention counting. These must be given interface seams upstream so non-Cypher backends can handle them. With those 3 new `GraphOperationsInterface` methods + 4 dispatch checks in `graphiti.py`, Dgraph becomes fully self-contained.

Claude Agent SDK integration is **separate scope** — not included in this plan.

---

## New Files

| File | Purpose |
|------|---------|
| `graphiti_core/driver/dgraph_driver.py` | `DgraphDriver(GraphDriver)` + `DgraphDriverSession` — httpx AsyncClient |
| `graphiti_core/driver/dgraph_search.py` | `DgraphSearchInterface(SearchInterface)` — all 14 search methods in native DQL |
| `graphiti_core/driver/dgraph_graph_operations.py` | `DgraphGraphOperations(GraphOperationsInterface)` — all CRUD methods in native DQL |
| `tests/driver/test_dgraph_driver.py` | Unit tests |
| `tests/test_dgraph_int.py` | Integration tests (mirrors `test_graphiti_int.py`) |
| `docker-compose.dgraph.yml` | Dgraph Zero + Alpha (v25.2.0) |
| `examples/quickstart/quickstart_dgraph.py` | Quickstart example |

## Core Changes

### Upstream seams: 3 new GraphOperationsInterface methods

`graphiti_core/driver/graph_operations/graph_operations.py` — add 3 methods (all default `raise NotImplementedError`, matching every existing method):

| Method | Signature (typed `Any` per interface convention) | Semantics |
|--------|--------------------------------------------------|-----------|
| `saga_node_get_by_name_and_group` | `(self, driver: Any, name: str, group_id: str) -> Any` | Find existing saga by name+group. Returns `SagaNode \| None`. |
| `get_latest_saga_episode` | `(self, driver: Any, saga_uuid: str, exclude_uuid: str \| None = None) -> Any` | Most recent episode UUID in a saga, optionally excluding one UUID. Returns `str \| None`. |
| `count_entity_episode_mentions` | `(self, driver: Any, entity_uuid: str) -> int` | How many episodes mention an entity via MENTIONS edges. |

**Compatibility**: Existing providers unchanged (Cypher fallback remains). Custom interface implementors only break if they use saga/remove_episode paths — in which case they'd need implementation anyway. Future: `count_entity_episode_mentions_bulk(driver, entity_uuids)` for batch performance.

### Upstream seams: 4 dispatch checks in graphiti.py

| Line | Current code | Change |
|------|-------------|--------|
| 345 | `_get_or_create_saga`: raw Cypher `MATCH (s:Saga {name:...})` | Add `if self.driver.graph_operations_interface:` check, call `saga_node_get_by_name_and_group`, else fall through to existing Cypher |
| 529 | `add_episode` saga previous episode: raw Cypher | Add interface check, call `get_latest_saga_episode(saga_uuid, exclude_uuid=episode.uuid)`, else fall through |
| 1195 | Bulk add saga previous episode: raw Cypher | Add interface check, call `get_latest_saga_episode(saga_uuid)`, else fall through |
| 1555 | `remove_episode` mention count: raw Cypher | Add interface check, call `count_entity_episode_mentions(entity_uuid)`, else fall through |

### Other minimal core touches

| File | Change |
|------|--------|
| `graphiti_core/driver/driver.py` | Add `DGRAPH = 'dgraph'` to `GraphProvider` enum |
| `pyproject.toml` | Add `dgraph = ["httpx>=0.27.0"]` to `[project.optional-dependencies]` |

### NOT modified

- **`graphiti_core/driver/__init__.py`** — no export (users import directly: `from graphiti_core.driver.dgraph_driver import DgraphDriver`)
- **`graphiti_core/nodes.py`** — already checks `graph_operations_interface` at every entry point (25 checks)
- **`graphiti_core/edges.py`** — already checks `graph_operations_interface` at every entry point
- **`graphiti_core/search/search_utils.py`** — already checks `search_interface` at every entry point (12 checks)
- **`graphiti_core/graph_queries.py`** — index/query builders only called in fallback paths; Dgraph's `build_indices_and_constraints` uses its own DQL schema alter
- **`graphiti_core/models/`** — query builders bypassed when interface handles the operation
- **`graphiti_core/utils/`** — already checks interfaces at entry points

---

## Dgraph Data Model

### ALL edges use intermediate nodes

Dgraph facets are fundamentally incompatible with Graphiti's edge access patterns:
- UUID-only edge lookups (`get_by_uuid`, `delete_by_uuids`) — facets require knowing the source node
- Group-based pagination (`get_by_group_ids` with UUID cursor) — facets can't be independently ordered
- Fulltext/vector search on edge properties — facets don't support these indexes

**Every edge type** is modeled as an intermediate node with `graphiti.edge_source` / `graphiti.edge_target` predicates.

### Schema

```graphql
# ===== Scalar Predicates =====
graphiti.uuid: string @index(exact) @unique .
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
graphiti.entity_edges: string .  # JSON-serialized ordered list (Dgraph lists are unordered sets)
graphiti.episodes: string .      # JSON-serialized ordered list (same reason)
graphiti.attributes: string .    # JSON-serialized dict

# ===== Vector Predicates =====
graphiti.name_embedding: float32vector @index(hnsw(metric:"cosine", exponent:"4")) .
graphiti.fact_embedding: float32vector @index(hnsw(metric:"cosine", exponent:"4")) .

# ===== Relationship Predicates (intermediate node → endpoint) =====
graphiti.edge_source: uid @reverse .
graphiti.edge_target: uid @reverse .

# ===== Node Types =====
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
}

type Community {
    graphiti.uuid
    graphiti.name
    graphiti.group_id
    graphiti.summary
    graphiti.created_at
    graphiti.name_embedding
}

type Saga {
    graphiti.uuid
    graphiti.name
    graphiti.group_id
    graphiti.created_at
}

# ===== Edge Types (ALL as intermediate nodes) =====
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
    graphiti.edge_source
    graphiti.edge_target
}

type MentionsEdge {
    graphiti.uuid
    graphiti.group_id
    graphiti.created_at
    graphiti.edge_source
    graphiti.edge_target
}

type HasMemberEdge {
    graphiti.uuid
    graphiti.group_id
    graphiti.created_at
    graphiti.edge_source
    graphiti.edge_target
}

type HasEpisodeEdge {
    graphiti.uuid
    graphiti.group_id
    graphiti.created_at
    graphiti.edge_source
    graphiti.edge_target
}

type NextEpisodeEdge {
    graphiti.uuid
    graphiti.group_id
    graphiti.created_at
    graphiti.edge_source
    graphiti.edge_target
}
```

**Design notes:**
- **`@unique` on uuid**: Database-enforced uniqueness (v24.0+), safety net on top of upsert blocks
- **`exponent:"4"`** (quoted string): Tunes HNSW for ~10,000 vectors; increase to `"5"` or `"6"` for larger embedding collections
- **Ordered lists as JSON strings**: `graphiti.episodes` and `graphiti.entity_edges` are stored as `json.dumps(list)` because Dgraph `[string]` is an unordered set
- **Shared predicates are idiomatic DQL**: All queries MUST include `@filter(type(...))` to avoid cross-type UUID collisions

---

## DQL Query Patterns

### Node upsert
```dql
upsert {
  query { v as var(func: eq(graphiti.uuid, "node-uuid")) @filter(type(Entity)) }
  mutation {
    set {
      uid(v) <dgraph.type> "Entity" .
      uid(v) <graphiti.uuid> "node-uuid" .
      uid(v) <graphiti.name> "Alice" .
      uid(v) <graphiti.group_id> "group-1" .
    }
  }
}
```
Vectors passed as JSON arrays via JSON mutation format. `@filter(type(...))` prevents cross-type UUID collisions.

### Edge upsert (resolves source/target UIDs in same query block)
```dql
upsert {
  query {
    edge_node as var(func: eq(graphiti.uuid, "edge-uuid")) @filter(type(RelatesToEdge))
    src as var(func: eq(graphiti.uuid, "source-uuid")) @filter(type(Entity))
    tgt as var(func: eq(graphiti.uuid, "target-uuid")) @filter(type(Entity))
  }
  mutation {
    set {
      uid(edge_node) <dgraph.type> "RelatesToEdge" .
      uid(edge_node) <graphiti.uuid> "edge-uuid" .
      uid(edge_node) <graphiti.edge_source> uid(src) .
      uid(edge_node) <graphiti.edge_target> uid(tgt) .
      uid(edge_node) <graphiti.fact> "some fact" .
    }
  }
}
```

### Delete (node or edge)
```dql
upsert {
  query { v as var(func: eq(graphiti.uuid, "uuid")) @filter(type(Entity)) }
  mutation {
    delete { uid(v) * * . }
  }
}
```
**Invariant**: `S * *` only removes predicates declared in the node's `dgraph.type`. All predicates written to a node MUST be declared in its type definition, or deletion will leave orphaned data.

### Bulk delete by group_id
```dql
upsert {
  query { nodes as var(func: eq(graphiti.group_id, "group-1")) }
  mutation {
    delete { uid(nodes) * * . }
  }
}
```

### Vector similarity search
```dql
query search($vec: float32vector) {
  q(func: similar_to(graphiti.name_embedding, 20, $vec, ef: 64, distance_threshold: 0.3), first: 10)
    @filter(type(Entity) AND eq(graphiti.group_id, "group-1")) {
    uid graphiti.uuid graphiti.name graphiti.summary
  }
}
```
**Critical**: v25.2.0 cosine distance is raw [0, 2] (not normalized to [0,1] as in pre-v25.2 versions). Conversion: `distance_threshold = 1.0 - min_score`. Request more than K from `similar_to()` since `@filter` may remove results.

### Fulltext search (composite name + summary)
```dql
{
  # eq() with a list matches ANY value in the list (OR semantics)
  by_name as var(func: anyoftext(graphiti.name, "search terms"))
    @filter(type(Entity) AND eq(graphiti.group_id, "group-1"))
  by_summary as var(func: anyoftext(graphiti.summary, "search terms"))
    @filter(type(Entity) AND eq(graphiti.group_id, "group-1"))
  # For multiple groups: eq(graphiti.group_id, ["group-1", "group-2"])

  results(func: uid(by_name, by_summary), first: 100) {
    uid graphiti.uuid graphiti.name graphiti.summary
  }
}
```
DQL does NOT support OR at the root `func:`. Variable block union is the correct pattern.

### BFS traversal (explicit variable blocks)
```dql
{
  seeds as var(func: eq(graphiti.uuid, ["uuid-1", "uuid-2"]))

  # Hop 1: edges connected to seeds
  var(func: uid(seeds)) {
    out as ~graphiti.edge_source
    in as ~graphiti.edge_target
  }
  hop1_edges as var(func: uid(out, in))

  # Hop 1: entities at other end
  var(func: uid(hop1_edges)) {
    t1 as graphiti.edge_target
    t2 as graphiti.edge_source
  }
  hop1_nodes as var(func: uid(t1, t2))

  # Hop 2 (from hop1 nodes)
  var(func: uid(hop1_nodes)) {
    e2_out as ~graphiti.edge_source
    e2_in as ~graphiti.edge_target
  }
  hop2_edges as var(func: uid(e2_out, e2_in))

  results(func: uid(hop1_edges, hop2_edges))
    @filter(type(RelatesToEdge)) {
    uid graphiti.uuid graphiti.name graphiti.fact
    graphiti.edge_source { uid graphiti.uuid graphiti.name }
    graphiti.edge_target { uid graphiti.uuid graphiti.name }
  }
}
```
Each logical hop = 2 Dgraph hops through intermediate nodes. For variable depth, generate query programmatically.

### Shortest path (for distance reranker)
```dql
{
  center as var(func: eq(graphiti.uuid, "center-uuid"))
  target as var(func: eq(graphiti.uuid, "target-uuid"))

  path as shortest(from: uid(center), to: uid(target), numpaths: 1) {
    graphiti.edge_source
    graphiti.edge_target
    ~graphiti.edge_source
    ~graphiti.edge_target
  }

  result(func: uid(path)) {
    uid graphiti.uuid graphiti.name
  }
}
```
Path length / 2 = logical hop count.

---

## SearchInterface: 14 Methods

12 async + 2 sync, matching `graphiti_core/driver/search_interface/search_interface.py` exactly:

| # | Method | DQL Pattern |
|---|--------|-------------|
| 1 | `edge_fulltext_search` | `alloftext(graphiti.fact, $q)` + `type(RelatesToEdge)` |
| 2 | `edge_similarity_search` | `similar_to(graphiti.fact_embedding, $k, $vec)` + type filter |
| 3 | `node_fulltext_search` | Variable block union: `anyoftext(name)` + `anyoftext(summary)` |
| 4 | `node_similarity_search` | `similar_to(graphiti.name_embedding, $k, $vec)` + `type(Entity)` |
| 5 | `episode_fulltext_search` | `alloftext(graphiti.content, $q)` + `type(Episodic)` |
| 6 | `edge_bfs_search` | Explicit variable block traversal through intermediate nodes |
| 7 | `node_bfs_search` | Same pattern, collecting entity nodes at each hop |
| 8 | `community_fulltext_search` | `alloftext(graphiti.name, $q)` + `type(Community)` |
| 9 | `community_similarity_search` | `similar_to(graphiti.name_embedding, $k, $vec)` + `type(Community)` |
| 10 | `get_embeddings_for_communities` | Load `graphiti.name_embedding` from Community nodes by UUID list → `dict[str, list[float]]` |
| 11 | `node_distance_reranker` | `shortest()` with path length / 2 |
| 12 | `episode_mentions_reranker` | For each entity node, count MentionsEdge nodes targeting it via `~graphiti.edge_target` |
| 13 | `build_node_search_filters` (sync) | Construct DQL `@filter()` clauses from SearchFilter config |
| 14 | `build_edge_search_filters` (sync) | Construct DQL `@filter()` clauses from SearchFilter config |

---

## HTTP Client Design

`httpx.AsyncClient` for Dgraph's HTTP API:

| Endpoint | Content-Type | Driver Method |
|----------|-------------|---------------|
| `POST /query` | `application/dql` | `execute_query()` — read-only DQL |
| `POST /mutate?commitNow=true` | `application/rdf` | `_mutate()` — upsert blocks (query+mutation) |
| `POST /alter` | text | `_alter()` — schema changes |
| `GET /health` | — | Connection verification |

**Query variables** passed as JSON: `{"query": "...", "variables": {"$vec": "[0.1, 0.2]"}}`

**Connection pooling**: `httpx.Limits(max_connections=100, max_keepalive_connections=20)`, `httpx.Timeout(30.0, connect=5.0)`

**Transaction retry**: Dgraph uses optimistic concurrency. Wrap mutations with automatic retry + exponential backoff (3 retries) for `ABORTED` errors.

**Response parsing**: Queries return `{"data": {"q": [...]}}`, mutations return `{"data": {"uids": {...}}}`, errors return `{"errors": [...]}`.

---

## Community Detection

Dgraph has no native graph algorithms. `get_community_clusters` must:
1. Fetch entities and edges for the given group_ids via DQL
2. Build adjacency list in Python
3. Run label propagation application-side
4. Return clusters

---

## Implementation Order

### Step 1: Upstream seams + DgraphDriver foundation
1. Add 3 methods to `GraphOperationsInterface` in `graph_operations.py` (all `raise NotImplementedError`)
2. Add 4 interface-first dispatch checks in `graphiti.py` at lines 345, 529, 1195, 1555
3. Add `DGRAPH = 'dgraph'` enum value in `driver.py`, `dgraph` optional dep in `pyproject.toml`
4. `dgraph_driver.py` — httpx connection, `execute_query`, `_mutate` (with retry), `_alter`, `build_indices_and_constraints` with full schema, health check

### Step 2: GraphOperationsInterface
`dgraph_graph_operations.py`:
- Node CRUD: Entity, Episodic, Community, Saga (save, delete, get_by_uuid, get_by_uuids, get_by_group_ids, bulk ops, load_embeddings)
- Edge CRUD: all 5 edge types with intermediate node pattern
- Saga/mention seams: `saga_node_get_by_name_and_group`, `get_latest_saga_episode`, `count_entity_episode_mentions`
- Maintenance: clear_data, community detection, get_mentioned_nodes

### Step 3: SearchInterface
`dgraph_search.py` — all 14 methods: fulltext (variable block union), vector similarity (query variables, distance conversion), BFS (explicit variable blocks), community embeddings, rerankers, filter builders.

### Step 4: Testing & examples
- Unit tests, integration tests, docker-compose, quickstart example

---

## Verification

1. `make check` passes (new backend doesn't break existing)
2. `pytest tests/driver/test_dgraph_driver.py` — driver unit tests
3. `pytest tests/test_dgraph_int.py` — integration tests with Dockerized Dgraph
4. Quickstart example runs end-to-end

---

## Reference

- `dgraph-backend-scope.md` — scope analysis
- `plans/keen-crafting-pearl.md` — original research team plan (superseded)
- `plans/zippy-frolicking-gem-agent-a97810a.md` — dgraph-expert review with DQL corrections
- `dgraph-planning/inboxes/` — raw research from 6 agents
