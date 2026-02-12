# Final Dgraph Expert Review of `zippy-frolicking-gem.md`

**Reviewer**: Dgraph v25.2.0+ specialist (second pass)
**Date**: 2026-02-11
**Overall assessment**: The plan is solid and well-structured. The previous 16 corrections were incorporated well. I found **4 issues** remaining -- 1 high severity, 2 medium severity, 1 low severity. None are architectural; all are fixable in implementation.

---

## Issue 1: HNSW `exponent` parameter uses wrong value type

**Severity: HIGH**

**Location**: Schema definition, line 100-101

**Current** (line 100-101):
```
graphiti.name_embedding: float32vector @index(hnsw(metric:"cosine", exponent:4)) .
graphiti.fact_embedding: float32vector @index(hnsw(metric:"cosine", exponent:4)) .
```

**Problem**: The `exponent` parameter value must be a **quoted string**, not a bare integer. Dgraph's HNSW parameter parser expects all values as strings. Per the Dgraph docs and schema examples:

```
large_vector: float32vector @index(hnsw(metric:"euclidean",exponent:"6")) .
```

**Fix**:
```
graphiti.name_embedding: float32vector @index(hnsw(metric:"cosine", exponent:"4")) .
graphiti.fact_embedding: float32vector @index(hnsw(metric:"cosine", exponent:"4")) .
```

The schema alter operation will fail at runtime if the unquoted integer form is used, because Dgraph's schema parser expects string-valued parameters inside `hnsw(...)`.

**Also**: The plan's design note (line 199) says `exponent:4` means `M=16 HNSW neighbors`. This is misleading. `exponent:"4"` means the index is tuned for approximately 10^4 (10,000) vectors. It sets internal HNSW parameters (M, efConstruction) to reasonable defaults for that dataset size. It does not directly mean M=16. The note should say: "tuned for ~10,000 vectors; increase to `exponent:"5"` or `exponent:"6"` for larger embedding collections."

---

## Issue 2: HTTP endpoint for upsert blocks is `/mutate`, not `/query`

**Severity: MEDIUM**

**Location**: HTTP Client Design table, line 378

**Current** (line 377-379):
```
| `POST /query` | `application/dql` | `execute_query()` |
| `POST /mutate?commitNow=true` | `application/json` | `_mutate()` |
```

**Problem**: DQL upsert blocks (which combine query + mutation in one request) must be sent to the `/mutate` endpoint, NOT `/query`. The plan's table implies `execute_query()` maps to `/query`, but upsert blocks -- which are the primary write path for all node and edge saves -- go to `/mutate`.

Specifically, when using the RDF format with a full `upsert { query { ... } mutation { ... } }` block, the request goes to:
```
POST /mutate?commitNow=true
Content-Type: application/rdf
```

When using JSON format for upserts, it goes to:
```
POST /mutate?commitNow=true
Content-Type: application/json
Body: {"query": "...", "set": {...}}
```

Read-only DQL queries (no mutations) go to:
```
POST /query
Content-Type: application/dql
```

**Recommendation**: The driver should have three internal methods:
1. `_query(dql)` -- read-only queries to `POST /query` with `Content-Type: application/dql`
2. `_mutate_upsert(upsert_block)` -- upsert blocks to `POST /mutate?commitNow=true` with `Content-Type: application/rdf`
3. `_alter(schema)` -- schema changes to `POST /alter`

The public `execute_query()` inherited from `GraphDriver` can dispatch to `_query` or `_mutate_upsert` depending on whether the DQL string starts with `upsert {`. This is an implementation detail, but the plan's table should not imply upserts go through `/query`.

---

## Issue 3: `delete { uid(v) * * . }` only deletes predicates declared in the node's type

**Severity: MEDIUM**

**Location**: Delete pattern, lines 243-251

**Current** (lines 244-251):
```dql
upsert {
  query { v as var(func: eq(graphiti.uuid, "uuid")) @filter(type(Entity)) }
  mutation {
    delete { uid(v) * * . }
  }
}
```

**Problem**: The `S * *` delete pattern only removes predicates that are declared in the node's `dgraph.type`. This is fine for the plan's schema **as long as every predicate stored on a node is declared in its type definition**. The current type definitions look correct for this -- e.g., `type Entity` lists all predicates an Entity node would have, `type RelatesToEdge` lists all predicates a RelatesToEdge intermediate node would have.

However, this becomes a subtle bug if:
1. A predicate is written to a node but not declared in its type (e.g., via a code bug or schema evolution)
2. The shared predicates `graphiti.edge_source` and `graphiti.edge_target` are on edge-type nodes but if the `dgraph.type` doesn't include them, they won't be deleted

Checking the schema: `RelatesToEdge` type DOES include `graphiti.edge_source` and `graphiti.edge_target`. The other 4 edge types (`MentionsEdge`, `HasMemberEdge`, `HasEpisodeEdge`, `NextEpisodeEdge`) also include these predicates. So this is currently correct.

**Recommendation**: Add a brief note to the plan acknowledging this behavior. Something like: "The `S * *` delete only removes predicates declared in the node's `dgraph.type`. All predicates written to a node MUST be declared in its type definition, or deletion will leave orphaned predicates. The schema above satisfies this requirement."

This protects future implementors from accidentally adding a predicate to a mutation without adding it to the type definition.

---

## Issue 4: Cosine distance range statement is slightly imprecise

**Severity: LOW**

**Location**: Vector similarity search section, line 272

**Current**:
```
**Critical**: v25.2.0 cosine distance is raw [0, 2] (not [0,1]). Conversion: `distance_threshold = 1.0 - min_score`.
```

**Problem**: The conversion formula `distance_threshold = 1.0 - min_score` is only correct when `min_score` is a cosine similarity in [0, 1]. Raw cosine distance = 1 - cosine_similarity, so:
- cosine_similarity = 1.0 (identical) => distance = 0.0
- cosine_similarity = 0.0 (orthogonal) => distance = 1.0
- cosine_similarity = -1.0 (opposite) => distance = 2.0

The formula is correct. The range [0, 2] is correct. But the Graphiti `min_score` parameter (from `SearchInterface.edge_similarity_search` and `node_similarity_search`) is documented as "Minimum similarity score threshold (0.0 to 1.0)". If `min_score=0.7`, then `distance_threshold = 1.0 - 0.7 = 0.3`, which means "return results with cosine distance <= 0.3", i.e., cosine similarity >= 0.7. This is correct.

No fix needed on the formula. The note is accurate. However, the parenthetical "(not [0,1])" could be clearer -- it should say "(not normalized to [0,1] as in pre-v25.2 versions)" to avoid confusion about what changed.

---

## Items Verified as Correct

### DQL Syntax
- **Node upsert** (lines 208-220): Correct. Multiple `var()` blocks in a single `query {}` block is valid DQL. The `@filter(type(Entity))` placement is correct.
- **Edge upsert** (lines 224-241): Correct. Three variable blocks (`edge_node`, `src`, `tgt`) in one query block, then `uid(src)` / `uid(tgt)` references in the mutation. Valid DQL.
- **Vector similarity** (lines 264-271): `similar_to()` syntax with named parameters `ef:` and `distance_threshold:` is correct for v25.2.0+. The `@filter` placement after `similar_to()` in the root function is correct.
- **Fulltext search** (lines 275-289): Variable block union pattern is the correct way to OR two root functions in DQL. `anyoftext()` is the right function for partial term matching.
- **BFS traversal** (lines 291-324): Explicit variable block pattern with `~graphiti.edge_source` (reverse traversal) is correct. The two-Dgraph-hop = one-logical-hop math is correct.
- **Shortest path** (lines 328-344): `shortest()` function with both forward and reverse predicates is correct. `numpaths: 1` is valid.
- **Bulk delete** (lines 253-261): `eq(graphiti.group_id, "group-1")` without type filter is intentional -- deletes all node types in that group. Correct for `clear_data`.

### Schema
- **`@unique` on `graphiti.uuid`**: Correct. `@unique` works on `string` with `exact` index (v24.0+). Dgraph auto-adds `@upsert`.
- **`@reverse` on `graphiti.edge_source` and `graphiti.edge_target`**: Correct. These are `uid` predicates (not list `[uid]`), which is fine since each intermediate edge node points to exactly one source and one target. `@reverse` enables `~graphiti.edge_source` traversal.
- **Shared predicates across types**: Correct and idiomatic DQL. All queries include `@filter(type(...))` to prevent cross-type collisions.
- **JSON-serialized ordered lists**: Correct approach for `graphiti.episodes` and `graphiti.entity_edges` since Dgraph `[string]` is an unordered set.

### Intermediate Node Pattern
- All 5 edge types (RelatesToEdge, MentionsEdge, HasMemberEdge, HasEpisodeEdge, NextEpisodeEdge) consistently use intermediate nodes with `graphiti.edge_source` / `graphiti.edge_target`. Correct.
- The decision to use intermediate nodes for ALL edges (not just RelatesToEdge) is sound -- it avoids two different access patterns and makes UUID-based edge lookup uniform.

### New Upstream Seams
- **`saga_node_get_by_name_and_group(name, group_id)`**: DQL is straightforward:
  ```dql
  { q(func: eq(graphiti.name, $name)) @filter(type(Saga) AND eq(graphiti.group_id, $group_id)) { ... } }
  ```
  Uses the `exact` index on both `graphiti.name` and `graphiti.group_id`. Correct.

- **`get_latest_saga_episode(saga_uuid, exclude_uuid=None)`**: DQL requires two hops through intermediate `HasEpisodeEdge` nodes:
  ```dql
  {
    saga as var(func: eq(graphiti.uuid, $saga_uuid)) @filter(type(Saga))
    var(func: uid(saga)) {
      ~graphiti.edge_source @filter(type(HasEpisodeEdge)) {
        episodes as graphiti.edge_target
      }
    }
    q(func: uid(episodes), orderdesc: graphiti.valid_at, first: 1)
      @filter(type(Episodic) AND NOT eq(graphiti.uuid, $exclude_uuid)) {
      graphiti.uuid
    }
  }
  ```
  This is more complex than the Cypher equivalent but entirely feasible. The `orderdesc` on `graphiti.valid_at` requires the `hour` index, which is present. Correct.

- **`count_entity_episode_mentions(entity_uuid)`**: DQL uses reverse traversal through `MentionsEdge`:
  ```dql
  {
    entity as var(func: eq(graphiti.uuid, $entity_uuid)) @filter(type(Entity))
    var(func: uid(entity)) {
      ~graphiti.edge_target @filter(type(MentionsEdge)) {
        mention_sources as graphiti.edge_source
      }
    }
    q(func: uid(mention_sources)) @filter(type(Episodic)) {
      count(uid)
    }
  }
  ```
  Or more simply using `count()` aggregation. Straightforward given the schema. Correct.

### Completeness
- The 14 SearchInterface methods are all accounted for with appropriate DQL patterns.
- The GraphOperationsInterface method inventory (from reading the actual interface at `/mnt/b/github/graphiti/graphiti_core/driver/graph_operations/graph_operations.py`) has ~60+ methods. The plan covers all categories: node CRUD (Entity, Episodic, Community, Saga), edge CRUD (5 types), embeddings, search helpers, maintenance, and community detection.
- The 4 upstream `graphiti.py` dispatch points were verified against the actual source code at lines 345, 529, 1195, and 1555. All 4 are correctly identified raw Cypher calls that need interface seams.

### v25.2.0 Specifics
- `similar_to()` with `ef:` and `distance_threshold:` named parameters: correct for v25.2.0+.
- `@unique` directive: correct syntax and semantics.
- HTTP endpoints: `/query`, `/mutate`, `/alter` -- correct (with the caveat in Issue 2 about upserts).

---

## Summary

| # | Issue | Severity | Fix |
|---|-------|----------|-----|
| 1 | HNSW `exponent` must be quoted string `"4"` not bare integer `4` | HIGH | Change `exponent:4` to `exponent:"4"` in schema; fix design note |
| 2 | Upsert blocks go to `/mutate` not `/query`; plan's HTTP table is ambiguous | MEDIUM | Clarify that `execute_query()` dispatches to different endpoints for reads vs writes |
| 3 | `S * *` delete only removes type-declared predicates; needs documentation note | MEDIUM | Add implementation note about keeping type definitions in sync with written predicates |
| 4 | Cosine distance note could be clearer about what changed in v25.2.0 | LOW | Minor wording improvement |

The plan is ready for implementation after these 4 fixes. The architecture is sound, the DQL patterns are correct, and the upstream seam analysis is verified against the actual codebase.
