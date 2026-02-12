"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import logging
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx
else:
    try:
        import httpx
    except ImportError:
        raise ImportError(
            'httpx is required for DgraphDriver. Install it with: pip install graphiti-core[dgraph]'
        ) from None

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider

logger = logging.getLogger(__name__)

# Maximum retries for Dgraph transaction conflicts (optimistic concurrency)
MAX_RETRIES = 3

DGRAPH_SCHEMA = """\
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
graphiti.entity_edges: string .
graphiti.episodes: string .
graphiti.attributes: string .

# ===== Vector Predicates =====
graphiti.name_embedding: float32vector @index(hnsw(metric:"cosine", exponent:"4")) .
graphiti.fact_embedding: float32vector @index(hnsw(metric:"cosine", exponent:"4")) .

# ===== Relationship Predicates (intermediate node -> endpoint) =====
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
"""


class DgraphDriverSession(GraphDriverSession):
    """Session wrapper for Dgraph HTTP API.

    Since Dgraph uses HTTP with commitNow=true, sessions are lightweight
    wrappers that delegate to the driver's HTTP client.
    """

    provider = GraphProvider.DGRAPH

    def __init__(self, driver: 'DgraphDriver'):
        self._driver = driver

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def close(self):
        pass

    async def run(self, query: str, **kwargs: Any) -> Any:
        return await self._driver.execute_query(query, **kwargs)

    async def execute_write(self, func, *args, **kwargs):
        return await func(self, *args, **kwargs)


class DgraphDriver(GraphDriver):
    """Dgraph backend driver using httpx AsyncClient.

    Communicates with Dgraph Alpha's HTTP API:
    - POST /query        — read-only DQL queries
    - POST /mutate       — upsert blocks (query + mutation)
    - POST /alter        — schema changes
    - GET  /health       — connection verification
    """

    provider = GraphProvider.DGRAPH
    default_group_id: str = ''

    def __init__(
        self,
        url: str = 'http://localhost:8080',
        api_key: str | None = None,
    ):
        super().__init__()
        self._url = url.rstrip('/')
        self._database = 'dgraph'

        headers: dict[str, str] = {}
        if api_key:
            headers['X-Dgraph-AuthToken'] = api_key

        self._client = httpx.AsyncClient(
            base_url=self._url,
            headers=headers,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            timeout=httpx.Timeout(30.0, connect=5.0),
        )

    # ------------------------------------------------------------------
    # Core HTTP helpers
    # ------------------------------------------------------------------

    async def execute_query(self, query: str, **kwargs: Any) -> Any:
        """Execute a read-only DQL query.

        Returns the parsed JSON response as a tuple (data_list, summary, keys)
        matching the GraphDriver interface convention used by Neo4j/FalkorDB.

        For Dgraph, the return is (records, None, None) where records is
        the list from the first query block, or the full data dict.
        """
        variables = kwargs.pop('variables', None)
        # Ignore Neo4j-specific kwargs like routing_
        kwargs.pop('routing_', None)

        body: dict[str, Any] = {'query': query}
        if variables:
            body['variables'] = variables

        resp = await self._client.post(
            '/query',
            content=json.dumps(body),
            headers={'Content-Type': 'application/json'},
        )
        resp.raise_for_status()
        result = resp.json()

        if 'errors' in result:
            raise DgraphQueryError(result['errors'])

        data = result.get('data', {})

        # Return first query block as records for compatibility
        if isinstance(data, dict):
            for key in data:
                if key.startswith('_') or key == 'extensions':
                    continue
                records = data[key]
                if isinstance(records, list):
                    return records, None, None
            # No list found, return full data
            return data, None, None

        return data, None, None

    async def execute_query_raw(self, query: str, variables: dict | None = None) -> dict:
        """Execute a DQL query and return the raw JSON data dict."""
        body: dict[str, Any] = {'query': query}
        if variables:
            body['variables'] = variables

        resp = await self._client.post(
            '/query',
            content=json.dumps(body),
            headers={'Content-Type': 'application/json'},
        )
        resp.raise_for_status()
        result = resp.json()

        if 'errors' in result:
            raise DgraphQueryError(result['errors'])

        return result.get('data', {})

    async def mutate(
        self,
        mutation: str,
        *,
        commit_now: bool = True,
    ) -> dict:
        """Execute a DQL upsert block (mutation with optional query).

        Uses RDF N-Quad format for mutations.
        Includes automatic retry with exponential backoff for transaction conflicts.
        """
        import asyncio

        params = 'commitNow=true' if commit_now else ''
        url = f'/mutate?{params}' if params else '/mutate'

        for attempt in range(MAX_RETRIES):
            resp = await self._client.post(
                url,
                content=mutation,
                headers={'Content-Type': 'application/rdf'},
            )
            result = resp.json()

            if 'errors' in result:
                errors = result['errors']
                # Check for transaction conflict (optimistic concurrency)
                is_conflict = any(
                    'Transaction has been aborted' in str(e.get('message', ''))
                    or 'ABORTED' in str(e.get('message', ''))
                    for e in errors
                )
                if is_conflict and attempt < MAX_RETRIES - 1:
                    wait = (2**attempt) * 0.1  # 0.1s, 0.2s, 0.4s
                    logger.warning(
                        f'Dgraph transaction conflict, retrying in {wait}s '
                        f'(attempt {attempt + 1}/{MAX_RETRIES})'
                    )
                    await asyncio.sleep(wait)
                    continue
                raise DgraphMutationError(errors)

            resp.raise_for_status()
            return result.get('data', {})

        raise DgraphMutationError([{'message': 'Max retries exceeded for transaction conflict'}])

    async def mutate_json(
        self,
        set_json: list[dict] | dict | None = None,
        delete_json: list[dict] | dict | None = None,
        *,
        commit_now: bool = True,
    ) -> dict:
        """Execute a JSON mutation (set and/or delete).

        Useful for setting vector embeddings and complex nested data.
        """
        import asyncio

        params = 'commitNow=true' if commit_now else ''
        url = f'/mutate?{params}' if params else '/mutate'

        body: dict[str, Any] = {}
        if set_json is not None:
            body['set'] = set_json
        if delete_json is not None:
            body['delete'] = delete_json

        for attempt in range(MAX_RETRIES):
            resp = await self._client.post(
                url,
                content=json.dumps(body),
                headers={'Content-Type': 'application/json'},
            )
            result = resp.json()

            if 'errors' in result:
                errors = result['errors']
                is_conflict = any(
                    'Transaction has been aborted' in str(e.get('message', ''))
                    or 'ABORTED' in str(e.get('message', ''))
                    for e in errors
                )
                if is_conflict and attempt < MAX_RETRIES - 1:
                    wait = (2**attempt) * 0.1
                    logger.warning(
                        f'Dgraph transaction conflict (JSON), retrying in {wait}s '
                        f'(attempt {attempt + 1}/{MAX_RETRIES})'
                    )
                    await asyncio.sleep(wait)
                    continue
                raise DgraphMutationError(errors)

            resp.raise_for_status()
            return result.get('data', {})

        raise DgraphMutationError([{'message': 'Max retries exceeded for transaction conflict'}])

    async def upsert(
        self,
        query: str,
        mutations: list[dict[str, Any]],
        *,
        commit_now: bool = True,
    ) -> dict:
        """Execute a DQL upsert block with query + mutations as JSON.

        Args:
            query: DQL query block for the upsert
            mutations: List of mutation objects, each with 'set_nquads' and/or 'del_nquads'
                       or 'set_json' / 'delete_json'
            commit_now: Whether to commit immediately
        """
        import asyncio

        params = 'commitNow=true' if commit_now else ''
        url = f'/mutate?{params}' if params else '/mutate'

        body: dict[str, Any] = {
            'query': query,
            'mutations': mutations,
        }

        for attempt in range(MAX_RETRIES):
            resp = await self._client.post(
                url,
                content=json.dumps(body),
                headers={'Content-Type': 'application/json'},
            )
            result = resp.json()

            if 'errors' in result:
                errors = result['errors']
                is_conflict = any(
                    'Transaction has been aborted' in str(e.get('message', ''))
                    or 'ABORTED' in str(e.get('message', ''))
                    for e in errors
                )
                if is_conflict and attempt < MAX_RETRIES - 1:
                    wait = (2**attempt) * 0.1
                    logger.warning(
                        f'Dgraph upsert conflict, retrying in {wait}s '
                        f'(attempt {attempt + 1}/{MAX_RETRIES})'
                    )
                    await asyncio.sleep(wait)
                    continue
                raise DgraphMutationError(errors)

            resp.raise_for_status()
            return result.get('data', {})

        raise DgraphMutationError([{'message': 'Max retries exceeded for transaction conflict'}])

    async def alter(self, schema: str) -> dict:
        """Execute a schema alter operation."""
        resp = await self._client.post(
            '/alter',
            content=schema,
            headers={'Content-Type': 'text/plain'},
        )
        resp.raise_for_status()
        return resp.json()

    async def drop_all(self) -> dict:
        """Drop all data and schema from Dgraph."""
        resp = await self._client.post(
            '/alter',
            content=json.dumps({'drop_all': True}),
            headers={'Content-Type': 'application/json'},
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # GraphDriver interface
    # ------------------------------------------------------------------

    def session(self, database: str | None = None) -> GraphDriverSession:
        return DgraphDriverSession(self)

    async def close(self) -> None:
        await self._client.aclose()

    def delete_all_indexes(self) -> Coroutine:
        # Dgraph indexes are managed via schema alter; drop_all removes everything
        return self.drop_all()

    async def build_indices_and_constraints(self, delete_existing: bool = False):
        if delete_existing:
            await self.drop_all()

        await self.alter(DGRAPH_SCHEMA)
        logger.info('Dgraph schema and indices created/updated')

    async def health_check(self) -> None:
        """Check Dgraph Alpha connectivity."""
        try:
            resp = await self._client.get('/health')
            resp.raise_for_status()
        except Exception as e:
            logger.error(f'Dgraph health check failed: {e}')
            raise


class DgraphQueryError(Exception):
    """Raised when a Dgraph query returns errors."""

    def __init__(self, errors: list[dict]):
        self.errors = errors
        messages = '; '.join(e.get('message', str(e)) for e in errors)
        super().__init__(f'Dgraph query error: {messages}')


class DgraphMutationError(Exception):
    """Raised when a Dgraph mutation returns errors."""

    def __init__(self, errors: list[dict]):
        self.errors = errors
        messages = '; '.join(e.get('message', str(e)) for e in errors)
        super().__init__(f'Dgraph mutation error: {messages}')
