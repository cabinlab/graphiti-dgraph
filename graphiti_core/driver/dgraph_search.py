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
from collections import defaultdict
from datetime import datetime
from typing import Any

from graphiti_core.driver.search_interface.search_interface import SearchInterface
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodeType, EpisodicNode
from graphiti_core.search.search_filters import (
    ComparisonOperator,
    SearchFilters,
)

logger = logging.getLogger(__name__)

# Dgraph v25.2.0 cosine distance is raw [0, 2] (not normalized).
# Conversion: distance_threshold = 1.0 - min_score
# We over-fetch from similar_to since @filter may remove results.
_VECTOR_OVERFETCH_FACTOR = 3


def _format_group_id_filter(group_ids: list[str] | None) -> str:
    """Build a DQL eq() clause for group_id filtering.

    eq(graphiti.group_id, ["g1", "g2"]) matches ANY value in the list (OR semantics).
    """
    if not group_ids:
        return ''
    escaped = ', '.join(f'"{g}"' for g in group_ids)
    return f'eq(graphiti.group_id, [{escaped}])'


def _format_vec_json(vec: list[float]) -> str:
    """Format a vector as a JSON string for DQL variables."""
    return '[' + ', '.join(str(v) for v in vec) + ']'


def _parse_datetime(val: Any) -> datetime | None:
    """Parse a datetime value from Dgraph (ISO format string)."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        if not val or val == '':
            return None
        return datetime.fromisoformat(val.replace('Z', '+00:00'))
    return None


def _parse_json_field(val: Any, default: Any = None) -> Any:
    """Parse a JSON-serialized string field from Dgraph."""
    if val is None:
        return default
    if isinstance(val, str):
        if not val:
            return default
        return json.loads(val)
    return val


def _entity_node_from_record(record: dict) -> EntityNode:
    """Convert a Dgraph result dict to an EntityNode."""
    labels = record.get('graphiti.labels', []) or []
    if 'Entity' in labels:
        labels = [lbl for lbl in labels if lbl != 'Entity']

    return EntityNode(
        uuid=record['graphiti.uuid'],
        name=record.get('graphiti.name', ''),
        group_id=record.get('graphiti.group_id', ''),
        labels=labels,
        created_at=_parse_datetime(record.get('graphiti.created_at')) or datetime.min,
        name_embedding=None,
        summary=record.get('graphiti.summary', ''),
        attributes=_parse_json_field(record.get('graphiti.attributes'), {}),
    )


def _entity_edge_from_record(record: dict) -> EntityEdge:
    """Convert a Dgraph result dict to an EntityEdge."""
    source_uuid = ''
    target_uuid = ''

    src = record.get('graphiti.edge_source')
    if isinstance(src, list) and src:
        source_uuid = src[0].get('graphiti.uuid', '')
    elif isinstance(src, dict):
        source_uuid = src.get('graphiti.uuid', '')

    tgt = record.get('graphiti.edge_target')
    if isinstance(tgt, list) and tgt:
        target_uuid = tgt[0].get('graphiti.uuid', '')
    elif isinstance(tgt, dict):
        target_uuid = tgt.get('graphiti.uuid', '')

    return EntityEdge(
        uuid=record['graphiti.uuid'],
        group_id=record.get('graphiti.group_id', ''),
        source_node_uuid=source_uuid,
        target_node_uuid=target_uuid,
        created_at=_parse_datetime(record.get('graphiti.created_at')) or datetime.min,
        name=record.get('graphiti.name', ''),
        fact=record.get('graphiti.fact', ''),
        fact_embedding=None,
        episodes=_parse_json_field(record.get('graphiti.episodes'), []),
        expired_at=_parse_datetime(record.get('graphiti.expired_at')),
        valid_at=_parse_datetime(record.get('graphiti.valid_at')),
        invalid_at=_parse_datetime(record.get('graphiti.invalid_at')),
        attributes=_parse_json_field(record.get('graphiti.attributes'), {}),
    )


def _episodic_node_from_record(record: dict) -> EpisodicNode:
    """Convert a Dgraph result dict to an EpisodicNode."""
    return EpisodicNode(
        uuid=record['graphiti.uuid'],
        name=record.get('graphiti.name', ''),
        group_id=record.get('graphiti.group_id', ''),
        source=EpisodeType.from_str(record.get('graphiti.source', 'text')),
        source_description=record.get('graphiti.source_description', ''),
        content=record.get('graphiti.content', ''),
        created_at=_parse_datetime(record.get('graphiti.created_at')) or datetime.min,
        valid_at=_parse_datetime(record.get('graphiti.valid_at')) or datetime.min,
        entity_edges=_parse_json_field(record.get('graphiti.entity_edges'), []),
    )


def _community_node_from_record(record: dict) -> CommunityNode:
    """Convert a Dgraph result dict to a CommunityNode."""
    return CommunityNode(
        uuid=record['graphiti.uuid'],
        name=record.get('graphiti.name', ''),
        group_id=record.get('graphiti.group_id', ''),
        created_at=_parse_datetime(record.get('graphiti.created_at')) or datetime.min,
        name_embedding=None,
        summary=record.get('graphiti.summary', ''),
    )


# ---------------------------------------------------------------------------
# Predicate return fragments for DQL queries
# ---------------------------------------------------------------------------
_ENTITY_NODE_FIELDS = (
    'uid graphiti.uuid graphiti.name graphiti.group_id graphiti.summary'
    ' graphiti.created_at graphiti.labels graphiti.attributes'
)

_ENTITY_EDGE_FIELDS = (
    'uid graphiti.uuid graphiti.name graphiti.fact graphiti.group_id'
    ' graphiti.created_at graphiti.expired_at graphiti.valid_at graphiti.invalid_at'
    ' graphiti.episodes graphiti.attributes'
    ' graphiti.edge_source { graphiti.uuid }'
    ' graphiti.edge_target { graphiti.uuid }'
)

_EPISODIC_NODE_FIELDS = (
    'uid graphiti.uuid graphiti.name graphiti.group_id graphiti.source'
    ' graphiti.source_description graphiti.content graphiti.created_at'
    ' graphiti.valid_at graphiti.entity_edges'
)

_COMMUNITY_NODE_FIELDS = (
    'uid graphiti.uuid graphiti.name graphiti.group_id graphiti.summary graphiti.created_at'
)


class DgraphSearchInterface(SearchInterface):
    """SearchInterface implementation for Dgraph v25 using DQL."""

    # ------------------------------------------------------------------
    # Fulltext searches
    # ------------------------------------------------------------------

    async def edge_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        if not query.strip():
            return []

        group_filter = _format_group_id_filter(group_ids)
        extra_filter = self.build_edge_search_filters(search_filter)

        type_filter = 'type(RelatesToEdge)'
        filter_parts = [type_filter]
        if group_filter:
            filter_parts.append(group_filter)
        if extra_filter:
            filter_parts.append(extra_filter.removeprefix(' AND '))

        combined_filter = ' AND '.join(filter_parts)

        dql = '{\n'
        dql += (
            f'  by_fact as var(func: anyoftext(graphiti.fact, {json.dumps(query)}))\n'
            f'    @filter({combined_filter})\n'
        )
        dql += (
            f'  by_name as var(func: anyoftext(graphiti.name, {json.dumps(query)}))\n'
            f'    @filter({combined_filter})\n'
        )
        dql += (
            f'  results(func: uid(by_fact, by_name), first: {limit}) {{\n'
            f'    {_ENTITY_EDGE_FIELDS}\n'
            f'  }}\n'
        )
        dql += '}'

        data = await driver.execute_query_raw(dql)
        records = data.get('results', [])
        return [_entity_edge_from_record(r) for r in records]

    async def edge_similarity_search(
        self,
        driver: Any,
        search_vector: list[float],
        source_node_uuid: str | None,
        target_node_uuid: str | None,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
        min_score: float = 0.7,
    ) -> list[Any]:
        group_filter = _format_group_id_filter(group_ids)
        extra_filter = self.build_edge_search_filters(search_filter)

        type_filter = 'type(RelatesToEdge)'
        filter_parts = [type_filter]
        if group_filter:
            filter_parts.append(group_filter)
        if extra_filter:
            filter_parts.append(extra_filter.removeprefix(' AND '))
        combined_filter = ' AND '.join(filter_parts)

        overfetch = limit * _VECTOR_OVERFETCH_FACTOR
        vec_str = _format_vec_json(search_vector)
        distance_threshold = max(0.0, min(2.0, 1.0 - min_score))

        dql = '{\n'
        dql += (
            f'  results(func: similar_to(graphiti.fact_embedding, {overfetch}, $vec,'
            f' distance_threshold: {distance_threshold}),'
            f' first: {limit})\n'
            f'    @filter({combined_filter}) {{\n'
            f'    {_ENTITY_EDGE_FIELDS}\n'
            f'  }}\n'
        )
        dql += '}'

        variables = {'$vec': vec_str}
        data = await driver.execute_query_raw(dql, variables=variables)
        records = data.get('results', [])

        edges = [_entity_edge_from_record(r) for r in records]

        # Post-filter by source/target node UUIDs if specified
        if source_node_uuid is not None:
            edges = [e for e in edges if e.source_node_uuid == source_node_uuid]
        if target_node_uuid is not None:
            edges = [e for e in edges if e.target_node_uuid == target_node_uuid]

        return edges[:limit]

    async def node_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        if not query.strip():
            return []

        group_filter = _format_group_id_filter(group_ids)
        extra_filter = self.build_node_search_filters(search_filter)

        type_filter = 'type(Entity)'
        filter_parts = [type_filter]
        if group_filter:
            filter_parts.append(group_filter)
        if extra_filter:
            filter_parts.append(extra_filter.removeprefix(' AND '))
        combined_filter = ' AND '.join(filter_parts)

        dql = '{\n'
        dql += (
            f'  by_name as var(func: anyoftext(graphiti.name, {json.dumps(query)}))\n'
            f'    @filter({combined_filter})\n'
        )
        dql += (
            f'  by_summary as var(func: anyoftext(graphiti.summary, {json.dumps(query)}))\n'
            f'    @filter({combined_filter})\n'
        )
        dql += (
            f'  results(func: uid(by_name, by_summary), first: {limit}) {{\n'
            f'    {_ENTITY_NODE_FIELDS}\n'
            f'  }}\n'
        )
        dql += '}'

        data = await driver.execute_query_raw(dql)
        records = data.get('results', [])
        return [_entity_node_from_record(r) for r in records]

    async def node_similarity_search(
        self,
        driver: Any,
        search_vector: list[float],
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
        min_score: float = 0.7,
    ) -> list[Any]:
        group_filter = _format_group_id_filter(group_ids)
        extra_filter = self.build_node_search_filters(search_filter)

        type_filter = 'type(Entity)'
        filter_parts = [type_filter]
        if group_filter:
            filter_parts.append(group_filter)
        if extra_filter:
            filter_parts.append(extra_filter.removeprefix(' AND '))
        combined_filter = ' AND '.join(filter_parts)

        overfetch = limit * _VECTOR_OVERFETCH_FACTOR
        vec_str = _format_vec_json(search_vector)
        distance_threshold = max(0.0, min(2.0, 1.0 - min_score))

        dql = '{\n'
        dql += (
            f'  results(func: similar_to(graphiti.name_embedding, {overfetch}, $vec,'
            f' distance_threshold: {distance_threshold}),'
            f' first: {limit})\n'
            f'    @filter({combined_filter}) {{\n'
            f'    {_ENTITY_NODE_FIELDS}\n'
            f'  }}\n'
        )
        dql += '}'

        variables = {'$vec': vec_str}
        data = await driver.execute_query_raw(dql, variables=variables)
        records = data.get('results', [])
        return [_entity_node_from_record(r) for r in records]

    async def episode_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        if not query.strip():
            return []

        group_filter = _format_group_id_filter(group_ids)
        type_filter = 'type(Episodic)'
        filter_parts = [type_filter]
        if group_filter:
            filter_parts.append(group_filter)
        combined_filter = ' AND '.join(filter_parts)

        dql = '{\n'
        dql += (
            f'  by_content as var(func: anyoftext(graphiti.content, {json.dumps(query)}))\n'
            f'    @filter({combined_filter})\n'
        )
        dql += (
            f'  results(func: uid(by_content), first: {limit}) {{\n'
            f'    {_EPISODIC_NODE_FIELDS}\n'
            f'  }}\n'
        )
        dql += '}'

        data = await driver.execute_query_raw(dql)
        records = data.get('results', [])
        return [_episodic_node_from_record(r) for r in records]

    # ------------------------------------------------------------------
    # BFS searches
    # ------------------------------------------------------------------

    async def edge_bfs_search(
        self,
        driver: Any,
        bfs_origin_node_uuids: list[str] | None,
        bfs_max_depth: int,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        if not bfs_origin_node_uuids or bfs_max_depth < 1:
            return []

        group_filter = _format_group_id_filter(group_ids)
        extra_filter = self.build_edge_search_filters(search_filter)

        result_filter_parts = ['type(RelatesToEdge)']
        if group_filter:
            result_filter_parts.append(group_filter)
        if extra_filter:
            result_filter_parts.append(extra_filter.removeprefix(' AND '))
        result_filter = ' AND '.join(result_filter_parts)

        uuid_list = ', '.join(f'"{u}"' for u in bfs_origin_node_uuids)

        dql = '{\n'
        dql += f'  seeds as var(func: eq(graphiti.uuid, [{uuid_list}]))\n\n'

        # Build programmatic BFS hops through intermediate edge nodes.
        # Each logical hop = 2 Dgraph hops (entity -> edge_node -> entity).
        prev_nodes_var = 'seeds'
        all_edge_vars: list[str] = []

        for hop in range(1, bfs_max_depth + 1):
            out_var = f'out_{hop}'
            in_var = f'in_{hop}'
            edge_var = f'hop{hop}_edges'
            t1_var = f't1_{hop}'
            t2_var = f't2_{hop}'
            nodes_var = f'hop{hop}_nodes'

            dql += f'  # Hop {hop}: edges connected to {prev_nodes_var}\n'
            dql += f'  var(func: uid({prev_nodes_var})) {{\n'
            dql += f'    {out_var} as ~graphiti.edge_source\n'
            dql += f'    {in_var} as ~graphiti.edge_target\n'
            dql += '  }\n'
            dql += f'  {edge_var} as var(func: uid({out_var}, {in_var}))\n\n'
            all_edge_vars.append(edge_var)

            if hop < bfs_max_depth:
                dql += f'  # Hop {hop}: entities at other end\n'
                dql += f'  var(func: uid({edge_var})) {{\n'
                dql += f'    {t1_var} as graphiti.edge_target\n'
                dql += f'    {t2_var} as graphiti.edge_source\n'
                dql += '  }\n'
                dql += f'  {nodes_var} as var(func: uid({t1_var}, {t2_var}))\n\n'
                prev_nodes_var = nodes_var

        all_edges_union = ', '.join(all_edge_vars)
        dql += (
            f'  results(func: uid({all_edges_union}), first: {limit})\n'
            f'    @filter({result_filter}) {{\n'
            f'    {_ENTITY_EDGE_FIELDS}\n'
            f'  }}\n'
        )
        dql += '}'

        data = await driver.execute_query_raw(dql)
        records = data.get('results', [])
        return [_entity_edge_from_record(r) for r in records]

    async def node_bfs_search(
        self,
        driver: Any,
        bfs_origin_node_uuids: list[str] | None,
        search_filter: Any,
        bfs_max_depth: int,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        if not bfs_origin_node_uuids or bfs_max_depth < 1:
            return []

        group_filter = _format_group_id_filter(group_ids)
        extra_filter = self.build_node_search_filters(search_filter)

        result_filter_parts = ['type(Entity)']
        if group_filter:
            result_filter_parts.append(group_filter)
        if extra_filter:
            result_filter_parts.append(extra_filter.removeprefix(' AND '))
        result_filter = ' AND '.join(result_filter_parts)

        uuid_list = ', '.join(f'"{u}"' for u in bfs_origin_node_uuids)

        dql = '{\n'
        dql += f'  seeds as var(func: eq(graphiti.uuid, [{uuid_list}]))\n\n'

        prev_nodes_var = 'seeds'
        all_node_vars: list[str] = []

        for hop in range(1, bfs_max_depth + 1):
            out_var = f'out_{hop}'
            in_var = f'in_{hop}'
            edge_var = f'hop{hop}_edges'
            t1_var = f't1_{hop}'
            t2_var = f't2_{hop}'
            nodes_var = f'hop{hop}_nodes'

            dql += f'  var(func: uid({prev_nodes_var})) {{\n'
            dql += f'    {out_var} as ~graphiti.edge_source\n'
            dql += f'    {in_var} as ~graphiti.edge_target\n'
            dql += '  }\n'
            dql += f'  {edge_var} as var(func: uid({out_var}, {in_var}))\n'

            dql += f'  var(func: uid({edge_var})) {{\n'
            dql += f'    {t1_var} as graphiti.edge_target\n'
            dql += f'    {t2_var} as graphiti.edge_source\n'
            dql += '  }\n'
            dql += f'  {nodes_var} as var(func: uid({t1_var}, {t2_var}))\n\n'
            all_node_vars.append(nodes_var)
            prev_nodes_var = nodes_var

        all_nodes_union = ', '.join(all_node_vars)
        dql += (
            f'  results(func: uid({all_nodes_union}), first: {limit})\n'
            f'    @filter({result_filter}) {{\n'
            f'    {_ENTITY_NODE_FIELDS}\n'
            f'  }}\n'
        )
        dql += '}'

        data = await driver.execute_query_raw(dql)
        records = data.get('results', [])
        return [_entity_node_from_record(r) for r in records]

    # ------------------------------------------------------------------
    # Community searches
    # ------------------------------------------------------------------

    async def community_fulltext_search(
        self,
        driver: Any,
        query: str,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        if not query.strip():
            return []

        group_filter = _format_group_id_filter(group_ids)
        type_filter = 'type(Community)'
        filter_parts = [type_filter]
        if group_filter:
            filter_parts.append(group_filter)
        combined_filter = ' AND '.join(filter_parts)

        dql = '{\n'
        dql += (
            f'  by_name as var(func: anyoftext(graphiti.name, {json.dumps(query)}))\n'
            f'    @filter({combined_filter})\n'
        )
        dql += (
            f'  results(func: uid(by_name), first: {limit}) {{\n'
            f'    {_COMMUNITY_NODE_FIELDS}\n'
            f'  }}\n'
        )
        dql += '}'

        data = await driver.execute_query_raw(dql)
        records = data.get('results', [])
        return [_community_node_from_record(r) for r in records]

    async def community_similarity_search(
        self,
        driver: Any,
        search_vector: list[float],
        group_ids: list[str] | None = None,
        limit: int = 100,
        min_score: float = 0.6,
    ) -> list[Any]:
        group_filter = _format_group_id_filter(group_ids)
        type_filter = 'type(Community)'
        filter_parts = [type_filter]
        if group_filter:
            filter_parts.append(group_filter)
        combined_filter = ' AND '.join(filter_parts)

        overfetch = limit * _VECTOR_OVERFETCH_FACTOR
        vec_str = _format_vec_json(search_vector)
        distance_threshold = max(0.0, min(2.0, 1.0 - min_score))

        dql = '{\n'
        dql += (
            f'  results(func: similar_to(graphiti.name_embedding, {overfetch}, $vec,'
            f' distance_threshold: {distance_threshold}),'
            f' first: {limit})\n'
            f'    @filter({combined_filter}) {{\n'
            f'    {_COMMUNITY_NODE_FIELDS}\n'
            f'  }}\n'
        )
        dql += '}'

        variables = {'$vec': vec_str}
        data = await driver.execute_query_raw(dql, variables=variables)
        records = data.get('results', [])
        return [_community_node_from_record(r) for r in records]

    # ------------------------------------------------------------------
    # Embeddings loader
    # ------------------------------------------------------------------

    async def get_embeddings_for_communities(
        self,
        driver: Any,
        communities: list[Any],
    ) -> dict[str, list[float]]:
        if not communities:
            return {}

        uuid_list = ', '.join(f'"{c.uuid}"' for c in communities)
        dql = (
            '{\n'
            f'  results(func: eq(graphiti.uuid, [{uuid_list}]))'
            ' @filter(type(Community)) {\n'
            '    graphiti.uuid\n'
            '    graphiti.name_embedding\n'
            '  }\n'
            '}'
        )

        data = await driver.execute_query_raw(dql)
        records = data.get('results', [])

        embeddings_dict: dict[str, list[float]] = {}
        for record in records:
            uuid = record.get('graphiti.uuid')
            embedding = record.get('graphiti.name_embedding')
            if uuid and embedding is not None:
                embeddings_dict[uuid] = embedding
        return embeddings_dict

    # ------------------------------------------------------------------
    # Rerankers
    # ------------------------------------------------------------------

    async def node_distance_reranker(
        self,
        driver: Any,
        node_uuids: list[str],
        center_node_uuid: str,
        min_score: float = 0,
    ) -> tuple[list[str], list[float]]:
        filtered_uuids = [u for u in node_uuids if u != center_node_uuid]
        scores: dict[str, float] = {}

        # For each candidate, find shortest path to center node.
        for target_uuid in filtered_uuids:
            dql = (
                '{\n'
                f'  center as var(func: eq(graphiti.uuid, "{center_node_uuid}"))\n'
                f'  target as var(func: eq(graphiti.uuid, "{target_uuid}"))\n'
                '\n'
                '  path as shortest(from: uid(center), to: uid(target), numpaths: 1) {\n'
                '    graphiti.edge_source\n'
                '    graphiti.edge_target\n'
                '    ~graphiti.edge_source\n'
                '    ~graphiti.edge_target\n'
                '  }\n'
                '\n'
                '  result(func: uid(path)) {\n'
                '    uid\n'
                '  }\n'
                '}'
            )

            data = await driver.execute_query_raw(dql)
            path_nodes = data.get('result', [])
            # Path length in nodes; each intermediate edge node counts as a node.
            # Logical hop count = raw_path_length / 2
            if path_nodes and len(path_nodes) > 1:
                raw_length = len(path_nodes) - 1
                logical_distance = raw_length / 2
                scores[target_uuid] = logical_distance
            else:
                scores[target_uuid] = float('inf')

        # Sort by distance ascending
        filtered_uuids.sort(key=lambda u: scores[u])

        # Add center node back if it was in the original list
        if center_node_uuid in node_uuids:
            scores[center_node_uuid] = 0.1
            filtered_uuids = [center_node_uuid] + filtered_uuids

        result_uuids = []
        result_scores = []
        for uuid in filtered_uuids:
            dist = scores[uuid]
            score = 1.0 / dist if dist != 0 else 0.1
            if score >= min_score:
                result_uuids.append(uuid)
                result_scores.append(score)

        return result_uuids, result_scores

    async def episode_mentions_reranker(
        self,
        driver: Any,
        node_uuids: list[list[str]],
        min_score: float = 0,
    ) -> tuple[list[str], list[float]]:
        # Use RRF as a preliminary ranker
        rrf_scores: dict[str, float] = defaultdict(float)
        for result_list in node_uuids:
            for i, uuid in enumerate(result_list):
                rrf_scores[uuid] += 1 / (i + 1)

        sorted_uuids = sorted(rrf_scores.keys(), key=lambda u: rrf_scores[u], reverse=True)

        if not sorted_uuids:
            return [], []

        # Count episode mentions for each node
        mention_counts: dict[str, float] = {}
        uuid_list = ', '.join(f'"{u}"' for u in sorted_uuids)

        dql = (
            '{\n'
            f'  entities as var(func: eq(graphiti.uuid, [{uuid_list}]))'
            ' @filter(type(Entity))\n'
            '\n'
            '  results(func: uid(entities)) {\n'
            '    graphiti.uuid\n'
            '    mention_count: count(~graphiti.edge_target'
            ' @filter(type(MentionsEdge)))\n'
            '  }\n'
            '}'
        )

        data = await driver.execute_query_raw(dql)
        records = data.get('results', [])
        for record in records:
            uuid = record.get('graphiti.uuid')
            count = record.get('mention_count', 0)
            if uuid:
                mention_counts[uuid] = float(count)

        for uuid in sorted_uuids:
            if uuid not in mention_counts:
                mention_counts[uuid] = 0.0

        # Re-sort by mention count descending
        sorted_uuids.sort(key=lambda u: mention_counts[u], reverse=True)

        result_uuids = [u for u in sorted_uuids if mention_counts[u] >= min_score]
        result_scores = [mention_counts[u] for u in result_uuids]

        return result_uuids, result_scores

    # ------------------------------------------------------------------
    # Search filter builders (sync)
    # ------------------------------------------------------------------

    def build_node_search_filters(self, search_filters: Any) -> Any:
        """Build a DQL @filter clause fragment for node searches.

        Returns a string like 'AND eq(graphiti.labels, ["L1"])' or empty string.
        """
        if not isinstance(search_filters, SearchFilters):
            return ''

        parts: list[str] = []

        if search_filters.node_labels:
            escaped = ', '.join(f'"{lbl}"' for lbl in search_filters.node_labels)
            parts.append(f'eq(graphiti.labels, [{escaped}])')

        if search_filters.created_at:
            date_or_groups: list[str] = []
            for or_group in search_filters.created_at:
                and_parts: list[str] = []
                for df in or_group:
                    and_parts.append(
                        _dql_date_filter('graphiti.created_at', df.date, df.comparison_operator)
                    )
                if and_parts:
                    date_or_groups.append('(' + ' AND '.join(and_parts) + ')')
            if date_or_groups:
                parts.append('(' + ' OR '.join(date_or_groups) + ')')

        if not parts:
            return ''
        return ' AND ' + ' AND '.join(parts)

    def build_edge_search_filters(self, search_filters: Any) -> Any:
        """Build a DQL @filter clause fragment for edge searches.

        Returns a string like 'AND ge(graphiti.created_at, "...")' or empty string.
        """
        if not isinstance(search_filters, SearchFilters):
            return ''

        parts: list[str] = []

        # TODO: node_labels filtering for edges requires DQL variable blocks to join
        # source/target Entity nodes. Omitted — the previous implementation silently
        # filtered out all valid edges because edge intermediate nodes lack graphiti.labels.

        if search_filters.edge_types:
            escaped = ', '.join(f'"{t}"' for t in search_filters.edge_types)
            parts.append(f'eq(graphiti.name, [{escaped}])')

        for field_name, attr_name in [
            ('graphiti.created_at', 'created_at'),
            ('graphiti.expired_at', 'expired_at'),
            ('graphiti.valid_at', 'valid_at'),
            ('graphiti.invalid_at', 'invalid_at'),
        ]:
            date_groups = getattr(search_filters, attr_name, None)
            if date_groups:
                date_or_groups: list[str] = []
                for or_group in date_groups:
                    and_parts: list[str] = []
                    for df in or_group:
                        and_parts.append(
                            _dql_date_filter(field_name, df.date, df.comparison_operator)
                        )
                    if and_parts:
                        date_or_groups.append('(' + ' AND '.join(and_parts) + ')')
                if date_or_groups:
                    parts.append('(' + ' OR '.join(date_or_groups) + ')')

        if not parts:
            return ''
        return ' AND ' + ' AND '.join(parts)


# ---------------------------------------------------------------------------
# Helpers for DQL filter construction
# ---------------------------------------------------------------------------


def _dql_date_filter(
    predicate: str,
    date_val: datetime | None,
    op: ComparisonOperator,
) -> str:
    """Build a single DQL date comparison expression."""
    op_map = {
        ComparisonOperator.equals: 'eq',
        ComparisonOperator.not_equals: 'NOT eq',
        ComparisonOperator.greater_than: 'gt',
        ComparisonOperator.less_than: 'lt',
        ComparisonOperator.greater_than_equal: 'ge',
        ComparisonOperator.less_than_equal: 'le',
    }

    if op == ComparisonOperator.is_null:
        return f'NOT has({predicate})'
    if op == ComparisonOperator.is_not_null:
        return f'has({predicate})'

    func_name = op_map.get(op, 'eq')
    if date_val is None:
        return f'has({predicate})'

    date_str = date_val.strftime('%Y-%m-%dT%H:%M:%SZ')
    if func_name.startswith('NOT '):
        return f'NOT {func_name[4:]}({predicate}, "{date_str}")'
    return f'{func_name}({predicate}, "{date_str}")'
