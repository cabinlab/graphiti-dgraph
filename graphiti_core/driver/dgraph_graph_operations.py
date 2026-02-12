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

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Any

from graphiti_core.driver.graph_operations.graph_operations import GraphOperationsInterface
from graphiti_core.edges import (
    CommunityEdge,
    EntityEdge,
    EpisodicEdge,
    HasEpisodeEdge,
    NextEpisodeEdge,
)
from graphiti_core.errors import EdgeNotFoundError, NodeNotFoundError
from graphiti_core.helpers import semaphore_gather
from graphiti_core.nodes import (
    CommunityNode,
    EntityNode,
    EpisodeType,
    EpisodicNode,
    SagaNode,
)

if TYPE_CHECKING:
    from graphiti_core.driver.dgraph_driver import DgraphDriver

logger = logging.getLogger(__name__)

# Dgraph type names
ENTITY_TYPE = 'Entity'
EPISODIC_TYPE = 'Episodic'
COMMUNITY_TYPE = 'Community'
SAGA_TYPE = 'Saga'
RELATES_TO_EDGE_TYPE = 'RelatesToEdge'
MENTIONS_EDGE_TYPE = 'MentionsEdge'
HAS_MEMBER_EDGE_TYPE = 'HasMemberEdge'
HAS_EPISODE_EDGE_TYPE = 'HasEpisodeEdge'
NEXT_EPISODE_EDGE_TYPE = 'NextEpisodeEdge'

# DQL field projections
_ENTITY_FIELDS = (
    'uid dgraph.type graphiti.uuid graphiti.name graphiti.group_id '
    'graphiti.summary graphiti.created_at graphiti.labels graphiti.attributes'
)
_EPISODIC_FIELDS = (
    'uid dgraph.type graphiti.uuid graphiti.name graphiti.group_id '
    'graphiti.source graphiti.source_description graphiti.content '
    'graphiti.created_at graphiti.valid_at graphiti.entity_edges'
)
_COMMUNITY_FIELDS = (
    'uid dgraph.type graphiti.uuid graphiti.name graphiti.group_id '
    'graphiti.summary graphiti.created_at'
)
_SAGA_FIELDS = 'uid dgraph.type graphiti.uuid graphiti.name graphiti.group_id graphiti.created_at'
_RELATES_TO_EDGE_FIELDS = (
    'uid dgraph.type graphiti.uuid graphiti.name graphiti.group_id '
    'graphiti.fact graphiti.episodes graphiti.created_at '
    'graphiti.expired_at graphiti.valid_at graphiti.invalid_at graphiti.attributes '
    'graphiti.edge_source { graphiti.uuid } graphiti.edge_target { graphiti.uuid }'
)
_SIMPLE_EDGE_FIELDS = (
    'uid dgraph.type graphiti.uuid graphiti.group_id graphiti.created_at '
    'graphiti.edge_source { graphiti.uuid } graphiti.edge_target { graphiti.uuid }'
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dt_to_str(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.isoformat()


def _parse_dt(val: str | None) -> datetime | None:
    if not val:
        return None
    return datetime.fromisoformat(val)


def _cast_driver(driver: Any) -> DgraphDriver:
    return driver  # type: ignore[return-value]


async def _find_uid(driver: DgraphDriver, uuid: str, dgraph_type: str) -> str | None:
    """Find the internal Dgraph uid for a given graphiti uuid and type."""
    query = f'{{ q(func: eq(graphiti.uuid, "{uuid}")) @filter(type({dgraph_type})) {{ uid }} }}'
    records, _, _ = await driver.execute_query(query)
    if records and len(records) > 0:
        return records[0].get('uid')
    return None


async def _find_uids_batch(
    driver: DgraphDriver, uuids: list[str], dgraph_type: str
) -> dict[str, str]:
    """Find internal Dgraph uids for multiple graphiti uuids. Returns {uuid: uid}."""
    if not uuids:
        return {}
    uuid_list = ', '.join(f'"{u}"' for u in uuids)
    query = (
        f'{{ q(func: eq(graphiti.uuid, [{uuid_list}])) '
        f'@filter(type({dgraph_type})) {{ uid graphiti.uuid }} }}'
    )
    records, _, _ = await driver.execute_query(query)
    return {r['graphiti.uuid']: r['uid'] for r in (records or [])}


def _unwrap_single(val: Any) -> dict:
    """Dgraph may return uid predicates as single-element lists; unwrap."""
    if isinstance(val, list):
        return val[0] if val else {}
    return val if val else {}


def _build_entity_node(record: dict) -> EntityNode:
    labels_raw = record.get('graphiti.labels', [])
    labels = labels_raw if isinstance(labels_raw, list) else []
    attrs_raw = record.get('graphiti.attributes', '{}')
    attributes = json.loads(attrs_raw) if isinstance(attrs_raw, str) else {}

    return EntityNode(
        uuid=record['graphiti.uuid'],
        name=record.get('graphiti.name', ''),
        group_id=record.get('graphiti.group_id', ''),
        labels=labels,
        created_at=_parse_dt(record.get('graphiti.created_at')),  # type: ignore[arg-type]
        summary=record.get('graphiti.summary', ''),
        attributes=attributes,
    )


def _build_episodic_node(record: dict) -> EpisodicNode:
    entity_edges_raw = record.get('graphiti.entity_edges', '[]')
    entity_edges = json.loads(entity_edges_raw) if isinstance(entity_edges_raw, str) else []

    return EpisodicNode(
        uuid=record['graphiti.uuid'],
        name=record.get('graphiti.name', ''),
        group_id=record.get('graphiti.group_id', ''),
        source=EpisodeType.from_str(record.get('graphiti.source', 'text')),
        source_description=record.get('graphiti.source_description', ''),
        content=record.get('graphiti.content', ''),
        created_at=_parse_dt(record.get('graphiti.created_at')),  # type: ignore[arg-type]
        valid_at=_parse_dt(record.get('graphiti.valid_at')),  # type: ignore[arg-type]
        entity_edges=entity_edges,
    )


def _build_community_node(record: dict) -> CommunityNode:
    return CommunityNode(
        uuid=record['graphiti.uuid'],
        name=record.get('graphiti.name', ''),
        group_id=record.get('graphiti.group_id', ''),
        created_at=_parse_dt(record.get('graphiti.created_at')),  # type: ignore[arg-type]
        summary=record.get('graphiti.summary', ''),
    )


def _build_saga_node(record: dict) -> SagaNode:
    return SagaNode(
        uuid=record['graphiti.uuid'],
        name=record.get('graphiti.name', ''),
        group_id=record.get('graphiti.group_id', ''),
        created_at=_parse_dt(record.get('graphiti.created_at')),  # type: ignore[arg-type]
    )


def _build_entity_edge(record: dict) -> EntityEdge:
    episodes_raw = record.get('graphiti.episodes', '[]')
    episodes = json.loads(episodes_raw) if isinstance(episodes_raw, str) else []
    attrs_raw = record.get('graphiti.attributes', '{}')
    attributes = json.loads(attrs_raw) if isinstance(attrs_raw, str) else {}

    src = _unwrap_single(record.get('graphiti.edge_source'))
    tgt = _unwrap_single(record.get('graphiti.edge_target'))

    return EntityEdge(
        uuid=record['graphiti.uuid'],
        group_id=record.get('graphiti.group_id', ''),
        source_node_uuid=src.get('graphiti.uuid', ''),
        target_node_uuid=tgt.get('graphiti.uuid', ''),
        name=record.get('graphiti.name', ''),
        fact=record.get('graphiti.fact', ''),
        episodes=episodes,
        created_at=_parse_dt(record.get('graphiti.created_at')),  # type: ignore[arg-type]
        expired_at=_parse_dt(record.get('graphiti.expired_at')),
        valid_at=_parse_dt(record.get('graphiti.valid_at')),
        invalid_at=_parse_dt(record.get('graphiti.invalid_at')),
        attributes=attributes,
    )


def _build_simple_edge(record: dict, edge_cls: type) -> Any:
    """Build EpisodicEdge, CommunityEdge, HasEpisodeEdge, or NextEpisodeEdge."""
    src = _unwrap_single(record.get('graphiti.edge_source'))
    tgt = _unwrap_single(record.get('graphiti.edge_target'))

    return edge_cls(
        uuid=record['graphiti.uuid'],
        group_id=record.get('graphiti.group_id', ''),
        source_node_uuid=src.get('graphiti.uuid', ''),
        target_node_uuid=tgt.get('graphiti.uuid', ''),
        created_at=_parse_dt(record.get('graphiti.created_at')),  # type: ignore[arg-type]
    )


async def _upsert_node(driver: DgraphDriver, uuid: str, dgraph_type: str, data: dict) -> None:
    """Upsert a node: find existing uid or create new."""
    existing_uid = await _find_uid(driver, uuid, dgraph_type)
    if existing_uid:
        data['uid'] = existing_uid
    else:
        data['uid'] = '_:new'
    data['dgraph.type'] = dgraph_type
    await driver.mutate_json(set_json=data)


async def _upsert_edge_node(
    driver: DgraphDriver,
    uuid: str,
    dgraph_type: str,
    source_uuid: str,
    source_type: str,
    target_uuid: str,
    target_type: str,
    data: dict,
) -> None:
    """Upsert an intermediate edge node with source/target uid links."""
    src_uid = await _find_uid(driver, source_uuid, source_type)
    tgt_uid = await _find_uid(driver, target_uuid, target_type)

    if not src_uid or not tgt_uid:
        logger.warning(
            f'Cannot save edge {uuid}: source ({source_uuid}) uid={src_uid}, '
            f'target ({target_uuid}) uid={tgt_uid}'
        )
        return

    existing_uid = await _find_uid(driver, uuid, dgraph_type)
    if existing_uid:
        data['uid'] = existing_uid
    else:
        data['uid'] = '_:new'

    data['dgraph.type'] = dgraph_type
    data['graphiti.edge_source'] = {'uid': src_uid}
    data['graphiti.edge_target'] = {'uid': tgt_uid}
    await driver.mutate_json(set_json=data)


async def _delete_node_by_uuid(driver: DgraphDriver, uuid: str, dgraph_type: str) -> None:
    uid = await _find_uid(driver, uuid, dgraph_type)
    if uid:
        await driver.mutate_json(delete_json={'uid': uid})


async def _delete_nodes_by_uuids(driver: DgraphDriver, uuids: list[str], dgraph_type: str) -> None:
    uid_map = await _find_uids_batch(driver, uuids, dgraph_type)
    if uid_map:
        await driver.mutate_json(delete_json=[{'uid': uid} for uid in uid_map.values()])


async def _delete_nodes_by_group_id(driver: DgraphDriver, group_id: str, dgraph_type: str) -> None:
    query = (
        f'{{ q(func: eq(graphiti.group_id, "{group_id}")) @filter(type({dgraph_type})) {{ uid }} }}'
    )
    records, _, _ = await driver.execute_query(query)
    if records:
        await driver.mutate_json(delete_json=[{'uid': r['uid']} for r in records])


# ---------------------------------------------------------------------------
# DgraphGraphOperations
# ---------------------------------------------------------------------------


class DgraphGraphOperations(GraphOperationsInterface):
    """GraphOperationsInterface implementation for Dgraph."""

    # -----------------------------------------------------------------------
    # EntityNode: Save / Delete
    # -----------------------------------------------------------------------

    async def node_save(self, node: Any, driver: Any) -> None:
        d = _cast_driver(driver)
        n: EntityNode = node
        data: dict[str, Any] = {
            'graphiti.uuid': n.uuid,
            'graphiti.name': n.name,
            'graphiti.group_id': n.group_id,
            'graphiti.summary': n.summary,
            'graphiti.created_at': _dt_to_str(n.created_at),
            'graphiti.labels': list(set(n.labels)),
            'graphiti.attributes': json.dumps(n.attributes or {}),
        }
        if n.name_embedding is not None:
            data['graphiti.name_embedding'] = n.name_embedding
        await _upsert_node(d, n.uuid, ENTITY_TYPE, data)
        logger.debug(f'Saved Entity node: {n.uuid}')

    async def node_delete(self, node: Any, driver: Any) -> None:
        d = _cast_driver(driver)
        uid = await _find_uid(d, node.uuid, ENTITY_TYPE)
        if uid:
            # Delete edge intermediate nodes connected to this entity
            query = (
                f'{{ src(func: uid({uid})) {{ '
                f'~graphiti.edge_source {{ uid }} }} '
                f'tgt(func: uid({uid})) {{ '
                f'~graphiti.edge_target {{ uid }} }} }}'
            )
            data = await d.execute_query_raw(query)
            edge_uids: set[str] = set()
            for block in ['src', 'tgt']:
                for rec in data.get(block, []):
                    for key in ['~graphiti.edge_source', '~graphiti.edge_target']:
                        for edge_rec in rec.get(key, []):
                            edge_uids.add(edge_rec['uid'])
            deletes = [{'uid': eu} for eu in edge_uids]
            deletes.append({'uid': uid})
            await d.mutate_json(delete_json=deletes)
        logger.debug(f'Deleted Entity node: {node.uuid}')

    async def node_save_bulk(
        self, _cls: Any, driver: Any, transaction: Any, nodes: list[Any], batch_size: int = 100
    ) -> None:
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i : i + batch_size]
            await semaphore_gather(*[self.node_save(n, driver) for n in batch])

    async def node_delete_by_group_id(
        self, _cls: Any, driver: Any, group_id: str, batch_size: int = 100
    ) -> None:
        d = _cast_driver(driver)
        for t in [
            ENTITY_TYPE,
            EPISODIC_TYPE,
            COMMUNITY_TYPE,
            SAGA_TYPE,
            RELATES_TO_EDGE_TYPE,
            MENTIONS_EDGE_TYPE,
            HAS_MEMBER_EDGE_TYPE,
            HAS_EPISODE_EDGE_TYPE,
            NEXT_EPISODE_EDGE_TYPE,
        ]:
            await _delete_nodes_by_group_id(d, group_id, t)

    async def node_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
        batch_size: int = 100,
    ) -> None:
        d = _cast_driver(driver)
        for t in [ENTITY_TYPE, EPISODIC_TYPE, COMMUNITY_TYPE, SAGA_TYPE]:
            await _delete_nodes_by_uuids(d, uuids, t)

    # -----------------------------------------------------------------------
    # EntityNode: Read
    # -----------------------------------------------------------------------

    async def node_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> EntityNode:
        d = _cast_driver(driver)
        query = (
            f'{{ q(func: eq(graphiti.uuid, "{uuid}")) '
            f'@filter(type({ENTITY_TYPE})) {{ {_ENTITY_FIELDS} }} }}'
        )
        records, _, _ = await d.execute_query(query)
        if not records:
            raise NodeNotFoundError(uuid)
        return _build_entity_node(records[0])

    async def node_get_by_uuids(self, _cls: Any, driver: Any, uuids: list[str]) -> list[EntityNode]:
        if not uuids:
            return []
        d = _cast_driver(driver)
        uuid_list = ', '.join(f'"{u}"' for u in uuids)
        query = (
            f'{{ q(func: eq(graphiti.uuid, [{uuid_list}])) '
            f'@filter(type({ENTITY_TYPE})) {{ {_ENTITY_FIELDS} }} }}'
        )
        records, _, _ = await d.execute_query(query)
        return [_build_entity_node(r) for r in (records or [])]

    async def node_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[EntityNode]:
        d = _cast_driver(driver)
        gid_list = ', '.join(f'"{g}"' for g in group_ids)
        cursor_filter = ''
        if uuid_cursor:
            cursor_filter = f' AND lt(graphiti.uuid, "{uuid_cursor}")'
        limit_clause = f', first: {limit}' if limit is not None else ''
        query = (
            f'{{ q(func: eq(graphiti.group_id, [{gid_list}]){limit_clause}'
            f', orderdesc: graphiti.uuid) '
            f'@filter(type({ENTITY_TYPE}){cursor_filter}) '
            f'{{ {_ENTITY_FIELDS} }} }}'
        )
        records, _, _ = await d.execute_query(query)
        return [_build_entity_node(r) for r in (records or [])]

    # -----------------------------------------------------------------------
    # EntityNode: Embeddings
    # -----------------------------------------------------------------------

    async def node_load_embeddings(self, node: Any, driver: Any) -> None:
        d = _cast_driver(driver)
        query = (
            f'{{ q(func: eq(graphiti.uuid, "{node.uuid}")) '
            f'@filter(type({ENTITY_TYPE})) {{ graphiti.name_embedding }} }}'
        )
        records, _, _ = await d.execute_query(query)
        if not records:
            raise NodeNotFoundError(node.uuid)
        node.name_embedding = records[0].get('graphiti.name_embedding')

    async def node_load_embeddings_bulk(
        self,
        driver: Any,
        nodes: list[Any],
        batch_size: int = 100,
    ) -> dict[str, list[float]]:
        d = _cast_driver(driver)
        result: dict[str, list[float]] = {}
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i : i + batch_size]
            uuid_list = ', '.join(f'"{n.uuid}"' for n in batch)
            query = (
                f'{{ q(func: eq(graphiti.uuid, [{uuid_list}])) '
                f'@filter(type({ENTITY_TYPE})) '
                f'{{ graphiti.uuid graphiti.name_embedding }} }}'
            )
            records, _, _ = await d.execute_query(query)
            for r in records or []:
                emb = r.get('graphiti.name_embedding')
                if emb is not None:
                    result[r['graphiti.uuid']] = emb
        return result

    # -----------------------------------------------------------------------
    # EpisodicNode: Save / Delete
    # -----------------------------------------------------------------------

    async def episodic_node_save(self, node: Any, driver: Any) -> None:
        d = _cast_driver(driver)
        n: EpisodicNode = node
        data: dict[str, Any] = {
            'graphiti.uuid': n.uuid,
            'graphiti.name': n.name,
            'graphiti.group_id': n.group_id,
            'graphiti.source': n.source.value,
            'graphiti.source_description': n.source_description,
            'graphiti.content': n.content,
            'graphiti.created_at': _dt_to_str(n.created_at),
            'graphiti.valid_at': _dt_to_str(n.valid_at),
            'graphiti.entity_edges': json.dumps(n.entity_edges),
        }
        await _upsert_node(d, n.uuid, EPISODIC_TYPE, data)
        logger.debug(f'Saved Episodic node: {n.uuid}')

    async def episodic_node_delete(self, node: Any, driver: Any) -> None:
        await _delete_node_by_uuid(_cast_driver(driver), node.uuid, EPISODIC_TYPE)
        logger.debug(f'Deleted Episodic node: {node.uuid}')

    async def episodic_node_save_bulk(
        self, _cls: Any, driver: Any, transaction: Any, nodes: list[Any], batch_size: int = 100
    ) -> None:
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i : i + batch_size]
            await semaphore_gather(*[self.episodic_node_save(n, driver) for n in batch])

    async def episodic_edge_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        episodic_edges: list[Any],
        batch_size: int = 100,
    ) -> None:
        for i in range(0, len(episodic_edges), batch_size):
            batch = episodic_edges[i : i + batch_size]
            await semaphore_gather(*[self.episodic_edge_save(e, driver) for e in batch])

    async def episodic_node_delete_by_group_id(
        self, _cls: Any, driver: Any, group_id: str, batch_size: int = 100
    ) -> None:
        await _delete_nodes_by_group_id(_cast_driver(driver), group_id, EPISODIC_TYPE)

    async def episodic_node_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
        batch_size: int = 100,
    ) -> None:
        await _delete_nodes_by_uuids(_cast_driver(driver), uuids, EPISODIC_TYPE)

    # -----------------------------------------------------------------------
    # EpisodicNode: Read
    # -----------------------------------------------------------------------

    async def episodic_node_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> EpisodicNode:
        d = _cast_driver(driver)
        query = (
            f'{{ q(func: eq(graphiti.uuid, "{uuid}")) '
            f'@filter(type({EPISODIC_TYPE})) {{ {_EPISODIC_FIELDS} }} }}'
        )
        records, _, _ = await d.execute_query(query)
        if not records:
            raise NodeNotFoundError(uuid)
        return _build_episodic_node(records[0])

    async def episodic_node_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[EpisodicNode]:
        if not uuids:
            return []
        d = _cast_driver(driver)
        uuid_list = ', '.join(f'"{u}"' for u in uuids)
        query = (
            f'{{ q(func: eq(graphiti.uuid, [{uuid_list}])) '
            f'@filter(type({EPISODIC_TYPE})) {{ {_EPISODIC_FIELDS} }} }}'
        )
        records, _, _ = await d.execute_query(query)
        return [_build_episodic_node(r) for r in (records or [])]

    async def episodic_node_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[EpisodicNode]:
        d = _cast_driver(driver)
        gid_list = ', '.join(f'"{g}"' for g in group_ids)
        cursor_filter = ''
        if uuid_cursor:
            cursor_filter = f' AND lt(graphiti.uuid, "{uuid_cursor}")'
        limit_clause = f', first: {limit}' if limit is not None else ''
        query = (
            f'{{ q(func: eq(graphiti.group_id, [{gid_list}]){limit_clause}'
            f', orderdesc: graphiti.uuid) '
            f'@filter(type({EPISODIC_TYPE}){cursor_filter}) '
            f'{{ {_EPISODIC_FIELDS} }} }}'
        )
        records, _, _ = await d.execute_query(query)
        return [_build_episodic_node(r) for r in (records or [])]

    async def retrieve_episodes(
        self,
        driver: Any,
        reference_time: Any,
        last_n: int = 3,
        group_ids: list[str] | None = None,
        source: Any | None = None,
        saga: str | None = None,
    ) -> list[EpisodicNode]:
        d = _cast_driver(driver)
        ref_time_str = _dt_to_str(reference_time)

        if saga is not None:
            group_id = group_ids[0] if group_ids else None
            # Find saga, follow HAS_EPISODE edges to episodes
            saga_filter = f'eq(graphiti.name, "{saga}")'
            if group_id is not None:
                saga_filter += f' AND eq(graphiti.group_id, "{group_id}")'

            query = (
                f'{{ saga(func: {saga_filter}) @filter(type({SAGA_TYPE})) {{ '
                f'~graphiti.edge_source @filter(type({HAS_EPISODE_EDGE_TYPE})) {{ '
                f'graphiti.edge_target @filter(type({EPISODIC_TYPE})) {{ '
                f'{_EPISODIC_FIELDS} '
                f'}} }} }} }}'
            )
            data = await d.execute_query_raw(query)
            episodes: list[EpisodicNode] = []
            for saga_rec in data.get('saga', []):
                for edge_rec in saga_rec.get('~graphiti.edge_source', []):
                    targets = edge_rec.get('graphiti.edge_target', [])
                    if not isinstance(targets, list):
                        targets = [targets]
                    for ep_rec in targets:
                        if not ep_rec or 'graphiti.uuid' not in ep_rec:
                            continue
                        ep = _build_episodic_node(ep_rec)
                        if ep.valid_at and ep.valid_at <= reference_time:
                            if source is not None and ep.source != source:
                                continue
                            episodes.append(ep)
            episodes.sort(key=lambda e: e.valid_at, reverse=True)
            return list(reversed(episodes[:last_n]))

        # Non-saga: direct query on Episodic nodes
        filters = [f'type({EPISODIC_TYPE})', f'le(graphiti.valid_at, "{ref_time_str}")']
        if group_ids:
            gid_list = ', '.join(f'"{g}"' for g in group_ids)
            filters.append(f'eq(graphiti.group_id, [{gid_list}])')
        if source is not None:
            filters.append(f'eq(graphiti.source, "{source.value}")')

        filter_str = ' AND '.join(filters)
        query = (
            f'{{ q(func: has(graphiti.valid_at), orderdesc: graphiti.valid_at'
            f', first: {last_n}) '
            f'@filter({filter_str}) '
            f'{{ {_EPISODIC_FIELDS} }} }}'
        )
        records, _, _ = await d.execute_query(query)
        episodes = [_build_episodic_node(r) for r in (records or [])]
        return list(reversed(episodes))

    # -----------------------------------------------------------------------
    # CommunityNode: Save / Delete
    # -----------------------------------------------------------------------

    async def community_node_save(self, node: Any, driver: Any) -> None:
        d = _cast_driver(driver)
        n: CommunityNode = node
        data: dict[str, Any] = {
            'graphiti.uuid': n.uuid,
            'graphiti.name': n.name,
            'graphiti.group_id': n.group_id,
            'graphiti.summary': n.summary,
            'graphiti.created_at': _dt_to_str(n.created_at),
        }
        if n.name_embedding is not None:
            data['graphiti.name_embedding'] = n.name_embedding
        await _upsert_node(d, n.uuid, COMMUNITY_TYPE, data)
        logger.debug(f'Saved Community node: {n.uuid}')

    async def community_node_delete(self, node: Any, driver: Any) -> None:
        await _delete_node_by_uuid(_cast_driver(driver), node.uuid, COMMUNITY_TYPE)
        logger.debug(f'Deleted Community node: {node.uuid}')

    async def community_node_save_bulk(
        self, _cls: Any, driver: Any, transaction: Any, nodes: list[Any], batch_size: int = 100
    ) -> None:
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i : i + batch_size]
            await semaphore_gather(*[self.community_node_save(n, driver) for n in batch])

    async def community_node_delete_by_group_id(
        self, _cls: Any, driver: Any, group_id: str, batch_size: int = 100
    ) -> None:
        await _delete_nodes_by_group_id(_cast_driver(driver), group_id, COMMUNITY_TYPE)

    async def community_node_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
        batch_size: int = 100,
    ) -> None:
        await _delete_nodes_by_uuids(_cast_driver(driver), uuids, COMMUNITY_TYPE)

    # -----------------------------------------------------------------------
    # CommunityNode: Read
    # -----------------------------------------------------------------------

    async def community_node_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> CommunityNode:
        d = _cast_driver(driver)
        query = (
            f'{{ q(func: eq(graphiti.uuid, "{uuid}")) '
            f'@filter(type({COMMUNITY_TYPE})) {{ {_COMMUNITY_FIELDS} }} }}'
        )
        records, _, _ = await d.execute_query(query)
        if not records:
            raise NodeNotFoundError(uuid)
        return _build_community_node(records[0])

    async def community_node_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[CommunityNode]:
        if not uuids:
            return []
        d = _cast_driver(driver)
        uuid_list = ', '.join(f'"{u}"' for u in uuids)
        query = (
            f'{{ q(func: eq(graphiti.uuid, [{uuid_list}])) '
            f'@filter(type({COMMUNITY_TYPE})) {{ {_COMMUNITY_FIELDS} }} }}'
        )
        records, _, _ = await d.execute_query(query)
        return [_build_community_node(r) for r in (records or [])]

    async def community_node_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[CommunityNode]:
        d = _cast_driver(driver)
        gid_list = ', '.join(f'"{g}"' for g in group_ids)
        cursor_filter = ''
        if uuid_cursor:
            cursor_filter = f' AND lt(graphiti.uuid, "{uuid_cursor}")'
        limit_clause = f', first: {limit}' if limit is not None else ''
        query = (
            f'{{ q(func: eq(graphiti.group_id, [{gid_list}]){limit_clause}'
            f', orderdesc: graphiti.uuid) '
            f'@filter(type({COMMUNITY_TYPE}){cursor_filter}) '
            f'{{ {_COMMUNITY_FIELDS} }} }}'
        )
        records, _, _ = await d.execute_query(query)
        return [_build_community_node(r) for r in (records or [])]

    # -----------------------------------------------------------------------
    # SagaNode: Save / Delete
    # -----------------------------------------------------------------------

    async def saga_node_save(self, node: Any, driver: Any) -> None:
        d = _cast_driver(driver)
        n: SagaNode = node
        data: dict[str, Any] = {
            'graphiti.uuid': n.uuid,
            'graphiti.name': n.name,
            'graphiti.group_id': n.group_id,
            'graphiti.created_at': _dt_to_str(n.created_at),
        }
        await _upsert_node(d, n.uuid, SAGA_TYPE, data)
        logger.debug(f'Saved Saga node: {n.uuid}')

    async def saga_node_delete(self, node: Any, driver: Any) -> None:
        await _delete_node_by_uuid(_cast_driver(driver), node.uuid, SAGA_TYPE)
        logger.debug(f'Deleted Saga node: {node.uuid}')

    async def saga_node_save_bulk(
        self, _cls: Any, driver: Any, transaction: Any, nodes: list[Any], batch_size: int = 100
    ) -> None:
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i : i + batch_size]
            await semaphore_gather(*[self.saga_node_save(n, driver) for n in batch])

    async def saga_node_delete_by_group_id(
        self, _cls: Any, driver: Any, group_id: str, batch_size: int = 100
    ) -> None:
        await _delete_nodes_by_group_id(_cast_driver(driver), group_id, SAGA_TYPE)

    async def saga_node_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
        batch_size: int = 100,
    ) -> None:
        await _delete_nodes_by_uuids(_cast_driver(driver), uuids, SAGA_TYPE)

    # -----------------------------------------------------------------------
    # SagaNode: Read
    # -----------------------------------------------------------------------

    async def saga_node_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> SagaNode:
        d = _cast_driver(driver)
        query = (
            f'{{ q(func: eq(graphiti.uuid, "{uuid}")) '
            f'@filter(type({SAGA_TYPE})) {{ {_SAGA_FIELDS} }} }}'
        )
        records, _, _ = await d.execute_query(query)
        if not records:
            raise NodeNotFoundError(uuid)
        return _build_saga_node(records[0])

    async def saga_node_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[SagaNode]:
        if not uuids:
            return []
        d = _cast_driver(driver)
        uuid_list = ', '.join(f'"{u}"' for u in uuids)
        query = (
            f'{{ q(func: eq(graphiti.uuid, [{uuid_list}])) '
            f'@filter(type({SAGA_TYPE})) {{ {_SAGA_FIELDS} }} }}'
        )
        records, _, _ = await d.execute_query(query)
        return [_build_saga_node(r) for r in (records or [])]

    async def saga_node_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[SagaNode]:
        d = _cast_driver(driver)
        gid_list = ', '.join(f'"{g}"' for g in group_ids)
        cursor_filter = ''
        if uuid_cursor:
            cursor_filter = f' AND lt(graphiti.uuid, "{uuid_cursor}")'
        limit_clause = f', first: {limit}' if limit is not None else ''
        query = (
            f'{{ q(func: eq(graphiti.group_id, [{gid_list}]){limit_clause}'
            f', orderdesc: graphiti.uuid) '
            f'@filter(type({SAGA_TYPE}){cursor_filter}) '
            f'{{ {_SAGA_FIELDS} }} }}'
        )
        records, _, _ = await d.execute_query(query)
        return [_build_saga_node(r) for r in (records or [])]

    # -----------------------------------------------------------------------
    # EntityEdge (RelatesToEdge): Save / Delete
    # -----------------------------------------------------------------------

    async def edge_save(self, edge: Any, driver: Any) -> None:
        d = _cast_driver(driver)
        e: EntityEdge = edge
        data: dict[str, Any] = {
            'graphiti.uuid': e.uuid,
            'graphiti.name': e.name,
            'graphiti.group_id': e.group_id,
            'graphiti.fact': e.fact,
            'graphiti.episodes': json.dumps(e.episodes),
            'graphiti.created_at': _dt_to_str(e.created_at),
            'graphiti.expired_at': _dt_to_str(e.expired_at),
            'graphiti.valid_at': _dt_to_str(e.valid_at),
            'graphiti.invalid_at': _dt_to_str(e.invalid_at),
            'graphiti.attributes': json.dumps(e.attributes or {}),
        }
        if e.fact_embedding is not None:
            data['graphiti.fact_embedding'] = e.fact_embedding
        await _upsert_edge_node(
            d,
            e.uuid,
            RELATES_TO_EDGE_TYPE,
            e.source_node_uuid,
            ENTITY_TYPE,
            e.target_node_uuid,
            ENTITY_TYPE,
            data,
        )
        logger.debug(f'Saved EntityEdge: {e.uuid}')

    async def edge_delete(self, edge: Any, driver: Any) -> None:
        d = _cast_driver(driver)
        for edge_type in [
            RELATES_TO_EDGE_TYPE,
            MENTIONS_EDGE_TYPE,
            HAS_MEMBER_EDGE_TYPE,
            HAS_EPISODE_EDGE_TYPE,
            NEXT_EPISODE_EDGE_TYPE,
        ]:
            uid = await _find_uid(d, edge.uuid, edge_type)
            if uid:
                await d.mutate_json(delete_json={'uid': uid})
                logger.debug(f'Deleted Edge: {edge.uuid}')
                return

    async def edge_save_bulk(
        self, _cls: Any, driver: Any, transaction: Any, edges: list[Any], batch_size: int = 100
    ) -> None:
        for i in range(0, len(edges), batch_size):
            batch = edges[i : i + batch_size]
            await semaphore_gather(*[self.edge_save(e, driver) for e in batch])

    async def edge_delete_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str], group_id: str | None = None
    ) -> None:
        d = _cast_driver(driver)
        for edge_type in [RELATES_TO_EDGE_TYPE, MENTIONS_EDGE_TYPE, HAS_MEMBER_EDGE_TYPE]:
            await _delete_nodes_by_uuids(d, uuids, edge_type)

    # -----------------------------------------------------------------------
    # EntityEdge (RelatesToEdge): Read
    # -----------------------------------------------------------------------

    async def edge_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> EntityEdge:
        d = _cast_driver(driver)
        query = (
            f'{{ q(func: eq(graphiti.uuid, "{uuid}")) '
            f'@filter(type({RELATES_TO_EDGE_TYPE})) {{ {_RELATES_TO_EDGE_FIELDS} }} }}'
        )
        records, _, _ = await d.execute_query(query)
        if not records:
            raise EdgeNotFoundError(uuid)
        return _build_entity_edge(records[0])

    async def edge_get_by_uuids(self, _cls: Any, driver: Any, uuids: list[str]) -> list[EntityEdge]:
        if not uuids:
            return []
        d = _cast_driver(driver)
        uuid_list = ', '.join(f'"{u}"' for u in uuids)
        query = (
            f'{{ q(func: eq(graphiti.uuid, [{uuid_list}])) '
            f'@filter(type({RELATES_TO_EDGE_TYPE})) {{ {_RELATES_TO_EDGE_FIELDS} }} }}'
        )
        records, _, _ = await d.execute_query(query)
        return [_build_entity_edge(r) for r in (records or [])]

    async def edge_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[EntityEdge]:
        from graphiti_core.errors import GroupsEdgesNotFoundError

        d = _cast_driver(driver)
        gid_list = ', '.join(f'"{g}"' for g in group_ids)
        cursor_filter = ''
        if uuid_cursor:
            cursor_filter = f' AND lt(graphiti.uuid, "{uuid_cursor}")'
        limit_clause = f', first: {limit}' if limit is not None else ''
        query = (
            f'{{ q(func: eq(graphiti.group_id, [{gid_list}]){limit_clause}'
            f', orderdesc: graphiti.uuid) '
            f'@filter(type({RELATES_TO_EDGE_TYPE}){cursor_filter}) '
            f'{{ {_RELATES_TO_EDGE_FIELDS} }} }}'
        )
        records, _, _ = await d.execute_query(query)
        edges = [_build_entity_edge(r) for r in (records or [])]
        if not edges:
            raise GroupsEdgesNotFoundError(group_ids)
        return edges

    # -----------------------------------------------------------------------
    # EntityEdge: Embeddings
    # -----------------------------------------------------------------------

    async def edge_load_embeddings(self, edge: Any, driver: Any) -> None:
        d = _cast_driver(driver)
        query = (
            f'{{ q(func: eq(graphiti.uuid, "{edge.uuid}")) '
            f'@filter(type({RELATES_TO_EDGE_TYPE})) {{ graphiti.fact_embedding }} }}'
        )
        records, _, _ = await d.execute_query(query)
        if not records:
            raise EdgeNotFoundError(edge.uuid)
        edge.fact_embedding = records[0].get('graphiti.fact_embedding')

    async def edge_load_embeddings_bulk(
        self,
        driver: Any,
        edges: list[Any],
        batch_size: int = 100,
    ) -> dict[str, list[float]]:
        d = _cast_driver(driver)
        result: dict[str, list[float]] = {}
        for i in range(0, len(edges), batch_size):
            batch = edges[i : i + batch_size]
            uuid_list = ', '.join(f'"{e.uuid}"' for e in batch)
            query = (
                f'{{ q(func: eq(graphiti.uuid, [{uuid_list}])) '
                f'@filter(type({RELATES_TO_EDGE_TYPE})) '
                f'{{ graphiti.uuid graphiti.fact_embedding }} }}'
            )
            records, _, _ = await d.execute_query(query)
            for r in records or []:
                emb = r.get('graphiti.fact_embedding')
                if emb is not None:
                    result[r['graphiti.uuid']] = emb
        return result

    # -----------------------------------------------------------------------
    # EpisodicEdge (MentionsEdge): Save / Delete / Read
    # -----------------------------------------------------------------------

    async def episodic_edge_save(self, edge: Any, driver: Any) -> None:
        d = _cast_driver(driver)
        e: EpisodicEdge = edge
        data: dict[str, Any] = {
            'graphiti.uuid': e.uuid,
            'graphiti.group_id': e.group_id,
            'graphiti.created_at': _dt_to_str(e.created_at),
        }
        await _upsert_edge_node(
            d,
            e.uuid,
            MENTIONS_EDGE_TYPE,
            e.source_node_uuid,
            EPISODIC_TYPE,
            e.target_node_uuid,
            ENTITY_TYPE,
            data,
        )

    async def episodic_edge_delete(self, edge: Any, driver: Any) -> None:
        await _delete_node_by_uuid(_cast_driver(driver), edge.uuid, MENTIONS_EDGE_TYPE)

    async def episodic_edge_delete_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str], group_id: str | None = None
    ) -> None:
        await _delete_nodes_by_uuids(_cast_driver(driver), uuids, MENTIONS_EDGE_TYPE)

    async def episodic_edge_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> EpisodicEdge:
        d = _cast_driver(driver)
        query = (
            f'{{ q(func: eq(graphiti.uuid, "{uuid}")) '
            f'@filter(type({MENTIONS_EDGE_TYPE})) {{ {_SIMPLE_EDGE_FIELDS} }} }}'
        )
        records, _, _ = await d.execute_query(query)
        if not records:
            raise EdgeNotFoundError(uuid)
        return _build_simple_edge(records[0], EpisodicEdge)

    async def episodic_edge_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[EpisodicEdge]:
        if not uuids:
            return []
        d = _cast_driver(driver)
        uuid_list = ', '.join(f'"{u}"' for u in uuids)
        query = (
            f'{{ q(func: eq(graphiti.uuid, [{uuid_list}])) '
            f'@filter(type({MENTIONS_EDGE_TYPE})) {{ {_SIMPLE_EDGE_FIELDS} }} }}'
        )
        records, _, _ = await d.execute_query(query)
        if not records:
            raise EdgeNotFoundError(uuids[0])
        return [_build_simple_edge(r, EpisodicEdge) for r in records]

    async def episodic_edge_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[EpisodicEdge]:
        from graphiti_core.errors import GroupsEdgesNotFoundError

        d = _cast_driver(driver)
        gid_list = ', '.join(f'"{g}"' for g in group_ids)
        cursor_filter = ''
        if uuid_cursor:
            cursor_filter = f' AND lt(graphiti.uuid, "{uuid_cursor}")'
        limit_clause = f', first: {limit}' if limit is not None else ''
        query = (
            f'{{ q(func: eq(graphiti.group_id, [{gid_list}]){limit_clause}'
            f', orderdesc: graphiti.uuid) '
            f'@filter(type({MENTIONS_EDGE_TYPE}){cursor_filter}) '
            f'{{ {_SIMPLE_EDGE_FIELDS} }} }}'
        )
        records, _, _ = await d.execute_query(query)
        edges = [_build_simple_edge(r, EpisodicEdge) for r in (records or [])]
        if not edges:
            raise GroupsEdgesNotFoundError(group_ids)
        return edges

    # -----------------------------------------------------------------------
    # CommunityEdge (HasMemberEdge): Save / Delete / Read
    # -----------------------------------------------------------------------

    async def community_edge_save(self, edge: Any, driver: Any) -> None:
        d = _cast_driver(driver)
        e: CommunityEdge = edge
        data: dict[str, Any] = {
            'graphiti.uuid': e.uuid,
            'graphiti.group_id': e.group_id,
            'graphiti.created_at': _dt_to_str(e.created_at),
        }
        await _upsert_edge_node(
            d,
            e.uuid,
            HAS_MEMBER_EDGE_TYPE,
            e.source_node_uuid,
            COMMUNITY_TYPE,
            e.target_node_uuid,
            ENTITY_TYPE,
            data,
        )

    async def community_edge_delete(self, edge: Any, driver: Any) -> None:
        await _delete_node_by_uuid(_cast_driver(driver), edge.uuid, HAS_MEMBER_EDGE_TYPE)

    async def community_edge_delete_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str], group_id: str | None = None
    ) -> None:
        await _delete_nodes_by_uuids(_cast_driver(driver), uuids, HAS_MEMBER_EDGE_TYPE)

    async def community_edge_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> CommunityEdge:
        d = _cast_driver(driver)
        query = (
            f'{{ q(func: eq(graphiti.uuid, "{uuid}")) '
            f'@filter(type({HAS_MEMBER_EDGE_TYPE})) {{ {_SIMPLE_EDGE_FIELDS} }} }}'
        )
        records, _, _ = await d.execute_query(query)
        if not records:
            raise EdgeNotFoundError(uuid)
        return _build_simple_edge(records[0], CommunityEdge)

    async def community_edge_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[CommunityEdge]:
        if not uuids:
            return []
        d = _cast_driver(driver)
        uuid_list = ', '.join(f'"{u}"' for u in uuids)
        query = (
            f'{{ q(func: eq(graphiti.uuid, [{uuid_list}])) '
            f'@filter(type({HAS_MEMBER_EDGE_TYPE})) {{ {_SIMPLE_EDGE_FIELDS} }} }}'
        )
        records, _, _ = await d.execute_query(query)
        return [_build_simple_edge(r, CommunityEdge) for r in (records or [])]

    async def community_edge_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[CommunityEdge]:
        d = _cast_driver(driver)
        gid_list = ', '.join(f'"{g}"' for g in group_ids)
        cursor_filter = ''
        if uuid_cursor:
            cursor_filter = f' AND lt(graphiti.uuid, "{uuid_cursor}")'
        limit_clause = f', first: {limit}' if limit is not None else ''
        query = (
            f'{{ q(func: eq(graphiti.group_id, [{gid_list}]){limit_clause}'
            f', orderdesc: graphiti.uuid) '
            f'@filter(type({HAS_MEMBER_EDGE_TYPE}){cursor_filter}) '
            f'{{ {_SIMPLE_EDGE_FIELDS} }} }}'
        )
        records, _, _ = await d.execute_query(query)
        return [_build_simple_edge(r, CommunityEdge) for r in (records or [])]

    # -----------------------------------------------------------------------
    # HasEpisodeEdge: Save / Delete / Read
    # -----------------------------------------------------------------------

    async def has_episode_edge_save(self, edge: Any, driver: Any) -> None:
        d = _cast_driver(driver)
        e: HasEpisodeEdge = edge
        data: dict[str, Any] = {
            'graphiti.uuid': e.uuid,
            'graphiti.group_id': e.group_id,
            'graphiti.created_at': _dt_to_str(e.created_at),
        }
        await _upsert_edge_node(
            d,
            e.uuid,
            HAS_EPISODE_EDGE_TYPE,
            e.source_node_uuid,
            SAGA_TYPE,
            e.target_node_uuid,
            EPISODIC_TYPE,
            data,
        )

    async def has_episode_edge_delete(self, edge: Any, driver: Any) -> None:
        await _delete_node_by_uuid(_cast_driver(driver), edge.uuid, HAS_EPISODE_EDGE_TYPE)

    async def has_episode_edge_save_bulk(
        self, _cls: Any, driver: Any, transaction: Any, edges: list[Any], batch_size: int = 100
    ) -> None:
        for i in range(0, len(edges), batch_size):
            batch = edges[i : i + batch_size]
            await semaphore_gather(*[self.has_episode_edge_save(e, driver) for e in batch])

    async def has_episode_edge_delete_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str], group_id: str | None = None
    ) -> None:
        await _delete_nodes_by_uuids(_cast_driver(driver), uuids, HAS_EPISODE_EDGE_TYPE)

    async def has_episode_edge_get_by_uuid(
        self, _cls: Any, driver: Any, uuid: str
    ) -> HasEpisodeEdge:
        d = _cast_driver(driver)
        query = (
            f'{{ q(func: eq(graphiti.uuid, "{uuid}")) '
            f'@filter(type({HAS_EPISODE_EDGE_TYPE})) {{ {_SIMPLE_EDGE_FIELDS} }} }}'
        )
        records, _, _ = await d.execute_query(query)
        if not records:
            raise EdgeNotFoundError(uuid)
        return _build_simple_edge(records[0], HasEpisodeEdge)

    async def has_episode_edge_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[HasEpisodeEdge]:
        if not uuids:
            return []
        d = _cast_driver(driver)
        uuid_list = ', '.join(f'"{u}"' for u in uuids)
        query = (
            f'{{ q(func: eq(graphiti.uuid, [{uuid_list}])) '
            f'@filter(type({HAS_EPISODE_EDGE_TYPE})) {{ {_SIMPLE_EDGE_FIELDS} }} }}'
        )
        records, _, _ = await d.execute_query(query)
        return [_build_simple_edge(r, HasEpisodeEdge) for r in (records or [])]

    async def has_episode_edge_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[HasEpisodeEdge]:
        d = _cast_driver(driver)
        gid_list = ', '.join(f'"{g}"' for g in group_ids)
        cursor_filter = ''
        if uuid_cursor:
            cursor_filter = f' AND lt(graphiti.uuid, "{uuid_cursor}")'
        limit_clause = f', first: {limit}' if limit is not None else ''
        query = (
            f'{{ q(func: eq(graphiti.group_id, [{gid_list}]){limit_clause}'
            f', orderdesc: graphiti.uuid) '
            f'@filter(type({HAS_EPISODE_EDGE_TYPE}){cursor_filter}) '
            f'{{ {_SIMPLE_EDGE_FIELDS} }} }}'
        )
        records, _, _ = await d.execute_query(query)
        return [_build_simple_edge(r, HasEpisodeEdge) for r in (records or [])]

    # -----------------------------------------------------------------------
    # NextEpisodeEdge: Save / Delete / Read
    # -----------------------------------------------------------------------

    async def next_episode_edge_save(self, edge: Any, driver: Any) -> None:
        d = _cast_driver(driver)
        e: NextEpisodeEdge = edge
        data: dict[str, Any] = {
            'graphiti.uuid': e.uuid,
            'graphiti.group_id': e.group_id,
            'graphiti.created_at': _dt_to_str(e.created_at),
        }
        await _upsert_edge_node(
            d,
            e.uuid,
            NEXT_EPISODE_EDGE_TYPE,
            e.source_node_uuid,
            EPISODIC_TYPE,
            e.target_node_uuid,
            EPISODIC_TYPE,
            data,
        )

    async def next_episode_edge_delete(self, edge: Any, driver: Any) -> None:
        await _delete_node_by_uuid(_cast_driver(driver), edge.uuid, NEXT_EPISODE_EDGE_TYPE)

    async def next_episode_edge_save_bulk(
        self, _cls: Any, driver: Any, transaction: Any, edges: list[Any], batch_size: int = 100
    ) -> None:
        for i in range(0, len(edges), batch_size):
            batch = edges[i : i + batch_size]
            await semaphore_gather(*[self.next_episode_edge_save(e, driver) for e in batch])

    async def next_episode_edge_delete_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str], group_id: str | None = None
    ) -> None:
        await _delete_nodes_by_uuids(_cast_driver(driver), uuids, NEXT_EPISODE_EDGE_TYPE)

    async def next_episode_edge_get_by_uuid(
        self, _cls: Any, driver: Any, uuid: str
    ) -> NextEpisodeEdge:
        d = _cast_driver(driver)
        query = (
            f'{{ q(func: eq(graphiti.uuid, "{uuid}")) '
            f'@filter(type({NEXT_EPISODE_EDGE_TYPE})) {{ {_SIMPLE_EDGE_FIELDS} }} }}'
        )
        records, _, _ = await d.execute_query(query)
        if not records:
            raise EdgeNotFoundError(uuid)
        return _build_simple_edge(records[0], NextEpisodeEdge)

    async def next_episode_edge_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[NextEpisodeEdge]:
        if not uuids:
            return []
        d = _cast_driver(driver)
        uuid_list = ', '.join(f'"{u}"' for u in uuids)
        query = (
            f'{{ q(func: eq(graphiti.uuid, [{uuid_list}])) '
            f'@filter(type({NEXT_EPISODE_EDGE_TYPE})) {{ {_SIMPLE_EDGE_FIELDS} }} }}'
        )
        records, _, _ = await d.execute_query(query)
        return [_build_simple_edge(r, NextEpisodeEdge) for r in (records or [])]

    async def next_episode_edge_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[NextEpisodeEdge]:
        d = _cast_driver(driver)
        gid_list = ', '.join(f'"{g}"' for g in group_ids)
        cursor_filter = ''
        if uuid_cursor:
            cursor_filter = f' AND lt(graphiti.uuid, "{uuid_cursor}")'
        limit_clause = f', first: {limit}' if limit is not None else ''
        query = (
            f'{{ q(func: eq(graphiti.group_id, [{gid_list}]){limit_clause}'
            f', orderdesc: graphiti.uuid) '
            f'@filter(type({NEXT_EPISODE_EDGE_TYPE}){cursor_filter}) '
            f'{{ {_SIMPLE_EDGE_FIELDS} }} }}'
        )
        records, _, _ = await d.execute_query(query)
        return [_build_simple_edge(r, NextEpisodeEdge) for r in (records or [])]

    # -----------------------------------------------------------------------
    # Search helpers
    # -----------------------------------------------------------------------

    async def get_mentioned_nodes(self, driver: Any, episodes: list[Any]) -> list[EntityNode]:
        d = _cast_driver(driver)
        if not episodes:
            return []
        episode_uuids = [ep.uuid for ep in episodes]
        uuid_list = ', '.join(f'"{u}"' for u in episode_uuids)

        # Episodic -> ~edge_source(MentionsEdge) -> edge_target(Entity)
        query = (
            f'{{ q(func: eq(graphiti.uuid, [{uuid_list}])) '
            f'@filter(type({EPISODIC_TYPE})) {{ '
            f'~graphiti.edge_source @filter(type({MENTIONS_EDGE_TYPE})) {{ '
            f'graphiti.edge_target @filter(type({ENTITY_TYPE})) {{ {_ENTITY_FIELDS} }} '
            f'}} }} }}'
        )
        data = await d.execute_query_raw(query)
        seen: set[str] = set()
        nodes: list[EntityNode] = []
        for ep_rec in data.get('q', []):
            for edge_rec in ep_rec.get('~graphiti.edge_source', []):
                targets = edge_rec.get('graphiti.edge_target', [])
                if not isinstance(targets, list):
                    targets = [targets]
                for t in targets:
                    if t and 'graphiti.uuid' in t and t['graphiti.uuid'] not in seen:
                        seen.add(t['graphiti.uuid'])
                        nodes.append(_build_entity_node(t))
        return nodes

    async def get_communities_by_nodes(self, driver: Any, nodes: list[Any]) -> list[CommunityNode]:
        d = _cast_driver(driver)
        if not nodes:
            return []
        node_uuids = [n.uuid for n in nodes]
        uuid_list = ', '.join(f'"{u}"' for u in node_uuids)

        # Entity -> ~edge_target(HasMemberEdge) -> edge_source(Community)
        query = (
            f'{{ q(func: eq(graphiti.uuid, [{uuid_list}])) '
            f'@filter(type({ENTITY_TYPE})) {{ '
            f'~graphiti.edge_target @filter(type({HAS_MEMBER_EDGE_TYPE})) {{ '
            f'graphiti.edge_source @filter(type({COMMUNITY_TYPE})) {{ {_COMMUNITY_FIELDS} }} '
            f'}} }} }}'
        )
        data = await d.execute_query_raw(query)
        seen: set[str] = set()
        communities: list[CommunityNode] = []
        for ent_rec in data.get('q', []):
            for edge_rec in ent_rec.get('~graphiti.edge_target', []):
                sources = edge_rec.get('graphiti.edge_source', [])
                if not isinstance(sources, list):
                    sources = [sources]
                for s in sources:
                    if s and 'graphiti.uuid' in s and s['graphiti.uuid'] not in seen:
                        seen.add(s['graphiti.uuid'])
                        communities.append(_build_community_node(s))
        return communities

    # -----------------------------------------------------------------------
    # Maintenance
    # -----------------------------------------------------------------------

    async def clear_data(self, driver: Any, group_ids: list[str] | None = None) -> None:
        d = _cast_driver(driver)
        if group_ids is None:
            from graphiti_core.driver.dgraph_driver import DGRAPH_SCHEMA

            await d.drop_all()
            await d.alter(DGRAPH_SCHEMA)
        else:
            all_types = [
                ENTITY_TYPE,
                EPISODIC_TYPE,
                COMMUNITY_TYPE,
                SAGA_TYPE,
                RELATES_TO_EDGE_TYPE,
                MENTIONS_EDGE_TYPE,
                HAS_MEMBER_EDGE_TYPE,
                HAS_EPISODE_EDGE_TYPE,
                NEXT_EPISODE_EDGE_TYPE,
            ]
            for group_id in group_ids:
                for t in all_types:
                    await _delete_nodes_by_group_id(d, group_id, t)

    async def get_community_clusters(
        self, driver: Any, group_ids: list[str] | None
    ) -> list[list[EntityNode]]:
        from graphiti_core.utils.maintenance.community_operations import (
            Neighbor,
            label_propagation,
        )

        d = _cast_driver(driver)
        clusters: list[list[EntityNode]] = []

        if group_ids is None:
            query = (
                f'{{ q(func: type({ENTITY_TYPE})) @groupby(graphiti.group_id) {{ count(uid) }} }}'
            )
            data = await d.execute_query_raw(query)
            group_ids = []
            for rec in data.get('q', []):
                for item in rec.get('@groupby', []):
                    gid = item.get('graphiti.group_id')
                    if gid is not None:
                        group_ids.append(gid)

        for group_id in group_ids:
            nodes = await self.node_get_by_group_ids(None, d, [group_id])
            projection: dict[str, list[Neighbor]] = {}

            for node in nodes:
                uid = await _find_uid(d, node.uuid, ENTITY_TYPE)
                if not uid:
                    projection[node.uuid] = []
                    continue

                # Find neighbors via RelatesToEdge in both directions
                q = (
                    f'{{ '
                    f'src(func: uid({uid})) {{ '
                    f'~graphiti.edge_source @filter(type({RELATES_TO_EDGE_TYPE})) {{ '
                    f'graphiti.edge_target @filter(type({ENTITY_TYPE}) '
                    f'AND eq(graphiti.group_id, "{group_id}")) {{ graphiti.uuid }} '
                    f'}} }} '
                    f'tgt(func: uid({uid})) {{ '
                    f'~graphiti.edge_target @filter(type({RELATES_TO_EDGE_TYPE})) {{ '
                    f'graphiti.edge_source @filter(type({ENTITY_TYPE}) '
                    f'AND eq(graphiti.group_id, "{group_id}")) {{ graphiti.uuid }} '
                    f'}} }} }}'
                )
                raw = await d.execute_query_raw(q)
                neighbor_counts: dict[str, int] = defaultdict(int)
                for rec in raw.get('src', []):
                    for er in rec.get('~graphiti.edge_source', []):
                        for t in (
                            er.get('graphiti.edge_target', [])
                            if isinstance(er.get('graphiti.edge_target'), list)
                            else [er.get('graphiti.edge_target', {})]
                        ):
                            if t and 'graphiti.uuid' in t:
                                neighbor_counts[t['graphiti.uuid']] += 1
                for rec in raw.get('tgt', []):
                    for er in rec.get('~graphiti.edge_target', []):
                        for s in (
                            er.get('graphiti.edge_source', [])
                            if isinstance(er.get('graphiti.edge_source'), list)
                            else [er.get('graphiti.edge_source', {})]
                        ):
                            if s and 'graphiti.uuid' in s:
                                neighbor_counts[s['graphiti.uuid']] += 1

                projection[node.uuid] = [
                    Neighbor(node_uuid=n_uuid, edge_count=cnt)
                    for n_uuid, cnt in neighbor_counts.items()
                ]

            cluster_uuids = label_propagation(projection)
            for cluster in cluster_uuids:
                cluster_nodes = await self.node_get_by_uuids(None, d, cluster)
                clusters.append(cluster_nodes)

        return clusters

    async def remove_communities(self, driver: Any) -> None:
        d = _cast_driver(driver)
        # Delete HasMemberEdge nodes
        query = f'{{ q(func: type({HAS_MEMBER_EDGE_TYPE})) {{ uid }} }}'
        records, _, _ = await d.execute_query(query)
        if records:
            await d.mutate_json(delete_json=[{'uid': r['uid']} for r in records])
        # Delete Community nodes
        query = f'{{ q(func: type({COMMUNITY_TYPE})) {{ uid }} }}'
        records, _, _ = await d.execute_query(query)
        if records:
            await d.mutate_json(delete_json=[{'uid': r['uid']} for r in records])

    async def determine_entity_community(
        self, driver: Any, entity: Any
    ) -> tuple[CommunityNode | None, bool]:
        d = _cast_driver(driver)
        entity_uid = await _find_uid(d, entity.uuid, ENTITY_TYPE)
        if not entity_uid:
            return None, False

        # Check if already in a community
        q = (
            f'{{ q(func: uid({entity_uid})) {{ '
            f'~graphiti.edge_target @filter(type({HAS_MEMBER_EDGE_TYPE})) {{ '
            f'graphiti.edge_source @filter(type({COMMUNITY_TYPE})) {{ {_COMMUNITY_FIELDS} }} '
            f'}} }} }}'
        )
        data = await d.execute_query_raw(q)
        for rec in data.get('q', []):
            for er in rec.get('~graphiti.edge_target', []):
                sources = er.get('graphiti.edge_source', [])
                if not isinstance(sources, list):
                    sources = [sources]
                for comm_rec in sources:
                    if comm_rec and 'graphiti.uuid' in comm_rec:
                        return _build_community_node(comm_rec), False

        # Find neighboring entities via RelatesToEdge
        q_nbrs = (
            f'{{ '
            f'src(func: uid({entity_uid})) {{ '
            f'~graphiti.edge_source @filter(type({RELATES_TO_EDGE_TYPE})) {{ '
            f'graphiti.edge_target @filter(type({ENTITY_TYPE})) {{ uid }} '
            f'}} }} '
            f'tgt(func: uid({entity_uid})) {{ '
            f'~graphiti.edge_target @filter(type({RELATES_TO_EDGE_TYPE})) {{ '
            f'graphiti.edge_source @filter(type({ENTITY_TYPE})) {{ uid }} '
            f'}} }} }}'
        )
        data = await d.execute_query_raw(q_nbrs)
        neighbor_uids: set[str] = set()
        for rec in data.get('src', []):
            for er in rec.get('~graphiti.edge_source', []):
                targets = er.get('graphiti.edge_target', [])
                if not isinstance(targets, list):
                    targets = [targets]
                for t in targets:
                    if t and 'uid' in t:
                        neighbor_uids.add(t['uid'])
        for rec in data.get('tgt', []):
            for er in rec.get('~graphiti.edge_target', []):
                sources = er.get('graphiti.edge_source', [])
                if not isinstance(sources, list):
                    sources = [sources]
                for s in sources:
                    if s and 'uid' in s:
                        neighbor_uids.add(s['uid'])

        if not neighbor_uids:
            return None, False

        # Find communities of those neighbors
        uid_list = ', '.join(neighbor_uids)
        q_comms = (
            f'{{ q(func: uid({uid_list})) {{ '
            f'~graphiti.edge_target @filter(type({HAS_MEMBER_EDGE_TYPE})) {{ '
            f'graphiti.edge_source @filter(type({COMMUNITY_TYPE})) {{ {_COMMUNITY_FIELDS} }} '
            f'}} }} }}'
        )
        data = await d.execute_query_raw(q_comms)
        communities: list[CommunityNode] = []
        for rec in data.get('q', []):
            for er in rec.get('~graphiti.edge_target', []):
                sources = er.get('graphiti.edge_source', [])
                if not isinstance(sources, list):
                    sources = [sources]
                for comm_rec in sources:
                    if comm_rec and 'graphiti.uuid' in comm_rec:
                        communities.append(_build_community_node(comm_rec))

        if not communities:
            return None, False

        # Pick the most common community
        community_map: dict[str, int] = defaultdict(int)
        for c in communities:
            community_map[c.uuid] += 1

        best_uuid = max(community_map, key=lambda u: community_map[u])
        for c in communities:
            if c.uuid == best_uuid:
                return c, True

        return None, False

    # -----------------------------------------------------------------------
    # Additional Node Operations
    # -----------------------------------------------------------------------

    async def episodic_node_get_by_entity_node_uuid(
        self, _cls: Any, driver: Any, entity_node_uuid: str
    ) -> list[EpisodicNode]:
        d = _cast_driver(driver)
        entity_uid = await _find_uid(d, entity_node_uuid, ENTITY_TYPE)
        if not entity_uid:
            return []

        # Entity -> ~edge_target(MentionsEdge) -> edge_source(Episodic)
        query = (
            f'{{ q(func: uid({entity_uid})) {{ '
            f'~graphiti.edge_target @filter(type({MENTIONS_EDGE_TYPE})) {{ '
            f'graphiti.edge_source @filter(type({EPISODIC_TYPE})) {{ {_EPISODIC_FIELDS} }} '
            f'}} }} }}'
        )
        data = await d.execute_query_raw(query)
        seen: set[str] = set()
        episodes: list[EpisodicNode] = []
        for rec in data.get('q', []):
            for er in rec.get('~graphiti.edge_target', []):
                sources = er.get('graphiti.edge_source', [])
                if not isinstance(sources, list):
                    sources = [sources]
                for ep_rec in sources:
                    if ep_rec and 'graphiti.uuid' in ep_rec and ep_rec['graphiti.uuid'] not in seen:
                        seen.add(ep_rec['graphiti.uuid'])
                        episodes.append(_build_episodic_node(ep_rec))
        return episodes

    async def community_node_load_name_embedding(self, node: Any, driver: Any) -> None:
        d = _cast_driver(driver)
        query = (
            f'{{ q(func: eq(graphiti.uuid, "{node.uuid}")) '
            f'@filter(type({COMMUNITY_TYPE})) {{ graphiti.name_embedding }} }}'
        )
        records, _, _ = await d.execute_query(query)
        if not records:
            raise NodeNotFoundError(node.uuid)
        node.name_embedding = records[0].get('graphiti.name_embedding')

    # -----------------------------------------------------------------------
    # Saga / Episode helpers
    # -----------------------------------------------------------------------

    async def saga_node_get_by_name_and_group(
        self, driver: Any, name: str, group_id: str
    ) -> SagaNode | None:
        d = _cast_driver(driver)
        query = (
            f'{{ q(func: eq(graphiti.name, "{name}")) '
            f'@filter(type({SAGA_TYPE}) AND eq(graphiti.group_id, "{group_id}")) '
            f'{{ {_SAGA_FIELDS} }} }}'
        )
        records, _, _ = await d.execute_query(query)
        if not records:
            return None
        return _build_saga_node(records[0])

    async def get_latest_saga_episode(
        self, driver: Any, saga_uuid: str, exclude_uuid: str | None = None
    ) -> str | None:
        d = _cast_driver(driver)
        saga_uid = await _find_uid(d, saga_uuid, SAGA_TYPE)
        if not saga_uid:
            return None

        query = (
            f'{{ q(func: uid({saga_uid})) {{ '
            f'~graphiti.edge_source @filter(type({HAS_EPISODE_EDGE_TYPE})) {{ '
            f'graphiti.edge_target @filter(type({EPISODIC_TYPE})) '
            f'{{ graphiti.uuid graphiti.valid_at }} '
            f'}} }} }}'
        )
        data = await d.execute_query_raw(query)
        candidates: list[tuple[str, datetime | None]] = []
        for rec in data.get('q', []):
            for er in rec.get('~graphiti.edge_source', []):
                targets = er.get('graphiti.edge_target', [])
                if not isinstance(targets, list):
                    targets = [targets]
                for ep_rec in targets:
                    if not ep_rec or 'graphiti.uuid' not in ep_rec:
                        continue
                    ep_uuid = ep_rec['graphiti.uuid']
                    if exclude_uuid and ep_uuid == exclude_uuid:
                        continue
                    candidates.append((ep_uuid, _parse_dt(ep_rec.get('graphiti.valid_at'))))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1] or datetime.min, reverse=True)
        return candidates[0][0]

    async def count_entity_episode_mentions(self, driver: Any, entity_uuid: str) -> int:
        d = _cast_driver(driver)
        entity_uid = await _find_uid(d, entity_uuid, ENTITY_TYPE)
        if not entity_uid:
            return 0

        query = (
            f'{{ q(func: uid({entity_uid})) {{ '
            f'count: count(~graphiti.edge_target @filter(type({MENTIONS_EDGE_TYPE}))) '
            f'}} }}'
        )
        records, _, _ = await d.execute_query(query)
        if records and len(records) > 0:
            return records[0].get('count', 0)
        return 0

    # -----------------------------------------------------------------------
    # Additional Edge Operations
    # -----------------------------------------------------------------------

    async def edge_get_between_nodes(
        self,
        _cls: Any,
        driver: Any,
        source_node_uuid: str,
        target_node_uuid: str,
    ) -> list[EntityEdge]:
        d = _cast_driver(driver)
        src_uid = await _find_uid(d, source_node_uuid, ENTITY_TYPE)
        tgt_uid = await _find_uid(d, target_node_uuid, ENTITY_TYPE)
        if not src_uid or not tgt_uid:
            return []

        query = (
            f'{{ q(func: type({RELATES_TO_EDGE_TYPE})) '
            f'@filter(uid_in(graphiti.edge_source, {src_uid}) '
            f'AND uid_in(graphiti.edge_target, {tgt_uid})) '
            f'{{ {_RELATES_TO_EDGE_FIELDS} }} }}'
        )
        records, _, _ = await d.execute_query(query)
        return [_build_entity_edge(r) for r in (records or [])]

    async def edge_get_by_node_uuid(
        self, _cls: Any, driver: Any, node_uuid: str
    ) -> list[EntityEdge]:
        d = _cast_driver(driver)
        node_uid = await _find_uid(d, node_uuid, ENTITY_TYPE)
        if not node_uid:
            return []

        query = (
            f'{{ q(func: type({RELATES_TO_EDGE_TYPE})) '
            f'@filter(uid_in(graphiti.edge_source, {node_uid}) '
            f'OR uid_in(graphiti.edge_target, {node_uid})) '
            f'{{ {_RELATES_TO_EDGE_FIELDS} }} }}'
        )
        records, _, _ = await d.execute_query(query)
        return [_build_entity_edge(r) for r in (records or [])]
