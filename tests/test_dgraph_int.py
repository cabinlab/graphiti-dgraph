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

Integration tests for Dgraph backend.
Requires a running Dgraph instance (docker compose -f docker-compose.dgraph.yml up -d).
"""

import os
from datetime import datetime, timezone

import pytest

from graphiti_core.driver.dgraph_driver import DgraphDriver
from graphiti_core.driver.dgraph_graph_operations import DgraphGraphOperations
from graphiti_core.driver.dgraph_search import DgraphSearchInterface
from graphiti_core.edges import (
    EntityEdge,
    EpisodicEdge,
)
from graphiti_core.nodes import (
    EntityNode,
    EpisodeType,
    EpisodicNode,
    SagaNode,
)

DGRAPH_URL = os.environ.get('DGRAPH_URL', 'http://localhost:8080')


@pytest.fixture(scope='module')
def driver():
    """Create a DgraphDriver with interfaces attached."""
    d = DgraphDriver(url=DGRAPH_URL)
    d.graph_operations_interface = DgraphGraphOperations()
    d.search_interface = DgraphSearchInterface()
    return d


@pytest.fixture(autouse=True)
async def setup_and_teardown(driver):
    """Set up schema and clean up after each test."""
    await driver.build_indices_and_constraints(delete_existing=True)
    yield
    await driver.graph_operations_interface.clear_data(driver)


@pytest.fixture
def now():
    return datetime.now(timezone.utc)


# ---------- Health ----------


@pytest.mark.asyncio
async def test_health_check_int(driver):
    await driver.health_check()


# ---------- Entity Node CRUD ----------


@pytest.mark.asyncio
async def test_entity_node_crud_int(driver, now):
    ops = driver.graph_operations_interface
    node = EntityNode(
        uuid='test-entity-1',
        name='Alice',
        group_id='test-group',
        labels=['Person'],
        created_at=now,
        summary='A test entity',
        attributes={'age': 30},
    )

    # Save
    await ops.node_save(node, driver)

    # Read
    fetched = await ops.node_get_by_uuid(EntityNode, driver, 'test-entity-1')
    assert fetched.uuid == 'test-entity-1'
    assert fetched.name == 'Alice'
    assert fetched.group_id == 'test-group'

    # Read multiple
    nodes = await ops.node_get_by_uuids(EntityNode, driver, ['test-entity-1'])
    assert len(nodes) == 1

    # Read by group
    nodes = await ops.node_get_by_group_ids(EntityNode, driver, ['test-group'])
    assert len(nodes) >= 1

    # Delete
    await ops.node_delete(node, driver)


# ---------- Episodic Node CRUD ----------


@pytest.mark.asyncio
async def test_episodic_node_crud_int(driver, now):
    ops = driver.graph_operations_interface
    ep = EpisodicNode(
        uuid='test-ep-1',
        name='Episode 1',
        group_id='test-group',
        created_at=now,
        valid_at=now,
        source=EpisodeType.text,
        source_description='test',
        content='Hello world',
        entity_edges=[],
    )

    await ops.episodic_node_save(ep, driver)
    fetched = await ops.episodic_node_get_by_uuid(EpisodicNode, driver, 'test-ep-1')
    assert fetched.uuid == 'test-ep-1'
    assert fetched.content == 'Hello world'

    await ops.episodic_node_delete(ep, driver)


# ---------- Saga Node CRUD ----------


@pytest.mark.asyncio
async def test_saga_node_crud_int(driver, now):
    ops = driver.graph_operations_interface
    saga = SagaNode(
        uuid='test-saga-1',
        name='Test Saga',
        group_id='test-group',
        created_at=now,
    )

    await ops.saga_node_save(saga, driver)
    fetched = await ops.saga_node_get_by_uuid(SagaNode, driver, 'test-saga-1')
    assert fetched.uuid == 'test-saga-1'
    assert fetched.name == 'Test Saga'

    # Test get_by_name_and_group
    found = await ops.saga_node_get_by_name_and_group(driver, 'Test Saga', 'test-group')
    assert found is not None
    assert found.uuid == 'test-saga-1'

    not_found = await ops.saga_node_get_by_name_and_group(driver, 'No Such Saga', 'test-group')
    assert not_found is None

    await ops.saga_node_delete(saga, driver)


# ---------- Entity Edge CRUD ----------


@pytest.mark.asyncio
async def test_entity_edge_crud_int(driver, now):
    ops = driver.graph_operations_interface

    # Create two entity nodes first
    node1 = EntityNode(
        uuid='edge-test-node-1',
        name='Node A',
        group_id='test-group',
        created_at=now,
        summary='',
    )
    node2 = EntityNode(
        uuid='edge-test-node-2',
        name='Node B',
        group_id='test-group',
        created_at=now,
        summary='',
    )
    await ops.node_save(node1, driver)
    await ops.node_save(node2, driver)

    edge = EntityEdge(
        uuid='test-edge-1',
        name='knows',
        group_id='test-group',
        source_node_uuid='edge-test-node-1',
        target_node_uuid='edge-test-node-2',
        fact='A knows B',
        episodes=['ep-1'],
        created_at=now,
    )
    await ops.edge_save(edge, driver)

    fetched = await ops.edge_get_by_uuid(EntityEdge, driver, 'test-edge-1')
    assert fetched.uuid == 'test-edge-1'
    assert fetched.fact == 'A knows B'
    assert fetched.source_node_uuid == 'edge-test-node-1'
    assert fetched.target_node_uuid == 'edge-test-node-2'

    # Get between nodes
    between = await ops.edge_get_between_nodes(
        EntityEdge, driver, 'edge-test-node-1', 'edge-test-node-2'
    )
    assert len(between) >= 1

    # Get by node
    by_node = await ops.edge_get_by_node_uuid(EntityEdge, driver, 'edge-test-node-1')
    assert len(by_node) >= 1

    await ops.edge_delete(edge, driver)


# ---------- Episodic Edge CRUD ----------


@pytest.mark.asyncio
async def test_episodic_edge_crud_int(driver, now):
    ops = driver.graph_operations_interface

    ep = EpisodicNode(
        uuid='ep-edge-test-1',
        name='Ep 1',
        group_id='test-group',
        created_at=now,
        valid_at=now,
        source=EpisodeType.text,
        source_description='test',
        content='test',
    )
    entity = EntityNode(
        uuid='ep-edge-entity-1',
        name='Entity 1',
        group_id='test-group',
        created_at=now,
        summary='',
    )
    await ops.episodic_node_save(ep, driver)
    await ops.node_save(entity, driver)

    edge = EpisodicEdge(
        uuid='ep-edge-1',
        group_id='test-group',
        source_node_uuid='ep-edge-test-1',
        target_node_uuid='ep-edge-entity-1',
        created_at=now,
    )
    await ops.episodic_edge_save(edge, driver)

    fetched = await ops.episodic_edge_get_by_uuid(EpisodicEdge, driver, 'ep-edge-1')
    assert fetched.uuid == 'ep-edge-1'

    # Test mention counting
    count = await ops.count_entity_episode_mentions(driver, 'ep-edge-entity-1')
    assert count >= 1


# ---------- Clear Data ----------


@pytest.mark.asyncio
async def test_clear_data_by_group_int(driver, now):
    ops = driver.graph_operations_interface
    node = EntityNode(
        uuid='clear-test-1',
        name='Clear Me',
        group_id='clear-group',
        created_at=now,
        summary='',
    )
    await ops.node_save(node, driver)

    await ops.clear_data(driver, group_ids=['clear-group'])

    from graphiti_core.errors import NodeNotFoundError

    with pytest.raises(NodeNotFoundError):
        await ops.node_get_by_uuid(EntityNode, driver, 'clear-test-1')
