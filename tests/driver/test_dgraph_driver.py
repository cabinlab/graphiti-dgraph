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

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from graphiti_core.driver.dgraph_driver import (
    DGRAPH_SCHEMA,
    DgraphDriver,
    DgraphDriverSession,
)
from graphiti_core.driver.dgraph_graph_operations import (
    DgraphGraphOperations,
    _build_entity_node,
    _build_episodic_node,
    _build_saga_node,
    _dt_to_str,
    _parse_dt,
)
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.graph_operations.graph_operations import GraphOperationsInterface


class TestDgraphDriver:
    """Tests for DgraphDriver class."""

    def test_provider(self):
        """Test that provider is set correctly."""
        with patch('graphiti_core.driver.dgraph_driver.httpx') as mock_httpx:
            mock_httpx.AsyncClient.return_value = MagicMock()
            mock_httpx.Limits.return_value = MagicMock()
            mock_httpx.Timeout.return_value = MagicMock()
            driver = DgraphDriver(url='http://localhost:8080')
            assert driver.provider == GraphProvider.DGRAPH

    def test_default_url(self):
        """Test default URL is set."""
        with patch('graphiti_core.driver.dgraph_driver.httpx') as mock_httpx:
            mock_httpx.AsyncClient.return_value = MagicMock()
            mock_httpx.Limits.return_value = MagicMock()
            mock_httpx.Timeout.return_value = MagicMock()
            driver = DgraphDriver()
            assert driver._url == 'http://localhost:8080'

    def test_url_trailing_slash_stripped(self):
        """Test trailing slash is stripped from URL."""
        with patch('graphiti_core.driver.dgraph_driver.httpx') as mock_httpx:
            mock_httpx.AsyncClient.return_value = MagicMock()
            mock_httpx.Limits.return_value = MagicMock()
            mock_httpx.Timeout.return_value = MagicMock()
            driver = DgraphDriver(url='http://localhost:8080/')
            assert driver._url == 'http://localhost:8080'


class TestDgraphDriverSession:
    """Tests for DgraphDriverSession class."""

    def test_provider(self):
        """Test session provider is DGRAPH."""
        mock_driver = MagicMock()
        session = DgraphDriverSession(mock_driver)
        assert session.provider == GraphProvider.DGRAPH


class TestDgraphSchema:
    """Tests for the Dgraph schema."""

    def test_schema_contains_all_types(self):
        """Test that the schema defines all required types."""
        required_types = [
            'type Entity',
            'type Episodic',
            'type Community',
            'type Saga',
            'type RelatesToEdge',
            'type MentionsEdge',
            'type HasMemberEdge',
            'type HasEpisodeEdge',
            'type NextEpisodeEdge',
        ]
        for t in required_types:
            assert t in DGRAPH_SCHEMA, f'Missing type: {t}'

    def test_schema_contains_vector_predicates(self):
        """Test vector predicates are defined."""
        assert 'graphiti.name_embedding: float32vector' in DGRAPH_SCHEMA
        assert 'graphiti.fact_embedding: float32vector' in DGRAPH_SCHEMA

    def test_schema_contains_uuid_index(self):
        """Test UUID has exact index and upsert directive."""
        assert 'graphiti.uuid: string @index(exact)' in DGRAPH_SCHEMA

    def test_schema_contains_edge_predicates(self):
        """Test edge source/target predicates."""
        assert 'graphiti.edge_source: uid @reverse' in DGRAPH_SCHEMA
        assert 'graphiti.edge_target: uid @reverse' in DGRAPH_SCHEMA


class TestHelpers:
    """Tests for helper functions."""

    def test_dt_to_str_none(self):
        assert _dt_to_str(None) is None

    def test_dt_to_str_datetime(self):
        dt = datetime(2024, 1, 15, 12, 30, 0, tzinfo=timezone.utc)
        result = _dt_to_str(dt)
        assert '2024-01-15' in result  # type: ignore
        assert '12:30:00' in result  # type: ignore

    def test_parse_dt_none(self):
        assert _parse_dt(None) is None

    def test_parse_dt_empty_string(self):
        assert _parse_dt('') is None

    def test_parse_dt_valid(self):
        result = _parse_dt('2024-01-15T12:30:00+00:00')
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15


class TestBuildHelpers:
    """Tests for _build_* helper functions."""

    def test_build_entity_node(self):
        record = {
            'graphiti.uuid': 'test-uuid',
            'graphiti.name': 'Test Entity',
            'graphiti.group_id': 'group-1',
            'graphiti.labels': ['Person', 'Entity'],
            'graphiti.created_at': '2024-01-15T12:00:00+00:00',
            'graphiti.summary': 'A test entity',
            'graphiti.attributes': '{"key": "value"}',
        }
        node = _build_entity_node(record)
        assert node.uuid == 'test-uuid'
        assert node.name == 'Test Entity'
        assert node.group_id == 'group-1'
        assert 'Person' in node.labels
        assert node.summary == 'A test entity'
        assert node.attributes == {'key': 'value'}

    def test_build_episodic_node(self):
        record = {
            'graphiti.uuid': 'ep-uuid',
            'graphiti.name': 'Episode 1',
            'graphiti.group_id': 'group-1',
            'graphiti.created_at': '2024-01-15T12:00:00+00:00',
            'graphiti.valid_at': '2024-01-15T12:00:00+00:00',
            'graphiti.source': 'text',
            'graphiti.source_description': 'test source',
            'graphiti.content': 'Test content',
            'graphiti.entity_edges': '["edge-1", "edge-2"]',
        }
        node = _build_episodic_node(record)
        assert node.uuid == 'ep-uuid'
        assert node.content == 'Test content'
        assert node.entity_edges == ['edge-1', 'edge-2']

    def test_build_saga_node(self):
        record = {
            'graphiti.uuid': 'saga-uuid',
            'graphiti.name': 'Test Saga',
            'graphiti.group_id': 'group-1',
            'graphiti.created_at': '2024-01-15T12:00:00+00:00',
        }
        node = _build_saga_node(record)
        assert node.uuid == 'saga-uuid'
        assert node.name == 'Test Saga'


class TestDgraphGraphOperations:
    """Tests for DgraphGraphOperations class."""

    def test_is_graph_operations_interface(self):
        """Test that DgraphGraphOperations is a valid GraphOperationsInterface."""
        ops = DgraphGraphOperations()
        assert isinstance(ops, GraphOperationsInterface)

    def test_has_all_required_methods(self):
        """Test that all interface methods are present."""
        ops = DgraphGraphOperations()
        required_methods = [
            'node_save',
            'node_delete',
            'node_get_by_uuid',
            'node_get_by_uuids',
            'node_get_by_group_ids',
            'node_load_embeddings',
            'episodic_node_save',
            'episodic_node_delete',
            'episodic_node_get_by_uuid',
            'episodic_node_get_by_uuids',
            'community_node_save',
            'community_node_delete',
            'saga_node_save',
            'saga_node_delete',
            'edge_save',
            'edge_delete',
            'edge_get_by_uuid',
            'edge_get_by_uuids',
            'episodic_edge_save',
            'episodic_edge_delete',
            'community_edge_save',
            'community_edge_delete',
            'has_episode_edge_save',
            'has_episode_edge_delete',
            'next_episode_edge_save',
            'next_episode_edge_delete',
            'get_mentioned_nodes',
            'get_communities_by_nodes',
            'clear_data',
            'get_community_clusters',
            'remove_communities',
            'determine_entity_community',
            'saga_node_get_by_name_and_group',
            'get_latest_saga_episode',
            'count_entity_episode_mentions',
            'edge_get_between_nodes',
            'edge_get_by_node_uuid',
        ]
        for method in required_methods:
            assert hasattr(ops, method), f'Missing method: {method}'
            assert callable(getattr(ops, method)), f'Not callable: {method}'


class TestDgraphSearchInterface:
    """Tests for DgraphSearchInterface class."""

    def test_is_search_interface(self):
        """Test that DgraphSearchInterface is a valid SearchInterface."""
        from graphiti_core.driver.dgraph_search import DgraphSearchInterface
        from graphiti_core.driver.search_interface.search_interface import SearchInterface

        search = DgraphSearchInterface()
        assert isinstance(search, SearchInterface)

    def test_has_all_required_methods(self):
        """Test that all search interface methods are present."""
        from graphiti_core.driver.dgraph_search import DgraphSearchInterface

        search = DgraphSearchInterface()
        required_methods = [
            'edge_fulltext_search',
            'edge_similarity_search',
            'node_fulltext_search',
            'node_similarity_search',
            'episode_fulltext_search',
            'edge_bfs_search',
            'node_bfs_search',
            'community_fulltext_search',
            'community_similarity_search',
            'get_embeddings_for_communities',
            'node_distance_reranker',
            'episode_mentions_reranker',
            'build_node_search_filters',
            'build_edge_search_filters',
        ]
        for method in required_methods:
            assert hasattr(search, method), f'Missing method: {method}'
            assert callable(getattr(search, method)), f'Not callable: {method}'


class TestGraphProviderEnum:
    """Tests for DGRAPH enum value."""

    def test_dgraph_enum_exists(self):
        assert GraphProvider.DGRAPH.value == 'dgraph'

    def test_dgraph_in_providers(self):
        providers = [p.value for p in GraphProvider]
        assert 'dgraph' in providers
