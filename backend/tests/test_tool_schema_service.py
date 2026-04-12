"""Tests for tool_schema_service — schema conversion for innate skills and external tools."""

import pytest
from unittest.mock import patch, MagicMock

from services.tool_schema_service import (
    get_external_tool_schemas,
    _manifest_to_schema,
)


pytestmark = pytest.mark.unit


class TestManifestToSchema:
    """Tests for _manifest_to_schema — manifest dict to native tool format."""

    def test_basic_conversion(self):
        manifest = {
            'name': 'weather_api',
            'description': 'Get weather forecasts',
            'parameters': {
                'city': {
                    'type': 'string',
                    'required': True,
                    'description': 'City name',
                },
                'units': {
                    'type': 'string',
                    'required': False,
                    'description': 'Temperature units',
                    'default': 'metric',
                },
            },
        }
        schema = _manifest_to_schema(manifest)

        assert schema['name'] == 'weather_api'
        assert schema['description'] == 'Get weather forecasts'
        assert schema['input_schema']['type'] == 'object'
        assert 'city' in schema['input_schema']['properties']
        assert 'units' in schema['input_schema']['properties']
        assert schema['input_schema']['required'] == ['city']
        assert schema['input_schema']['properties']['units']['default'] == 'metric'

    def test_no_parameters(self):
        manifest = {
            'name': 'simple_tool',
            'description': 'Does a thing',
            'parameters': {},
        }
        schema = _manifest_to_schema(manifest)
        assert schema['name'] == 'simple_tool'
        assert schema['input_schema']['properties'] == {}
        assert schema['input_schema']['required'] == []

    def test_missing_name_returns_none(self):
        manifest = {'description': 'no name'}
        assert _manifest_to_schema(manifest) is None

    def test_enum_parameter(self):
        manifest = {
            'name': 'formatter',
            'description': 'Format text',
            'parameters': {
                'format': {
                    'type': 'string',
                    'required': True,
                    'enum': ['json', 'xml', 'csv'],
                },
            },
        }
        schema = _manifest_to_schema(manifest)
        assert schema['input_schema']['properties']['format']['enum'] == ['json', 'xml', 'csv']

    def test_missing_parameters_key(self):
        manifest = {'name': 'bare', 'description': 'Bare tool'}
        schema = _manifest_to_schema(manifest)
        assert schema is not None
        assert schema['input_schema']['properties'] == {}

    def test_defaults_type_to_string(self):
        manifest = {
            'name': 'tool',
            'description': 'test',
            'parameters': {
                'arg': {'required': True},
            },
        }
        schema = _manifest_to_schema(manifest)
        assert schema['input_schema']['properties']['arg']['type'] == 'string'


class TestGetExternalToolSchemas:

    @patch('services.tool_registry_service.ToolRegistryService')
    def test_converts_registered_tools(self, MockRegistry):
        mock_reg = MagicMock()
        mock_reg.tools = {
            'weather': {
                'manifest': {
                    'name': 'weather',
                    'description': 'Search the web',
                    'parameters': {
                        'query': {'type': 'string', 'required': True, 'description': 'Search query'},
                    },
                },
            },
        }
        MockRegistry.return_value = mock_reg

        schemas = get_external_tool_schemas(['weather'])
        assert len(schemas) == 1
        assert schemas[0]['name'] == 'weather'
        assert 'query' in schemas[0]['input_schema']['properties']

    @patch('services.tool_registry_service.ToolRegistryService')
    def test_skips_missing_tools(self, MockRegistry):
        mock_reg = MagicMock()
        mock_reg.tools = {}
        MockRegistry.return_value = mock_reg

        schemas = get_external_tool_schemas(['nonexistent'])
        assert schemas == []

    @patch('services.tool_registry_service.ToolRegistryService')
    def test_handles_multiple_tools(self, MockRegistry):
        mock_reg = MagicMock()
        mock_reg.tools = {
            'tool_a': {'manifest': {'name': 'tool_a', 'description': 'A', 'parameters': {}}},
            'tool_b': {'manifest': {'name': 'tool_b', 'description': 'B', 'parameters': {}}},
        }
        MockRegistry.return_value = mock_reg

        schemas = get_external_tool_schemas(['tool_a', 'tool_b'])
        assert len(schemas) == 2
        names = {s['name'] for s in schemas}
        assert names == {'tool_a', 'tool_b'}

    @patch('services.tool_registry_service.ToolRegistryService')
    def test_skips_tool_with_no_name_in_manifest(self, MockRegistry):
        mock_reg = MagicMock()
        mock_reg.tools = {
            'broken': {'manifest': {'description': 'no name field'}},
        }
        MockRegistry.return_value = mock_reg

        schemas = get_external_tool_schemas(['broken'])
        assert schemas == []


class TestInputSchemaFastPath:
    """Tests for the input_schema fast path in _manifest_to_schema."""

    def test_input_schema_passthrough(self):
        """Manifest with input_schema returns it unchanged without conversion."""
        input_schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        }
        manifest = {
            "name": "search",
            "description": "Search the web",
            "input_schema": input_schema,
        }
        schema = _manifest_to_schema(manifest)

        assert schema is not None
        assert schema["name"] == "search"
        assert schema["description"] == "Search the web"
        assert schema["input_schema"] is input_schema

    def test_input_schema_takes_precedence_over_parameters(self):
        """When both input_schema and parameters are present, input_schema wins."""
        input_schema = {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        }
        manifest = {
            "name": "dual_format",
            "description": "Has both formats",
            "input_schema": input_schema,
            "parameters": {
                "legacy_param": {"type": "string", "required": True},
            },
        }
        schema = _manifest_to_schema(manifest)

        # input_schema wins: legacy_param should NOT appear
        assert "q" in schema["input_schema"]["properties"]
        assert "legacy_param" not in schema["input_schema"]["properties"]
        # The required list is from input_schema, not derived from parameters
        assert schema["input_schema"]["required"] == ["q"]

    def test_legacy_parameters_still_works(self):
        """Interface tools using the legacy parameters format still convert correctly."""
        manifest = {
            "name": "interface_tool",
            "description": "An interface tool",
            "parameters": {
                "action": {"type": "string", "required": True, "description": "What to do"},
                "value": {"type": "string", "required": False, "default": "default_val"},
            },
        }
        schema = _manifest_to_schema(manifest)

        assert schema is not None
        assert schema["name"] == "interface_tool"
        assert schema["input_schema"]["type"] == "object"
        assert "action" in schema["input_schema"]["properties"]
        assert "value" in schema["input_schema"]["properties"]
        assert "action" in schema["input_schema"]["required"]
        assert "value" not in schema["input_schema"]["required"]
        assert schema["input_schema"]["properties"]["value"]["default"] == "default_val"

    def test_input_schema_with_nested_items(self):
        """input_schema with array items passthrough preserves nested structure."""
        input_schema = {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"action": {"type": "string"}},
                        "required": ["action"],
                    },
                },
            },
            "required": ["steps"],
        }
        manifest = {"name": "multi_step", "description": "Multi-step tool", "input_schema": input_schema}
        schema = _manifest_to_schema(manifest)

        assert schema["input_schema"]["properties"]["steps"]["items"]["required"] == ["action"]
