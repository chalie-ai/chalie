"""FileMapperService.get_policy_defaults_path resolves the static seed asset."""
from services.file_mapper_service import FileMapperService


def test_policy_defaults_path_points_at_abilities_assets():
    p = FileMapperService.get_policy_defaults_path()
    assert p.name == "policy_defaults.json"
    assert p.parent.name == "assets"
    assert p.parent.parent.name == "abilities"
