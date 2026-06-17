import pytest

from services.filename_utils import safe_filename


@pytest.mark.unit
class TestSafeFilename:
    def test_strips_path_separators_and_traversal(self):
        assert '/' not in safe_filename('../../etc/passwd')
        assert '\\' not in safe_filename('..\\windows\\system32')
        assert safe_filename('../../etc/passwd') == 'etc_passwd'

    def test_strips_null_and_control_chars(self):
        result = safe_filename('file\x00\x01\x1f.txt')
        assert not any(ord(c) < 0x20 or ord(c) == 0x7f for c in result)
        assert result.endswith('.txt')

    def test_strips_leading_dots(self):
        assert not safe_filename('...hidden').startswith('.')

    def test_unsafe_only_name_returns_empty(self):
        # Caller is responsible for substituting a fallback.
        assert safe_filename('..') == ''
        assert safe_filename('') == ''

    def test_caps_length_at_255_preserving_extension(self):
        result = safe_filename('a' * 300 + '.txt')
        assert len(result) == 255
        assert result.endswith('.txt')

    def test_preserves_normal_name_and_extension(self):
        assert safe_filename('report.tar.gz') == 'report.tar.gz'

    def test_caps_when_extension_itself_exceeds_limit(self):
        # Pathological "extension" longer than the cap must not re-overflow it.
        result = safe_filename('a.' + 'x' * 300)
        assert len(result) <= 255
