"""
Tests for abilities/read.py (ReadAbility)
"""

import pytest
from unittest.mock import MagicMock, patch


# ─── Source classification ────────────────────────────────────────────────────

@pytest.mark.unit
class TestSourceClassification:
    def _classify(self, source):
        from abilities.read import _classify_source
        return _classify_source(source)

    def test_http_url(self):
        assert self._classify('http://example.com/page') == 'url'

    def test_https_url(self):
        assert self._classify('https://example.com/article') == 'url'

    def test_absolute_path(self):
        assert self._classify('/home/user/document.pdf') == 'file'


# ─── handle_read entry point ──────────────────────────────────────────────────

@pytest.mark.unit
class TestHandleRead:
    def test_empty_source_returns_error(self):
        from abilities.read import ReadAbility
        raw = ReadAbility().execute('topic', {}, None)
        assert raw is not None, "ReadAbility.execute() returned None"
        result = raw['text']
        assert '[read(' in result
        assert 'error=source-required' in result
        assert '[end:read]' in result

    def test_max_chars_uncapped(self):
        with patch('abilities.read._read_url') as mock_read:
            mock_read.return_value = '[read(source=x, max_chars=4000)] ok'
            from abilities.read import ReadAbility
            ReadAbility().execute('topic', {'source': 'https://ex.com', 'max_chars': 100_000_000}, None)
        _, called_max = mock_read.call_args[0]
        assert called_max == 100_000_000

    def test_max_chars_clamped_to_floor(self):
        with patch('abilities.read._read_url') as mock_read:
            mock_read.return_value = '[read(source=x, max_chars=4000)] ok'
            from abilities.read import ReadAbility
            ReadAbility().execute('topic', {'source': 'https://ex.com', 'max_chars': 0}, None)
        _, called_max = mock_read.call_args[0]
        assert called_max == 100

    def test_invalid_max_chars_uses_default(self):
        with patch('abilities.read._read_url', return_value='[read(source=x, max_chars=4000)] ok\n[end:read]'):
            from abilities.read import ReadAbility
            # Should not raise
            result = ReadAbility().execute('topic', {'source': 'https://ex.com', 'max_chars': 'banana'}, None)['text']
        assert '[read(' in result
        assert '[end:read]' in result


# ─── SSRF guard ───────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestUrlSSRF:
    def _check(self, url):
        from abilities.read import _is_private_url
        return _is_private_url(url)

    def test_localhost_blocked(self):
        with patch('socket.getaddrinfo') as mock_dns:
            mock_dns.return_value = [(None, None, None, None, ('127.0.0.1', 0))]
            assert self._check('http://localhost') is True

    def test_private_10_range_blocked(self):
        with patch('socket.getaddrinfo') as mock_dns:
            mock_dns.return_value = [(None, None, None, None, ('10.0.0.1', 0))]
            assert self._check('http://internal.corp') is True

    def test_private_192_168_blocked(self):
        with patch('socket.getaddrinfo') as mock_dns:
            mock_dns.return_value = [(None, None, None, None, ('192.168.1.1', 0))]
            assert self._check('http://router.local') is True

    def test_link_local_169_254_blocked(self):
        with patch('socket.getaddrinfo') as mock_dns:
            mock_dns.return_value = [(None, None, None, None, ('169.254.169.254', 0))]
            assert self._check('http://169.254.169.254') is True

    def test_unresolvable_hostname_blocked(self):
        import socket
        with patch('socket.getaddrinfo', side_effect=socket.gaierror):
            assert self._check('http://does-not-exist.invalid') is True

    def test_no_hostname_blocked(self):
        assert self._check('file:///etc/passwd') is True

    def test_public_ip_allowed(self):
        with patch('socket.getaddrinfo') as mock_dns:
            mock_dns.return_value = [(None, None, None, None, ('93.184.216.34', 0))]
            assert self._check('https://example.com') is False


# ─── URL reading ──────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestUrlReading:
    def _make_mock_requests(self, response_text=None, side_effect=None):
        """Build a mock requests module for sys.modules patching.

        Wires Session() context-manager .get() to the same response/side_effect,
        matching the real implementation which uses `with requests.Session() as s`.
        """
        import requests as real_requests
        mock_req = MagicMock()
        mock_req.RequestException = real_requests.RequestException

        mock_session = MagicMock()
        mock_session.headers = {}
        if side_effect:
            mock_session.get.side_effect = side_effect
        else:
            mock_response = MagicMock()
            mock_response.text = response_text or ''
            mock_response.raise_for_status = MagicMock()
            mock_session.get.return_value = mock_response

        # `with requests.Session() as session:` — __enter__ returns the session,
        # __exit__ is a no-op.
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = False
        mock_req.Session.return_value = mock_session
        return mock_req

    def test_successful_fetch_returns_content(self):
        mock_req = self._make_mock_requests('<html><body><article><p>Article.</p></article></body></html>')
        with patch.dict('sys.modules', {'requests': mock_req}), \
             patch('abilities.read._is_private_url', return_value=False), \
             patch('services.text_extractor.extract_html', return_value='Article content.'):
            from abilities.read import _read_url
            result = _read_url('https://example.com', 4000)
        assert '[read(source=https://example.com' in result
        assert 'Article content' in result
        assert '[end:read]' in result

    def test_private_url_blocked_before_fetch(self):
        from abilities.read import _read_url
        with patch('abilities.read._is_private_url', return_value=True):
            result = _read_url('http://localhost', 4000)
        assert 'blocked' in result.lower()
        assert '[end:read]' in result

    def test_fetch_network_error_returns_error_string(self):
        import requests as real_requests
        mock_req = self._make_mock_requests(side_effect=real_requests.RequestException('Connection refused'))
        with patch.dict('sys.modules', {'requests': mock_req}), \
             patch('abilities.read._is_private_url', return_value=False):
            from abilities.read import _read_url
            result = _read_url('https://unreachable.example.com', 4000)
        assert 'error=fetch-failed' in result
        assert '[end:read]' in result

    def test_empty_content_returns_informative_message(self):
        mock_req = self._make_mock_requests('<html></html>')
        with patch.dict('sys.modules', {'requests': mock_req}), \
             patch('abilities.read._is_private_url', return_value=False), \
             patch('services.text_extractor.extract_html', return_value=''):
            from abilities.read import _read_url
            result = _read_url('https://example.com', 4000)
        assert 'no-readable-content' in result
        assert '[end:read]' in result


# ─── File reading ─────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestFileReading:
    def test_file_not_found_returns_error(self):
        from abilities.read import _read_file
        result = _read_file('/nonexistent/path/to/file.txt', 4000)
        assert 'file-not-found' in result
        assert '[end:read]' in result

    def test_directory_rejected(self):
        from abilities.read import _read_file
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isfile', return_value=False):
            result = _read_file('/tmp/some_directory', 4000)
        assert 'not-a-file' in result
        assert '[end:read]' in result

    def test_no_read_permission_returns_error(self):
        from abilities.read import _read_file
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isfile', return_value=True), \
             patch('os.access', return_value=False):
            result = _read_file('/tmp/locked.txt', 4000)
        assert 'no-read-permission' in result
        assert '[end:read]' in result

    def test_successful_extraction_returns_content(self):
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isfile', return_value=True), \
             patch('os.access', return_value=True), \
             patch('os.path.realpath', return_value='/tmp/doc.pdf'), \
             patch('services.text_extractor.detect_mime_type', return_value='application/pdf'), \
             patch('services.text_extractor.extract_text', return_value='PDF content here.'), \
             patch('services.text_extractor.normalize_text', return_value='PDF content here.'):
            from abilities.read import _read_file
            result = _read_file('/tmp/doc.pdf', 4000)
        assert '[read(source=/tmp/doc.pdf' in result
        assert 'mime=application/pdf' in result
        assert 'PDF content' in result
        assert '[end:read]' in result

    def test_empty_extraction_returns_informative_message(self):
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isfile', return_value=True), \
             patch('os.access', return_value=True), \
             patch('os.path.realpath', return_value='/tmp/empty.txt'), \
             patch('services.text_extractor.detect_mime_type', return_value='text/plain'), \
             patch('services.text_extractor.extract_text', return_value=''), \
             patch('services.text_extractor.normalize_text', return_value=''):
            from abilities.read import _read_file
            result = _read_file('/tmp/empty.txt', 4000)
        assert 'no-text-content' in result
        assert '[end:read]' in result

    def test_content_truncated_at_max_chars(self):
        long_content = 'x' * 10000
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isfile', return_value=True), \
             patch('os.access', return_value=True), \
             patch('os.path.realpath', return_value='/tmp/big.txt'), \
             patch('services.text_extractor.detect_mime_type', return_value='text/plain'), \
             patch('services.text_extractor.extract_text', return_value=long_content), \
             patch('services.text_extractor.normalize_text', return_value=long_content):
            from abilities.read import _read_file
            result = _read_file('/tmp/big.txt', 500)
        assert 'truncated' in result
        assert '[end:read]' in result


# ─── File read security ───────────────────────────────────────────────────────

@pytest.mark.unit
class TestFileReadSecurity:
    def _read(self, path):
        from abilities.read import _read_file
        return _read_file(path, 4000)

    def test_etc_passwd_blocked(self):
        result = self._read('/etc/passwd')
        assert 'blocked' in result.lower()
        assert '[end:read]' in result

    def test_normal_path_not_blocked(self):
        # A non-system path with a nonexistent file → "not found", NOT "access denied"
        result = self._read('/tmp/nonexistent_file_xyz.txt')
        assert 'access denied' not in result.lower()
        assert 'file-not-found' in result
        assert '[end:read]' in result


