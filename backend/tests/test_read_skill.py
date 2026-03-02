"""
Tests for services/innate_skills/read_skill.py
"""

import pytest
from unittest.mock import MagicMock, patch


# ─── Source classification ────────────────────────────────────────────────────

@pytest.mark.unit
class TestSourceClassification:
    def _classify(self, source):
        from services.innate_skills.read_skill import _classify_source
        return _classify_source(source)

    def test_http_url(self):
        assert self._classify('http://example.com/page') == 'url'

    def test_https_url(self):
        assert self._classify('https://example.com/article') == 'url'

    def test_ftp_url(self):
        assert self._classify('ftp://files.example.com/data') == 'url'

    def test_absolute_path(self):
        assert self._classify('/home/user/document.pdf') == 'file'

    def test_home_tilde_path(self):
        assert self._classify('~/Documents/notes.txt') == 'file'

    def test_relative_path(self):
        assert self._classify('./data/report.docx') == 'file'

    def test_bare_string_no_scheme(self):
        # No scheme → treated as file path
        assert self._classify('example.com/page') == 'file'

    def test_windows_style_path(self):
        # No scheme → file
        assert self._classify('C:/Users/user/doc.pdf') == 'file'


# ─── handle_read entry point ──────────────────────────────────────────────────

@pytest.mark.unit
class TestHandleRead:
    def test_empty_source_returns_error(self):
        from services.innate_skills.read_skill import handle_read
        result = handle_read('topic', {})
        assert '[READ] Error' in result
        assert 'source' in result.lower()

    def test_url_alias_accepted(self):
        with patch('services.innate_skills.read_skill._read_url', return_value='[READ] ok'):
            from services.innate_skills.read_skill import handle_read
            result = handle_read('topic', {'url': 'https://example.com'})
        assert 'ok' in result

    def test_max_chars_clamped_to_ceiling(self):
        with patch('services.innate_skills.read_skill._read_url') as mock_read:
            mock_read.return_value = '[READ] ok'
            from services.innate_skills.read_skill import handle_read
            handle_read('topic', {'source': 'https://ex.com', 'max_chars': 999999})
        _, called_max = mock_read.call_args[0]
        assert called_max == 8000

    def test_max_chars_clamped_to_floor(self):
        with patch('services.innate_skills.read_skill._read_url') as mock_read:
            mock_read.return_value = '[READ] ok'
            from services.innate_skills.read_skill import handle_read
            handle_read('topic', {'source': 'https://ex.com', 'max_chars': 0})
        _, called_max = mock_read.call_args[0]
        assert called_max == 100

    def test_invalid_max_chars_uses_default(self):
        with patch('services.innate_skills.read_skill._read_url', return_value='[READ] ok'):
            from services.innate_skills.read_skill import handle_read
            # Should not raise
            result = handle_read('topic', {'source': 'https://ex.com', 'max_chars': 'banana'})
        assert '[READ]' in result


# ─── SSRF guard ───────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestUrlSSRF:
    def _check(self, url):
        from services.innate_skills.read_skill import _is_private_url
        return _is_private_url(url)

    def test_localhost_blocked(self):
        with patch('socket.getaddrinfo') as mock_dns:
            mock_dns.return_value = [(None, None, None, None, ('127.0.0.1', 0))]
            assert self._check('http://localhost') is True

    def test_loopback_ip_blocked(self):
        with patch('socket.getaddrinfo') as mock_dns:
            mock_dns.return_value = [(None, None, None, None, ('127.0.0.1', 0))]
            assert self._check('http://127.0.0.1') is True

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
        """Build a mock requests module for sys.modules patching."""
        import requests as real_requests
        mock_req = MagicMock()
        mock_req.RequestException = real_requests.RequestException
        if side_effect:
            mock_req.get.side_effect = side_effect
        else:
            mock_response = MagicMock()
            mock_response.text = response_text or ''
            mock_response.raise_for_status = MagicMock()
            mock_req.get.return_value = mock_response
        return mock_req

    def test_successful_fetch_returns_content(self):
        mock_req = self._make_mock_requests('<html><body><article><p>Article.</p></article></body></html>')
        with patch.dict('sys.modules', {'requests': mock_req}), \
             patch('services.innate_skills.read_skill._is_private_url', return_value=False), \
             patch('services.text_extractor.extract_html', return_value='Article content.'):
            from services.innate_skills.read_skill import _read_url
            result = _read_url('https://example.com', 4000)
        assert '[READ]' in result
        assert 'Article content' in result

    def test_private_url_blocked_before_fetch(self):
        from services.innate_skills.read_skill import _read_url
        with patch('services.innate_skills.read_skill._is_private_url', return_value=True):
            result = _read_url('http://localhost', 4000)
        assert 'access denied' in result.lower()

    def test_fetch_network_error_returns_error_string(self):
        import requests as real_requests
        mock_req = self._make_mock_requests(side_effect=real_requests.RequestException('Connection refused'))
        with patch.dict('sys.modules', {'requests': mock_req}), \
             patch('services.innate_skills.read_skill._is_private_url', return_value=False):
            from services.innate_skills.read_skill import _read_url
            result = _read_url('https://unreachable.example.com', 4000)
        assert '[READ] Fetch failed' in result

    def test_empty_content_returns_informative_message(self):
        mock_req = self._make_mock_requests('<html></html>')
        with patch.dict('sys.modules', {'requests': mock_req}), \
             patch('services.innate_skills.read_skill._is_private_url', return_value=False), \
             patch('services.text_extractor.extract_html', return_value=''):
            from services.innate_skills.read_skill import _read_url
            result = _read_url('https://example.com', 4000)
        assert 'No readable content' in result


# ─── File reading ─────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestFileReading:
    def test_file_not_found_returns_error(self):
        from services.innate_skills.read_skill import _read_file
        result = _read_file('/nonexistent/path/to/file.txt', 4000)
        assert 'not found' in result.lower()

    def test_directory_rejected(self):
        from services.innate_skills.read_skill import _read_file
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isfile', return_value=False):
            result = _read_file('/tmp/some_directory', 4000)
        assert 'not a file' in result.lower()

    def test_no_read_permission_returns_error(self):
        from services.innate_skills.read_skill import _read_file
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isfile', return_value=True), \
             patch('os.access', return_value=False):
            result = _read_file('/tmp/locked.txt', 4000)
        assert 'permission' in result.lower()

    def test_successful_extraction_returns_content(self):
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isfile', return_value=True), \
             patch('os.access', return_value=True), \
             patch('os.path.realpath', return_value='/tmp/doc.pdf'), \
             patch('services.text_extractor.detect_mime_type', return_value='application/pdf'), \
             patch('services.text_extractor.extract_text', return_value='PDF content here.'), \
             patch('services.text_extractor.normalize_text', return_value='PDF content here.'):
            from services.innate_skills.read_skill import _read_file
            result = _read_file('/tmp/doc.pdf', 4000)
        assert '[READ]' in result
        assert 'PDF content' in result

    def test_empty_extraction_returns_informative_message(self):
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isfile', return_value=True), \
             patch('os.access', return_value=True), \
             patch('os.path.realpath', return_value='/tmp/empty.txt'), \
             patch('services.text_extractor.detect_mime_type', return_value='text/plain'), \
             patch('services.text_extractor.extract_text', return_value=''), \
             patch('services.text_extractor.normalize_text', return_value=''):
            from services.innate_skills.read_skill import _read_file
            result = _read_file('/tmp/empty.txt', 4000)
        assert 'No text content' in result

    def test_content_truncated_at_max_chars(self):
        long_content = 'x' * 10000
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isfile', return_value=True), \
             patch('os.access', return_value=True), \
             patch('os.path.realpath', return_value='/tmp/big.txt'), \
             patch('services.text_extractor.detect_mime_type', return_value='text/plain'), \
             patch('services.text_extractor.extract_text', return_value=long_content), \
             patch('services.text_extractor.normalize_text', return_value=long_content):
            from services.innate_skills.read_skill import _read_file
            result = _read_file('/tmp/big.txt', 500)
        assert 'truncated' in result


# ─── File read security ───────────────────────────────────────────────────────

@pytest.mark.unit
class TestFileReadSecurity:
    def _read(self, path):
        from services.innate_skills.read_skill import _read_file
        return _read_file(path, 4000)

    def test_etc_passwd_blocked(self):
        result = self._read('/etc/passwd')
        assert 'access denied' in result.lower()

    def test_proc_blocked(self):
        result = self._read('/proc/self/environ')
        assert 'access denied' in result.lower()

    def test_dev_blocked(self):
        result = self._read('/dev/urandom')
        assert 'access denied' in result.lower()

    def test_sys_blocked(self):
        result = self._read('/sys/kernel/hostname')
        assert 'access denied' in result.lower()

    def test_normal_path_not_blocked(self):
        # A non-system path with a nonexistent file → "not found", NOT "access denied"
        result = self._read('/tmp/nonexistent_file_xyz.txt')
        assert 'access denied' not in result.lower()
        assert 'not found' in result.lower()


# ─── Link extraction ──────────────────────────────────────────────────────────

@pytest.mark.unit
class TestLinkExtraction:
    def test_extracts_basic_links(self):
        html = '<a href="/page2">Page Two</a><a href="https://example.com/page3">Three</a>'
        from services.innate_skills.read_skill import _extract_links
        links = _extract_links(html, 'https://example.com')
        assert len(links) >= 1

    def test_skips_social_media_domains(self):
        html = '<a href="https://facebook.com/post">FB</a><a href="/real">Real</a>'
        from services.innate_skills.read_skill import _extract_links
        links = _extract_links(html, 'https://example.com')
        assert all('facebook' not in l['url'] for l in links)

    def test_skips_navigation_paths(self):
        html = '<a href="/login">Login</a><a href="/signup">Sign up</a><a href="/article">Article</a>'
        from services.innate_skills.read_skill import _extract_links
        links = _extract_links(html, 'https://example.com')
        assert all('/login' not in l['url'] for l in links)
        assert all('/signup' not in l['url'] for l in links)

    def test_skips_fragment_only_anchors(self):
        html = '<a href="#section">Jump</a><a href="/page">Page</a>'
        from services.innate_skills.read_skill import _extract_links
        links = _extract_links(html, 'https://example.com')
        assert all('#' not in l['url'].split('/')[-1] for l in links)

    def test_deduplicates_links(self):
        html = '<a href="/page">One</a><a href="/page">Two</a>'
        from services.innate_skills.read_skill import _extract_links
        links = _extract_links(html, 'https://example.com')
        urls = [l['url'] for l in links]
        assert len(urls) == len(set(urls))

    def test_max_15_links(self):
        html = ''.join(f'<a href="/page{i}">Page {i}</a>' for i in range(30))
        from services.innate_skills.read_skill import _extract_links
        links = _extract_links(html, 'https://example.com')
        assert len(links) <= 15
