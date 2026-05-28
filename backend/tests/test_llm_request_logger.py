# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for llm_request_logger.py."""

import pytest

from services import llm_request_logger
from services.llm_request_logger import log_llm_request


@pytest.fixture
def logs_dir(tmp_path, monkeypatch):
    logs = tmp_path / 'logs'
    monkeypatch.setattr(llm_request_logger, '_LOGS_DIR', logs)
    return logs


@pytest.mark.unit
def test_writes_file_with_all_fields_verbatim(logs_dir):
    """One file per call, all required fields present, content verbatim
    (including large strings and unicode)."""
    big_user = 'x' * 100_000
    unicode_sys = 'Hello 🌍 — こんにちは — <>&"'
    tools = [{'name': 'search', 'description': 'Search the web'}]

    log_llm_request(
        caller='TestCaller',
        job='my-job',
        provider='openai',
        model='gpt-4o',
        system_message=unicode_sys,
        user_message=big_user,
        tools=tools,
    )

    files = list(logs_dir.glob('*.log'))
    assert len(files) == 1
    content = files[0].read_text(encoding='utf-8')

    # Header fields
    assert '# caller: TestCaller' in content
    assert '# job: my-job' in content
    assert '# provider: openai' in content
    assert '# model: gpt-4o' in content
    assert '# timestamp:' in content

    # Verbatim body
    assert big_user in content
    assert unicode_sys in content
    assert '"search"' in content  # tools serialized as JSON


@pytest.mark.unit
def test_does_not_raise_on_unwritable_dir(tmp_path, monkeypatch):
    """Logger must never block the LLM call — any write failure is swallowed."""
    fake_file = tmp_path / 'i_am_a_file'
    fake_file.write_text('block')
    monkeypatch.setattr(llm_request_logger, '_LOGS_DIR', fake_file / 'logs')

    # Must not raise
    log_llm_request(
        caller='FailCaller',
        job='unified',
        provider='anthropic',
        model='claude',
        system_message='s',
        user_message='u',
        tools=None,
    )
