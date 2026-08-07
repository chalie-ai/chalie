# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import os
from pathlib import Path

import pytest

from services.tmp_storage import (
    TMP_DIR,
    TMP_PATH_PREFIX,
    TMP_PREFIX,
    TmpStorage,
)

pytestmark = pytest.mark.unit


def test_prefix_is_under_os_tempdir_and_not_hardcoded_slash_tmp() -> None:
    # The prefix must live under the OS temp dir (resolved), composed with the
    # chalie_ marker — not a literal '/tmp/chalie_'.
    assert TMP_PATH_PREFIX == os.path.join(TMP_DIR, TMP_PREFIX)
    assert TMP_PATH_PREFIX.endswith(os.path.join("", "chalie_"))
    assert TMP_PREFIX == "chalie_"


def test_new_tmp_path_lives_under_prefix() -> None:
    p = TmpStorage.new_tmp_path("deadbeef.png")
    assert p == TMP_PATH_PREFIX + "deadbeef.png"
    assert p.startswith(TMP_PATH_PREFIX)


def test_is_chalie_tmp_file_accepts_real_file_under_prefix() -> None:
    path = TmpStorage.new_tmp_path("unit_probe.bin")
    with open(path, "wb") as fh:
        fh.write(b"x")
    try:
        assert TmpStorage.is_chalie_tmp_file(path) is True
    finally:
        os.unlink(path)


def test_is_chalie_tmp_file_rejects_path_outside_prefix(tmp_path: "Path") -> None:
    outside = tmp_path / "elsewhere.bin"
    outside.write_bytes(b"x")
    assert TmpStorage.is_chalie_tmp_file(str(outside)) is False


def test_is_chalie_tmp_file_rejects_missing_file() -> None:
    assert TmpStorage.is_chalie_tmp_file(TmpStorage.new_tmp_path("does_not_exist.bin")) is False


def test_is_chalie_tmp_file_rejects_directory_under_prefix() -> None:
    # A directory under the prefix is not a *file*; the guard requires a file.
    dir_path = TmpStorage.new_tmp_path("unit_probe_dir")
    os.mkdir(dir_path)
    try:
        assert TmpStorage.is_chalie_tmp_file(dir_path) is False
    finally:
        os.rmdir(dir_path)
