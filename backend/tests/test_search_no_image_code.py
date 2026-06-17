# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License")

"""
Feature test for the search redesign's image-extraction removal: the result
transformers no longer emit an ``image`` key.

Drives the real ``transform()`` against a reddit fixture that carries both a
``preview.images`` block and a ``thumbnail`` — the two signals the old code
mined for an image. The redesign ignores them, so no result dict carries an
``image`` key.

Baseline-fail proof: on the pre-redesign code ``transform("reddit", …)`` returns
an ``image`` field, so this test fails before the extraction code is removed.
"""

import json

import pytest

from tools.search.transformers import transform

pytestmark = pytest.mark.unit


# ── Reddit fixture shaped from the real FIELD_MAPS entry ─────────────────────
#   results_path = 'data.children', unwrap_key = 'data'
#   title = 'title', snippet = 'selftext', url = 'url', date = 'created_utc'
# The preview.images / thumbnail blocks below were the old image sources; the
# redesign must ignore them.
_REDDIT_FIXTURE = {
    "data": {
        "children": [
            {
                "data": {
                    "title": "Best Python async libraries 2026",
                    "selftext": "I have been using asyncio and trio, what do others prefer?",
                    "url": "https://www.reddit.com/r/Python/comments/abc123/",
                    "created_utc": 1718000000.0,
                    "permalink": "/r/Python/comments/abc123/",
                    "preview": {
                        "images": [
                            {
                                "source": {
                                    "url": "https://external-preview.redd.it/img.jpg",
                                    "width": 640,
                                    "height": 480,
                                }
                            }
                        ]
                    },
                }
            },
            {
                "data": {
                    "title": "Async vs threads — the definitive answer",
                    "selftext": "Let me explain why GIL matters here.",
                    "url": "https://www.reddit.com/r/Python/comments/def456/",
                    "created_utc": 1718100000.0,
                    "permalink": "/r/Python/comments/def456/",
                    # no preview — old code fell through to thumbnail
                    "thumbnail": "https://a.thumbs.redditmedia.com/thumb.jpg",
                }
            },
        ]
    }
}


def test_transform_reddit_output_has_no_image_key():
    """After the redesign, transform() output dicts must not contain 'image'."""
    results = transform("reddit", "json", _REDDIT_FIXTURE, limit=10)

    assert results, "Expected at least one result from the reddit fixture"

    for r in results:
        assert "image" not in r, (
            f"Result still contains 'image' key: {json.dumps(r)}"
        )
