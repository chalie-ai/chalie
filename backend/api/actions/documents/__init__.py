"""Documents action package — the non-CRUD verb operations for documents.

The plain CRUD routes (list / get-by-id / soft-delete) live on the
``Documents`` endpoint in ``api/endpoints/documents.py``; everything else —
file upload/download/preview, full-text content, classification, lifecycle
transitions (restore/purge/confirm/augment/supersede), semantic search, and
classification groupings — is a verb Action here, one file per verb.

All classes return the URL segment ``documents`` from ``slug()`` so their
routes mount under ``/api/documents/<verb>[/<id>]``, alongside the endpoint.
"""
