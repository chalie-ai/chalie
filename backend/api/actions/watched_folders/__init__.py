"""Watched-folders action package — nested verb operations for watched folders.

Directory name is underscored (Python packages can't have dashes); the URL
segment stays ``watched-folders`` via each Action subclass's ``slug()``,
matching the CRUD endpoint in ``api/endpoints/watched_folders.py``.
"""
