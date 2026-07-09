"""Migrated CRUD Endpoint subclasses — one file, one class. Presence in this package = migrated.

Auto-discovery instantiates each concrete subclass with no arguments and
registers its generated Namespace. The legacy module serving the same slug
must be deleted in the same commit — both registering means duplicate routes.
"""
