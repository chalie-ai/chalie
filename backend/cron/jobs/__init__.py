"""Concrete cron jobs.

One module per job. Each defines a single ``ScheduledJob``/``IdleGatedJob``
subclass implementing ``_run``; instances are registered in ``cron.JOBS``.
"""
