"""Hermes plugin root for Ariadne (installed by ``hermes plugins install``).

Registers a memory provider (``register_memory_provider``) and, when Hermes's
LLM facade is available, a context engine (``register_context_engine``). Both
are selected as ``ariadne``. Keep this module free of engine/provider classes:
Hermes's context-engine directory loader scans module classes when no engine
was registered, and must fall through to the general plugin's engine.
"""


def register(ctx):
    from .src.hermes_ariadne.plugin_entry import register as _register

    return _register(ctx)
