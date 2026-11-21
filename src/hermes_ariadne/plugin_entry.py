"""Combined Ariadne plugin registration (memory provider + context engine).

Hermes runs ``register(ctx)`` through three loaders, each with a different
context (verified against the installed host; see docs/mvp-plugin.md):

* the general plugin manager's full ``PluginContext`` (has ``ctx.llm``):
  registers the context engine; its memory registration is recorded inertly;
* the memory-provider loader's collector (no ``llm``): takes the provider;
* the context-engine directory collector (no ``llm``): must register nothing so
  the host falls back to the general manager's engine.

The engine needs Hermes's LLM facade for summaries, so it is registered only
when that facade is present. Import and construction perform no I/O.

The ``ariadne`` engine is the profile memory-tree engine (``tree_engine``, issue
#54), configured by the plugin's ``tree`` settings; summary calls run under the
plugin-owned auxiliary task ``ariadne_summary``. The ``rolling`` block is still
validated (an invalid value refuses registration) and passed to the base class's
archive helpers; rolling summaries remain the development ``ariadne-rolling``
engine's.
"""


def _facade(ctx):
    try:
        llm = getattr(ctx, "llm", None)
    except Exception:  # collectors may raise for unknown attributes
        return None
    return llm if callable(getattr(llm, "complete", None)) else None


def _register_skill(ctx):
    from pathlib import Path

    register_skill = getattr(ctx, "register_skill", None)
    if not callable(register_skill):
        return
    path = Path(__file__).parent / "skills" / "import" / "SKILL.md"
    try:
        register_skill("import", path, "Import supplied files into Ariadne memory")
    except Exception:  # a skill problem must never cost the provider or engine
        pass


def register(ctx):
    from .memory_provider import AriadneMemoryProvider

    llm = _facade(ctx)
    ctx.register_memory_provider(AriadneMemoryProvider())
    _register_skill(ctx)
    if llm is None:
        return None
    from .rolling import RollingSettings
    from .tree_builder import TreeSettings, register_summary_task
    from .tree_engine import TreeAriadneEngine

    settings = RollingSettings(**ctx.get_config("rolling", {}))
    tree = TreeSettings.from_config(ctx.get_config("tree", None))
    register_task = getattr(ctx, "register_auxiliary_task", None)
    if callable(register_task):
        register_summary_task(ctx)
    return ctx.register_context_engine(
        TreeAriadneEngine(llm=llm, settings=settings, tree_settings=tree)
    )
