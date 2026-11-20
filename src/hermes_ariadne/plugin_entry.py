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
    from .mvp_engine import AriadneEngine
    from .rolling import RollingSettings

    settings = RollingSettings(**ctx.get_config("rolling", {}))
    return ctx.register_context_engine(AriadneEngine(llm=llm, settings=settings))
