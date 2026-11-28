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
plugin-owned auxiliary task ``ariadne_summary``. The legacy ``rolling`` block is
still validated (an invalid value refuses registration) but unused by the engine;
it goes with the development engines (#70, phase 2). It is validated here, without
importing ``rolling``, so registration loads no development module.
"""

# ``rolling.RollingSettings`` field names; each value must be a positive bounded int.
_LEGACY_ROLLING_KEYS = frozenset(
    {
        "max_logical_jobs",
        "max_request_bytes",
        "max_total_request_bytes",
        "max_readable_delta_bytes",
        "max_delta_exchanges",
        "trigger_bytes",
        "max_summary_bytes",
        "max_navigation_bytes",
        "max_candidate_bytes",
        "reserve_tokens",
        "max_prefix_events",
        "max_checkpoint_bytes",
        "max_tokens",
        "timeout",
    }
)


def _validate_legacy_rolling(config):
    """``RollingSettings(**config)``'s refusals: TypeError for keys, ValueError for values."""
    unknown = set(config) - _LEGACY_ROLLING_KEYS
    if unknown:
        raise TypeError(f"unknown rolling settings: {sorted(map(str, unknown))}")
    for value in config.values():
        if type(value) is not int or not 0 < value <= 2**31 - 1:
            raise ValueError("rolling limits must be positive bounded integers")


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
    from .tree_builder import TreeSettings, register_summary_task
    from .tree_engine import TreeAriadneEngine

    _validate_legacy_rolling(ctx.get_config("rolling", {}))
    tree = TreeSettings.from_config(ctx.get_config("tree", None))
    register_task = getattr(ctx, "register_auxiliary_task", None)
    if callable(register_task):
        register_summary_task(ctx)
    engine = TreeAriadneEngine(llm=llm, tree_settings=tree)
    registration = ctx.register_context_engine(engine)
    register_idle = getattr(ctx, "register_idle_task", None)
    if callable(register_idle):

        def resume(_run):
            # Hermes invokes idle tasks in the registering profile's scope.
            from hermes_cli.config import load_config
            from hermes_constants import get_hermes_home

            if (load_config().get("context") or {}).get("engine") == engine.name:
                engine.resume_backlog(get_hermes_home())

        register_idle(
            "resume-tree", resume, interval_hours=1, min_idle_hours=0, first_run="immediate"
        )
    return registration
