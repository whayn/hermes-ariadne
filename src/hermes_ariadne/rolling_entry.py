"""Opt-in general-plugin entry; import and construction perform no I/O."""


def register(ctx):
    """Read plugin-relative limits and retain Hermes's own LLM facade.

    The installation directory deliberately differs from the engine name. Never
    expose an engine class here: the limited directory loader also scans module
    classes after a failed registration and must not construct a fallback engine.
    """
    from .rolling import RollingSettings
    from .rolling_engine import RollingAriadneEngine

    settings = RollingSettings(**ctx.get_config("rolling", {}))
    llm = getattr(ctx, "llm", None)
    if not callable(getattr(llm, "complete", None)):
        raise ValueError("rolling requires the general-plugin host LLM facade")
    return ctx.register_context_engine(RollingAriadneEngine(llm=llm, settings=settings))
