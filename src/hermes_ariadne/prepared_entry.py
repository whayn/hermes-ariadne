"""Directory entrypoint for the explicitly installed prepared engine."""


def register(ctx):
    from .live_engine import PreparedAriadneEngine

    ctx.register_context_engine(PreparedAriadneEngine())
