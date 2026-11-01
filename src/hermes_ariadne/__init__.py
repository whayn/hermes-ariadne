"""Ariadne's archive primitives (no Hermes import or I/O at package import)."""

__version__ = "0.1.0"


def register(ctx):
    """Hermes directory-loader entry point; engine construction itself does no I/O."""
    from .engine import AriadneEngine

    ctx.register_context_engine(AriadneEngine())
