"""Normal Hermes plugin entry for the optional profile history library."""

__all__ = ["register"]


def register(ctx):
    from .src.hermes_ariadne.library_plugin import register as register_library

    register_library(ctx)
