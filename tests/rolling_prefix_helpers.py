"""Test/oracle-only models of the validated prefix state (#19).

Production code never builds a prefix state outside ``validate_generation``
(which uses ``_check_prefix_extension`` then ``_record_identities``) and never
needs to list a prefix's identities. The differential oracles still want to
compose the shipped primitives generation by generation and to read a state's
identity sets, so that composition lives here, not in ``rolling``.
"""

from hermes_ariadne import rolling


def empty_prefix():
    """A root parent prefix over a fresh identity index (a throwaway memo's)."""
    return rolling._ValidationMemo(None, "s", None).empty_prefix()


def extend_prefix(prefix, rows, uids, payloads, settings):
    """State of ``prefix`` plus ``rows`` (already ``_canonical_rows``-valid).

    The check-then-record composition ``validate_generation`` performs, under a
    fresh opaque introducer instead of its generation id and its lineage
    bookkeeping. Same verdicts and messages as ``_canonical_rows`` of the whole
    prefix.
    """
    delta = rolling._check_prefix_extension(prefix, rows, uids, payloads, settings)
    introducer = object()
    rolling._record_identities(prefix.index, introducer, delta)
    return rolling._PrefixState(
        prefix.length + len(rows), prefix.lineage | {introducer}, prefix.index
    )


def identities(prefix, kind):
    """``prefix``'s identities of ``kind`` ("message", "reasoning", "occurrence")."""
    return frozenset(
        identity
        for identity, introducers in prefix.index[kind].items()
        if rolling._introduced_in(introducers, prefix.lineage)
    )
