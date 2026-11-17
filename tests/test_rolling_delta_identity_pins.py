"""First format-2 durable identities, captured from real schema-3 publication.

Legacy literals remain in test_rolling_identity_pins.py. Change these only with
an explicit storage migration, never to make an unrelated refactor pass.
"""

from test_rolling_state import ready, setup, target

from hermes_ariadne import rolling, rolling_state

PINS = [
    {
        "target_id": "89d41c534e61debde9c80cadf70c1f070a56ca975d0b883e7c1e8e0f77b38b4e",
        "job_id": "83e1ae5d2a771a009a117659395da7c7da3fac4e98dba086316cc6edbbcbd3d6",
        "generation_id": "d1febd02f85eb79e64376a876cde54598fca27e11de6e570d9befa882d4e4913",
        "prefix_sha256": "7a62937d71c0fce84f942ed24875c0825ec406f5bfdfdbad1f5d086c0de49161",
        "manifest_sha256": "363cb2facc2337340c1aa5989613d8532974fc2c639a0be18bc4848aedf8254a",
    },
    {
        "target_id": "cb5663b569113947599f6cc2c49634909b3e28a75c766d90720e5bef58af0432",
        "job_id": "d6c4f513eb79c895c180bc7f939a20d23f503417c0f0235b45e50c748aefdd06",
        "generation_id": "70e9ec8d439c1f77ea7d0285665cf1b7268c8329b5df2bef8813a4bc5cb3198a",
        "prefix_sha256": "138d491afb5f2b8d7350c61d355c7b983272e76c71b0dfa8ce5b6baabf4984fe",
        "manifest_sha256": "4faf284c6bcd4c25034b88f6d035e5d5c5f53e88ff7ad4e450863d11bc18e396",
    },
    {
        "target_id": "12e7d2edf8ec359521c03698bb77581de5035726b28b41b494a857882bcfeb30",
        "job_id": "e7f2e110cd5fc9060f4cc815420343cfb4b1a9ed84376f91d5ea7fefd1ee80a9",
        "generation_id": "e34761a8676afc4778b3a45dd44c2b3b6408309e7e6a8379d1e98147b759f465",
        "prefix_sha256": "f2870ef66189d5651964e679a23efe739f15f6aebd62f7ec7ac4204a2eec3039",
        "manifest_sha256": "308b2e6bb4e9462ad2bd8e5197c73e18a0a793787e829146e3b45ef5c1d5fcdd",
    },
]


def test_literal_delta_target_job_generation_and_manifest_identities(tmp_path):
    store, settings = setup(tmp_path)
    parent, actual = None, []
    with rolling_state.RollingJournal(
        tmp_path, "s", settings, store=store, storage_version=3
    ) as journal:
        for sequence in range(1, 4):
            frozen = journal.encode_target(target(store, settings, sequence, parent), parent)
            text = "summary " + str(sequence)
            ready(journal, frozen, text)
            parent = journal.publish_ready(
                frozen.job_id,
                rolling.generation_bytes(frozen, text, sequence=sequence, settings=settings),
            )
            actual.append(
                {
                    "target_id": frozen.target_id,
                    "job_id": frozen.job_id,
                    "generation_id": parent.generation_id,
                    "prefix_sha256": frozen.prefix_sha256,
                    "manifest_sha256": rolling.sha256(journal.manifest_path.read_bytes()),
                }
            )
    assert actual == PINS
    with rolling_state.RollingJournal(tmp_path, "s", settings, store=store) as journal:
        assert journal.load_view() == parent
