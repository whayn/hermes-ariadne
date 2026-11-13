"""Pinned durable rolling identities (#22).

Every digest below was produced by the code at 23664bc, before journal-open
work stopped re-deriving them. Existing journals store these exact values
(target/job/generation ids, payload hashes, artifact names), so a change that
alters any of them breaks journals already on disk. Update a literal only
together with a migration plan, never to make a refactor pass.
"""

import sqlite3

import pytest
from test_rolling_selection import POLICY, exchange
from test_rolling_state import setup

from hermes_ariadne import rolling, rolling_state
from hermes_ariadne.rolling import sha256

REQUEST = b'[{"content":"summarize","role":"user"}]'
ROUTE = ("codex_backend", "gpt-test")


def plain(sequence):
    return [
        {"message_uid": f"{sequence}-user", "role": "user", "content": f"user {sequence}"},
        {"message_uid": f"{sequence}-assistant", "role": "assistant", "content": f"ok {sequence}"},
    ]


def native(sequence):
    return exchange(f"g{sequence}a") + exchange(f"g{sequence}b")


def build(tmp_path, rows_for, policy, generations=3):
    """Publish ``generations`` generations, then leave one ready successor pending."""
    store, settings = setup(tmp_path, max_delta_exchanges=8)
    source, parent, records = [], None, []
    with rolling_state.RollingJournal(
        tmp_path, "s", settings, store=store, policy=policy
    ) as journal:
        for sequence in range(1, generations + 2):
            delta = rows_for(sequence)
            store.ingest("s", delta, origin="canonical")
            source += delta
            frozen = rolling.Target.create(
                [dict(row) for row in source],
                session_id="s",
                settings=settings,
                policy=policy,
                parent=parent,
                request_bytes=REQUEST,
                route_labels=ROUTE,
            )
            journal.freeze_target(frozen)
            assert journal.reserve(frozen.job_id, REQUEST, lambda: None, lambda: None)
            summary = f"summary {sequence}"
            journal.record_ready(frozen.job_id, summary, {"provider": "t", "model": "t"}, None)
            record = {
                "target_id": frozen.target_id,
                "job_id": frozen.job_id,
                "prefix_sha256": frozen.prefix_sha256,
                "readable_input_sha256": frozen.readable_input_digest,
            }
            if sequence <= generations:
                parent = journal.publish_ready(
                    frozen.job_id,
                    rolling.generation_bytes(frozen, summary, sequence=sequence, settings=settings),
                )
                record["generation_id"] = parent.generation_id
            records.append(record)
        directory = journal.directory
    return store, settings, records, directory


def durable(directory):
    """Content digests of every durable artifact a reopened journal must accept."""
    conn = sqlite3.connect(directory / "journal.sqlite3")
    try:
        targets = []
        for target_id, payload, payload_sha256 in conn.execute(
            "SELECT target_id,payload,payload_sha256 FROM targets"
        ):
            # A target's id is the hash of its exact stored canonical payload.
            assert sha256(payload) == payload_sha256 == target_id
            targets.append(target_id)
        targets.sort()
        jobs = sorted(
            tuple(row)
            for row in conn.execute(
                "SELECT job_id,target_id,state,text_sha256,generation_id FROM jobs"
            )
        )
    finally:
        conn.close()
    artifacts = []
    for path in (directory / "generations").iterdir():
        # A generation's artifact is named by the hash of its exact bytes.
        assert path.name == sha256(path.read_bytes()) + ".json"
        artifacts.append(path.name)
    artifacts.sort()
    return {
        "targets": targets,
        "jobs": jobs,
        "artifacts": artifacts,
        "manifest": sha256((directory / "manifest.json").read_bytes()),
    }


PLAIN = {
    "settings": "5ef99cbebc44d2527f5b4e226d55cbced13fc84eabf6c1e816ac4549d37771ea",
    "records": [
        {
            "target_id": "6152de274e564e98ee208cae7bc82ec0b7d53e9ae264f12714021fb29b8dff7d",
            "job_id": "0b2058fd630e353e9a8dc04e4431ad89815384212ac3336ece6288347987445c",
            "prefix_sha256": "a828829ec7d63ddf6a3add9bb885cb0f38965424d02efc91de7cf5e970bc8a06",
            "readable_input_sha256": "d715b0ca3b9c1717dd1e34082cdd3b7c26cae4959d8f0f2f9095d75d3d9c4619",
            "generation_id": "2653a2aa5e6d1829860e9d68d3f480c4e887152aa575b6ab1f57b1cb3c96880f",
        },
        {
            "target_id": "96530558cf2f5986e4e367cab64d3e463493dc291dd954c21c52743dfbabe713",
            "job_id": "829d56295c570fb71bbc45f1e924d8b62e9df6917c7b1c8f432b73937c3c8410",
            "prefix_sha256": "1d6706cf1011216ee4e22c334e14be62efbfbef11fba078c616c792d00ebd150",
            "readable_input_sha256": "696aa34e8590b5107996bd179cb7e130403029b3590b456276bd07d52e0025c0",
            "generation_id": "a47fdeceee8446bac0f099130939296c2ba881b5e5b426abf065cd04f9a5df00",
        },
        {
            "target_id": "91471ef2d2b37fcb9fe8438875b2147e980015a0303a36845d622f9bf711b4c7",
            "job_id": "c8bf2ddbd64e53b4574826fa6562758b3618a31cf88f19df737d0c17b9453ff0",
            "prefix_sha256": "41132331a2cbc89bd54503b2cf8abe1f74e78628b1ea641861ae20c7a1c68503",
            "readable_input_sha256": "b899c7c77aae00fb86f39131a6af1fa75e93765d593126439c421043a69041f5",
            "generation_id": "31b9febf74fddddf0b0b224a0e45a6fbe6b06755098f9d2ec15627aea3e57ada",
        },
        {
            "target_id": "6d4f7802ce2ee2c02bd1790e1105e7627b930b6f62a47c322061295142f3557d",
            "job_id": "d7dbc2cb87803ff5e2afc985123f2f9e421f75b23cd125db521dabc37c611f26",
            "prefix_sha256": "6640f01861c4ef16e853e120545f70c86c8780df730ae3b715bb34cd7f509c15",
            "readable_input_sha256": "0484e2590c414266cd66b225a5969ce46c531742d6f96c5815f369f96df1d023",
        },
    ],
    "durable": {
        "targets": [
            "6152de274e564e98ee208cae7bc82ec0b7d53e9ae264f12714021fb29b8dff7d",
            "6d4f7802ce2ee2c02bd1790e1105e7627b930b6f62a47c322061295142f3557d",
            "91471ef2d2b37fcb9fe8438875b2147e980015a0303a36845d622f9bf711b4c7",
            "96530558cf2f5986e4e367cab64d3e463493dc291dd954c21c52743dfbabe713",
        ],
        "jobs": [
            (
                "0b2058fd630e353e9a8dc04e4431ad89815384212ac3336ece6288347987445c",
                "6152de274e564e98ee208cae7bc82ec0b7d53e9ae264f12714021fb29b8dff7d",
                "published",
                "ba13b349a092d690d670ddcc372aa320ca255088a417cfba7e961ab06b8ae9b6",
                "2653a2aa5e6d1829860e9d68d3f480c4e887152aa575b6ab1f57b1cb3c96880f",
            ),
            (
                "829d56295c570fb71bbc45f1e924d8b62e9df6917c7b1c8f432b73937c3c8410",
                "96530558cf2f5986e4e367cab64d3e463493dc291dd954c21c52743dfbabe713",
                "published",
                "4e790fa58cd7dc5b4a3c92f3f93a495bb7a32c96ec427c0bdfefac2dca126e5a",
                "a47fdeceee8446bac0f099130939296c2ba881b5e5b426abf065cd04f9a5df00",
            ),
            (
                "c8bf2ddbd64e53b4574826fa6562758b3618a31cf88f19df737d0c17b9453ff0",
                "91471ef2d2b37fcb9fe8438875b2147e980015a0303a36845d622f9bf711b4c7",
                "published",
                "444e13459b05cc337e46094714cdb9610f61cba2da9837bd7bb327d88e4e20ae",
                "31b9febf74fddddf0b0b224a0e45a6fbe6b06755098f9d2ec15627aea3e57ada",
            ),
            (
                "d7dbc2cb87803ff5e2afc985123f2f9e421f75b23cd125db521dabc37c611f26",
                "6d4f7802ce2ee2c02bd1790e1105e7627b930b6f62a47c322061295142f3557d",
                "ready",
                "a488d39302f7ca86cbeaaa1c6f535b74db5a9ff9cf3d98a57ce50984b5a5d658",
                None,
            ),
        ],
        "artifacts": [
            "2653a2aa5e6d1829860e9d68d3f480c4e887152aa575b6ab1f57b1cb3c96880f.json",
            "31b9febf74fddddf0b0b224a0e45a6fbe6b06755098f9d2ec15627aea3e57ada.json",
            "a47fdeceee8446bac0f099130939296c2ba881b5e5b426abf065cd04f9a5df00.json",
        ],
        "manifest": "cdcd1d9013bf091d1fe5e778c11529078f3861beb28bacae56f304cfa572d14a",
    },
}

NATIVE = {
    "settings": "5ef99cbebc44d2527f5b4e226d55cbced13fc84eabf6c1e816ac4549d37771ea",
    "records": [
        {
            "target_id": "ef758d3cf10d46844a585d414e693b10d4d72cdea73b07bad053d01c05dab385",
            "job_id": "e3e3fc01858ace8a1090097a6fa6337361a1ee224653c7c5c57c414567b457f3",
            "prefix_sha256": "29074d7739c254da0282bdbcbea53cb17875fb9e497ed636b70aef557fb77283",
            "readable_input_sha256": "a319ef28ad8e76f641d5730f148d0366ebab817f2fa78591872b506a7f902852",
            "generation_id": "d60873e870f1fac12d5d3ade1a0fd4cf42a6db8d79fdb27d677ce0026196e138",
        },
        {
            "target_id": "d82cf47c7fcc31d27210f530de4acece1a645b9c16722e02037ee52406b6800b",
            "job_id": "5aa8541b922b4c8fb961db79339fec0f34fc602f4dafdda512321e7b6096593b",
            "prefix_sha256": "7ef70f87de34cdd0b9db7790868d1d28aa284b8fd212035d7bf60db859dc94de",
            "readable_input_sha256": "150ac939180471d624d489fa491dd93c2abec6e2067a18961cfeefa32d85d7d6",
            "generation_id": "f24a3c227f4fbb1d9b51f4d0c8e57a65c563f6e9a2c97d9498729e758948a286",
        },
        {
            "target_id": "ac9f8130182e775e50c8a173c1a808989dace6ca33cc64076437faba32b7c911",
            "job_id": "ac9b0768918dddbba672a5b14c8ec4ab39abf4faad1606d926431d70e282684d",
            "prefix_sha256": "272a92fd20f02f48f81985663eb8d98ca7fbb1fba850044939be12c0fd8e82d0",
            "readable_input_sha256": "effd0605712a77b1580462fc3dc2bcba081976c50cb10c8c7097ba2616804717",
            "generation_id": "0ccc0e0d362ade47d338de2e0d3fb50ff853e00b272941faceb6df647896b1ca",
        },
        {
            "target_id": "59f093c7c97b9dfaf0c80e7823aeb7f2db794b9c262627c763e0a573714f2960",
            "job_id": "7fefd16785498828aa9a9256292f03bc2045fe15f6f921ce5944ecf6dd3f93fc",
            "prefix_sha256": "26b59a3758c0d1a47c00a6715f65036e67f5de7ec937aaefa079a0a6504387f2",
            "readable_input_sha256": "599fcade217eb2438214a70e344b58bc576f510e45020711116003fd15fa94ea",
        },
    ],
    "durable": {
        "targets": [
            "59f093c7c97b9dfaf0c80e7823aeb7f2db794b9c262627c763e0a573714f2960",
            "ac9f8130182e775e50c8a173c1a808989dace6ca33cc64076437faba32b7c911",
            "d82cf47c7fcc31d27210f530de4acece1a645b9c16722e02037ee52406b6800b",
            "ef758d3cf10d46844a585d414e693b10d4d72cdea73b07bad053d01c05dab385",
        ],
        "jobs": [
            (
                "5aa8541b922b4c8fb961db79339fec0f34fc602f4dafdda512321e7b6096593b",
                "d82cf47c7fcc31d27210f530de4acece1a645b9c16722e02037ee52406b6800b",
                "published",
                "4e790fa58cd7dc5b4a3c92f3f93a495bb7a32c96ec427c0bdfefac2dca126e5a",
                "f24a3c227f4fbb1d9b51f4d0c8e57a65c563f6e9a2c97d9498729e758948a286",
            ),
            (
                "7fefd16785498828aa9a9256292f03bc2045fe15f6f921ce5944ecf6dd3f93fc",
                "59f093c7c97b9dfaf0c80e7823aeb7f2db794b9c262627c763e0a573714f2960",
                "ready",
                "a488d39302f7ca86cbeaaa1c6f535b74db5a9ff9cf3d98a57ce50984b5a5d658",
                None,
            ),
            (
                "ac9b0768918dddbba672a5b14c8ec4ab39abf4faad1606d926431d70e282684d",
                "ac9f8130182e775e50c8a173c1a808989dace6ca33cc64076437faba32b7c911",
                "published",
                "444e13459b05cc337e46094714cdb9610f61cba2da9837bd7bb327d88e4e20ae",
                "0ccc0e0d362ade47d338de2e0d3fb50ff853e00b272941faceb6df647896b1ca",
            ),
            (
                "e3e3fc01858ace8a1090097a6fa6337361a1ee224653c7c5c57c414567b457f3",
                "ef758d3cf10d46844a585d414e693b10d4d72cdea73b07bad053d01c05dab385",
                "published",
                "ba13b349a092d690d670ddcc372aa320ca255088a417cfba7e961ab06b8ae9b6",
                "d60873e870f1fac12d5d3ade1a0fd4cf42a6db8d79fdb27d677ce0026196e138",
            ),
        ],
        "artifacts": [
            "0ccc0e0d362ade47d338de2e0d3fb50ff853e00b272941faceb6df647896b1ca.json",
            "d60873e870f1fac12d5d3ade1a0fd4cf42a6db8d79fdb27d677ce0026196e138.json",
            "f24a3c227f4fbb1d9b51f4d0c8e57a65c563f6e9a2c97d9498729e758948a286.json",
        ],
        "manifest": "4a52de62377d3536bb3f0a01e6d339153ba9878655ec4e2f7058f04580abf7da",
    },
}


@pytest.mark.parametrize(
    "rows_for, policy, expected", [(plain, None, PLAIN), (native, POLICY, NATIVE)]
)
def test_durable_rolling_identities_are_pinned(tmp_path, rows_for, policy, expected):
    store, settings, records, directory = build(tmp_path, rows_for, policy)
    assert settings.digest == expected["settings"]
    assert records == expected["records"]
    assert durable(directory) == expected["durable"]
    # Reopening must accept the same artifacts and derive the same identities.
    with rolling_state.RollingJournal(
        tmp_path, "s", settings, store=store, policy=policy
    ) as journal:
        view = journal.load_view()
        pending = journal.pending_target()
        assert view.generation_id == records[-2]["generation_id"]
        assert view.job_id == records[-2]["job_id"]
        assert pending.target_id == records[-1]["target_id"]
        assert pending.job_id == records[-1]["job_id"]
        assert pending.prefix_sha256 == records[-1]["prefix_sha256"]
        assert sha256(pending.to_bytes(settings)) == pending.target_id
        assert journal.inspect(pending.job_id)["state"] == "ready"
    assert durable(directory) == expected["durable"]
