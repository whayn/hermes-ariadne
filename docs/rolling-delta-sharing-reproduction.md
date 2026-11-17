# Delta-sharing reproduction archive

These are the exact executed launchers for runtime `2dc33570cb6c810ff94125055757b5e3235c078a` and baseline `b7b432a54166c126def78e6c22a54def131539d2`. They are historical execution records, not portable commands or an exclusive-host claim. The paths and pidfd barrier PID are the ones actually used. For a new run, select fresh scratch paths, explicitly pin both source trees, replace the historical barrier PID with your own job, and refuse existing outputs; never blindly replay a PID that may have been reused. Fingerprints run in low-RSS children.

Actual campaign launch (after creating the detached source trees):

```bash
. /opt/data/cache/scratch/ariadne-19-sharing/env.sh
.venv/bin/python "$ARIADNE_SCR/run-checkpoints.py" > "$ARIADNE_SCR/campaign.log" 2>&1
```

The profile launcher was started separately, waited via pidfd for campaign PID 26311, and required the 24-pair completion marker before doing CPU-heavy work. It then ran the candidate callback profile and sequential cold baseline/candidate profiles. The baseline callback profile had already run, without competing local heavy jobs, through the same archived callback script against b7 source. The isolated env and actual source origins/CPU affinity are recorded in the evidence and main document. No provider requests are made by these scripts.

## run-checkpoints.py

SHA-256: `d2a360d19cfa0b06224cf36d4f4d452ac80a982219a4d7dd351f203654c7c9f9`.

```python
"""Frozen sequential A/B campaign; fingerprinting stays in low-RSS children."""
import json
import os
import platform
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path('/opt/data/cache/scratch/ariadne-19-sharing')
FIX = Path('/opt/data/cache/scratch/ariadne-26')
PY = '/opt/data/workspace/hermes-ariadne/.venv/bin/python'
TREES = {'baseline': ROOT/'frozen-baseline', 'sharing': ROOT/'frozen-runtime'}
SHAS = {'baseline': 'b7b432a54166c126def78e6c22a54def131539d2', 'sharing': '2dc33570cb6c810ff94125055757b5e3235c078a'}
HARNESS = TREES['sharing']/'scripts/bench_rolling_open.py'
OUT = ROOT/'open-checkpoints.jsonl'
if OUT.exists(): raise SystemExit('refuse to overwrite campaign')
ENV = dict(os.environ, HOME=str(ROOT/'home'), HERMES_HOME=str(ROOT/'home/hermes'), TMPDIR='/opt/data/cache/scratch')
for mode,tree in TREES.items():
 actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=tree,text=True).strip()
 if actual!=SHAS[mode]:raise SystemExit('wrong source head '+mode)
metadata={'heads':SHAS,'uname':list(platform.uname()),'affinity':sorted(os.sched_getaffinity(0)), 'cpuinfo':Path('/proc/cpuinfo').read_text(), 'launch':'inherited affinity; no taskset or governor changes; sequential local campaign; no provider'}
(ROOT/'campaign-environment.json').write_text(json.dumps(metadata,indent=2)+'\n')
FP='''import hashlib,json,sys
from pathlib import Path
from hermes_ariadne import rolling_migrate as m
h=Path(sys.argv[1]);r=h/'ariadne/rolling'/hashlib.sha256(b's').hexdigest();selected=r/'v3' if (r/'storage.json').exists() else r
v={'active':m.fingerprint(selected),'archive':m._file_hash(h/'ariadne/events.sqlite3'),'root_files':m._files(r),'metadata':{p.name:m._file_hash(p) for p in r.iterdir() if p.suffix=='.json'}}
if (r/'legacy-snapshot').exists():v['snapshot_files']=m._files(r/'legacy-snapshot')
print(json.dumps(v,sort_keys=True))'''
def run(command,label,env):
 with (ROOT/(label+'.stderr')).open('w') as error:
  value=subprocess.run(command,env=env,stdout=subprocess.PIPE,stderr=error,text=True,check=True)
 (ROOT/(label+'.stdout')).write_text(value.stdout)
 return json.loads(value.stdout)
def fingerprint(home,label):
 return run([PY,'-c',FP,str(home)],label,dict(ENV,PYTHONPATH=str(TREES['sharing']/'src')))
cases=[]
for size in (256,4096,16384):
 for repeat in range(1,4):
  for mode in TREES:cases.append((mode,'native',size,repeat,FIX/f'native3-{size}'))
for kind,path in [('native',FIX/'native3-100000'),('migrated',FIX/'migrated-213cd77-100000'),('legacy',FIX/'legacy-256')]:
 for mode in TREES:cases.append((mode,kind,100000 if kind!='legacy' else 256,1,path))
assert len(cases)==24
inventories={}
for index,(mode,kind,size,repeat,home) in enumerate(cases,1):
 if not home.is_dir():raise SystemExit('missing fixture '+str(home))
 label=f'{index:02}-{mode}-{kind}-{size}-{repeat}'
 before=fingerprint(home,label+'-before')
 key=(kind,size)
 if key in inventories and inventories[key]!=before:raise RuntimeError('cross-variant fixture changed '+label)
 inventories[key]=before
 tree=TREES[mode]
 data=run([PY,str(HARNESS),'measure',str(home),'--in-place','--cache'],label,dict(ENV,PYTHONPATH=str(tree/'src')))
 after=fingerprint(home,label+'-after')
 if before!=after:raise RuntimeError('fixture changed '+label)
 assert data['events']==size and data['generations']==64
 assert data['open_source_checks']==data['warm_source_checks']==size
 assert data['source']==str(tree/'src/hermes_ariadne')
 if kind!='legacy':
  assert data['stored_identity_references']==2*size
  assert data['open_canonical_calls']==(65 if mode=='sharing' else 128)
  assert data['warm_canonical_calls']==0
  if mode=='sharing':
   assert data['open_source_proofs']['hits']==63
   assert data['open_source_proofs']['accepted']==64
   assert data['open_source_proofs']['pressure']==0
   assert data['open_source_proofs']['input_bytes']<=16*1024*1024
   assert data['warm_source_proofs']['accepted']==0
 record={'mode':mode,'kind':kind,'size':size,'repeat':repeat,'sha':SHAS[mode],'measurement':data,'before':before,'after':after}
 with OUT.open('a') as stream:stream.write(json.dumps(record,sort_keys=True)+'\n')
 print(json.dumps({'completed':index,'of':len(cases),'label':label,'cold':data['open_s'],'warm':data['warm_open_s'],'canonical':data['open_canonical_rows']}),flush=True)
records=[json.loads(line) for line in OUT.read_text().splitlines()]
assert len(records)==24
counts=Counter((r['mode'],r['kind'],r['size']) for r in records)
assert len(counts)==12 and len(inventories)==6
assert all(v==(3 if k[1]=='native' and k[2]!=100000 else 1) for k,v in counts.items())
(ROOT/'open-checkpoints-complete.json').write_text(json.dumps({'complete':True,'pairs':24,'groups':12,'unchanged_inventories':6}))
print('checkpoint campaign complete',flush=True)
```

## run-profiles.py

SHA-256: `e240347c0dd6803f55bc8aee3f8eca31a3c0ae0c6f0569cb19307d96b5bb936d`.

```python
"""Run callback/cold profiles only after the sequential checkpoint campaign."""
import ast
import json
import os
import select
import subprocess
from pathlib import Path

ROOT=Path('/opt/data/cache/scratch/ariadne-19-sharing')
PY='/opt/data/workspace/hermes-ariadne/.venv/bin/python'
try:
 fd=os.pidfd_open(26311)
except ProcessLookupError:
 fd=None
if fd is not None:
 select.select([fd],[],[])
 os.close(fd)
marker=json.loads((ROOT/'open-checkpoints-complete.json').read_text())
if marker!={'complete':True,'pairs':24,'groups':12,'unchanged_inventories':6}:
 raise SystemExit('campaign incomplete; do not start subsequent work')
base=dict(os.environ, HOME=str(ROOT/'home'), HERMES_HOME=str(ROOT/'home/hermes'), TMPDIR='/opt/data/cache/scratch')
site='/opt/data/cache/scratch/ariadne-26/host-sdk/lib/python3.14/site-packages'
host='/opt/hermes/.venv/lib/python3.14/site-packages'
tree=ROOT/'frozen-runtime'
env=dict(base,PYTHONPATH=site+':'+str(tree/'src')+':'+host)
with (ROOT/'callbacks-sharing.json').open('w') as out, (ROOT/'callbacks-sharing.stderr').open('w') as err:
 subprocess.run([PY,'/opt/data/cache/scratch/ariadne-19-cold/profile-callbacks.py',str(tree),str(ROOT/'callbacks-sharing')],env=env,cwd=tree,stdout=out,stderr=err,check=True)
# Exact same isolated fingerprint child used in the campaign.
module=ast.parse((ROOT/'run-checkpoints.py').read_text())
fp=next(ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FP' for t in n.targets))
fixture='/opt/data/cache/scratch/ariadne-26/native3-16384'
def fingerprint():
 return json.loads(subprocess.check_output([PY,'-c',fp,fixture],env=env,text=True))
before=fingerprint()
for mode in ('baseline','runtime'):
 current=ROOT/('frozen-'+mode)
 with (ROOT/f'cold-profile-{mode}.json').open('w') as out, (ROOT/f'cold-profile-{mode}.txt').open('w') as err:
  subprocess.run([PY,str(tree/'scripts/bench_rolling_open.py'),'measure',fixture,'--in-place','--profile','35'],env=dict(base,PYTHONPATH=str(current/'src')),cwd=current,stdout=out,stderr=err,check=True)
 if fingerprint()!=before:raise RuntimeError('profile fixture changed '+mode)
(ROOT/'profiles-complete.json').write_text(json.dumps({'complete':True,'fixture_unchanged':True}))
print('callback and cold profiles complete; fixture unchanged',flush=True)
```

## profile-callbacks.py

SHA-256: `81872eb7bc804cac1e89a73deaad1ee3d003e302d3157a4d3cad7701035a716c`.

```python
"""Disposable engine-callback profile using existing synthetic test fixtures.

No AIAgent/native dispatch/provider calls: this exercises real engine public
callbacks against the existing injected DB/Facade. Set isolated HOME/HERMES_HOME,
HERMES_SOURCE and PYTHONPATH to the frozen package under test before invocation.
Arguments: REPO_FIXTURES OUTPUT_DIRECTORY. It writes only beneath that new scratch
output directory and creates profiles after a 32-generation fixture seed.
"""

import copy
import cProfile
import io
import json
import os
import pstats
import shutil
import sys
import time
from pathlib import Path

repo, output = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
home = Path(os.environ["HERMES_HOME"]).resolve()
if home == Path("/opt/data") or "/cache/scratch/" not in str(output) or output.exists():
    raise SystemExit("fresh scratch output and isolated home required")
output.mkdir(mode=0o700)
sys.path.insert(0, str(repo / "tests"))
sys.path.insert(0, os.environ["HERMES_SOURCE"])

import pytest

import hermes_ariadne
from hermes_ariadne import rolling
from test_rolling_engine import DB, complete, exchange, select, setup_engine

origin = str(Path(hermes_ariadne.__file__).resolve())
results = {"origin": origin, "claim": "synthetic-injected-facade-callbacks-not-provider-latency", "generations": 32}
with pytest.MonkeyPatch.context() as patch:
    engine, db, facade = setup_engine(output / "untraced", patch)
    for index in range(33):
        tag = f"turn-{index}"
        db.rows += exchange(tag, complete=False)[:1]
        select(engine, db, tag)
        db.rows = db.rows[:-1] + exchange(tag)
        complete(engine, db, tag)
    if engine._published.sequence != 32:
        raise RuntimeError("fixture did not reach32 publications")
    tag = "probe"
    db.rows += exchange(tag, complete=False)[:1]
    # No callbacks/journals are running while copying these synthetic files.
    shutil.copytree(output / "untraced", output / "profiled")
    other_db = DB()
    other_db.rows = copy.deepcopy(db.rows)
    profiled, _, _ = setup_engine(output / "profiled", patch, db=other_db)

    for label, candidate, source_db, trace in (("untraced", engine, db, False), ("profiled", profiled, other_db, True)):
        def selection():
            if select(candidate, source_db, tag) is None:
                raise RuntimeError("selection declined")

        def navigation():
            response = json.loads(candidate.handle_tool_call("ariadne_rolling_page", {"ref": "rolling:" + candidate._pinned.generation_id}))
            if not response["ok"]:
                raise RuntimeError("navigation declined")

        def completion():
            source_db.rows = source_db.rows[:-1] + exchange(tag)
            complete(candidate, source_db, tag)
            if candidate._published.sequence != 33:
                raise RuntimeError("completion did not publish generation33")

        for name, function in (("select", selection), ("navigation", navigation), ("completion", completion)):
            candidate._verification = rolling.RollingVerificationCache()
            profiler = cProfile.Profile()
            started = time.perf_counter()
            if trace:
                profiler.runcall(function)
                stream = io.StringIO()
                pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats("hermes_ariadne", 25)
                (output / (name + ".profile.txt")).write_text(stream.getvalue())
            else:
                function()
            results[label + "_" + name + "_s"] = round(time.perf_counter() - started, 6)
    results["final_publications"] = [engine._published.sequence, profiled._published.sequence]
print(json.dumps(results))
```

## run-mutants.py

SHA-256: `0c19d71ee6243184c74e72505bca15b205ed553b8d0c28e6db6168c1cc3181e0`.

```python
"""Run guard mutants against isolated source/test copies, never the real tree."""
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path('/opt/data/cache/scratch/ariadne-19-sharing')
WT = Path('/opt/data/workspace/hermes-ariadne/.worktrees/19-delta-sharing')
PY = str(WT / '.venv/bin/python')
M = 'src/hermes_ariadne/rolling_delta.py'
TEST = 'tests/test_rolling_delta_sharing.py::'
cases = [
 ('accepted', M, 'memo.accepted.get(view.generation_id) is not view\n            or recorded is None', 'False\n            or recorded is None', TEST+'test_incompatible_proof_does_not_bypass_the_full_checker[accepted]'),
 ('input', M, '            or target.readable_input != proof.readable_input\n', '', TEST+'test_incompatible_proof_does_not_bypass_the_full_checker[input]'),
 ('parent', M, '                or state[1] is not proof.parent_state\n', '', TEST+'test_incompatible_proof_does_not_bypass_the_full_checker[parent-state]'),
 ('scope', M, 'if not self.in_scope(memo, store, session, settings.digest, policy) or memo.bound is None:', 'if memo.bound is None:', TEST+'test_incompatible_proof_does_not_bypass_the_full_checker[scope]'),
 ('records', M, 'owner.records >= _PROOF_RECORDS', 'False', TEST+'test_proof_pressure_falls_back_without_changing_results[2-16777216-2]'),
 ('bytes', M, 'or owner.input_bytes + len(readable_input) > _PROOF_BYTES', 'or False', TEST+'test_proof_pressure_falls_back_without_changing_results[256-1-0]'),
 ('capture', M, '        for _ in target.delta_event_ids:\n            guard()\n            guard()\n', '', TEST+'test_proof_hit_keeps_capture_checks_and_failed_operation_cleanup'),
 ('cleanup', 'src/hermes_ariadne/rolling_state.py', '            if proofs is not None:\n                proofs.clear()', '            pass', TEST+'test_late_startup_failure_discards_candidates_and_never_promotes'),
]
def hashes():
 return {str(p.relative_to(WT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (WT/'src').rglob('*.py')}
before=hashes()
results=[]
for name,path,old,new,test in cases:
 work=ROOT/'mutants'/name
 if work.exists():raise SystemExit('refuse existing mutant '+str(work))
 work.mkdir(parents=True)
 for part in ('src','tests'):shutil.copytree(WT/part,work/part)
 shutil.copy2(WT/'pyproject.toml',work/'pyproject.toml')
 file=work/path
 text=file.read_text()
 if text.count(old)!=1:raise RuntimeError('nonunique mutant '+name)
 file.write_text(text.replace(old,new))
 env=dict(os.environ,PYTHONPATH='/opt/data/cache/scratch/ariadne-26/host-sdk/lib/python3.14/site-packages:'+str(work/'src')+':/opt/hermes/.venv/lib/python3.14/site-packages')
 origin=subprocess.check_output([PY,'-c','import hermes_ariadne;print(hermes_ariadne.__file__)'],cwd=work,env=env,text=True).strip()
 if not origin.startswith(str(work/'src')):raise RuntimeError('wrong mutant import '+origin)
 run=subprocess.run([PY,'-m','pytest',test,'-q'],cwd=work,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 (ROOT/f'mutant-{name}.log').write_text(run.stdout)
 killed=run.returncode==1 and '1 failed' in run.stdout and 'ERROR collecting' not in run.stdout
 result={'guard':name,'killed':killed,'exit':run.returncode,'origin':origin,'test':test}
 results.append(result)
 print(json.dumps(result),flush=True)
if before!=hashes():raise RuntimeError('real source changed during mutants')
(ROOT/'mutants.json').write_text(json.dumps({'results':results,'source_unchanged':True},indent=2)+'\n')
if not all(r['killed'] for r in results):raise SystemExit('surviving or invalid mutant')
```

