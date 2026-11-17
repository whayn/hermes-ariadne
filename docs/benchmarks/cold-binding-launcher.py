"""Sequential A/B checkpoint campaign, low-RSS launcher, no provider calls."""
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path('/opt/data/cache/scratch/ariadne-19-cold')
FIX = Path('/opt/data/cache/scratch/ariadne-26')
PY = '/opt/data/workspace/hermes-ariadne/.venv/bin/python'
TREES = {'baseline': FIX / 'final-sync', 'binding': ROOT / 'frozen-binding'}
OUT = ROOT / 'open-checkpoints.jsonl'
if OUT.exists():
    raise SystemExit('refuse to overwrite an existing campaign')
ENV = dict(os.environ, HOME=str(ROOT / 'home'), HERMES_HOME=str(ROOT / 'home/hermes'), TMPDIR='/opt/data/cache/scratch')
FP = '''import hashlib,json,sys
from pathlib import Path
from hermes_ariadne import rolling_migrate as m
h=Path(sys.argv[1]);r=h/'ariadne/rolling'/hashlib.sha256(b's').hexdigest(); selected=r/'v3' if (r/'storage.json').exists() else r
v={'active':m.fingerprint(selected),'archive':m._file_hash(h/'ariadne/events.sqlite3'),'root_files':m._files(r),'metadata':{p.name:m._file_hash(p) for p in r.iterdir() if p.suffix=='.json'}}
if (r/'legacy-snapshot').exists():v['snapshot_files']=m._files(r/'legacy-snapshot')
print(json.dumps(v,sort_keys=True))'''

def run(command, label, env):
    with (ROOT / (label + '.stderr')).open('w') as error:
        value = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=error, text=True, check=True)
    (ROOT / (label + '.stdout')).write_text(value.stdout)
    return json.loads(value.stdout)

def fingerprint(home, label):
    env = dict(ENV, PYTHONPATH=str(TREES['binding'] / 'src'))
    return run([PY, '-c', FP, str(home)], label, env)

cases=[]
for size in (256,4096,16384):
    for repeat in range(1,4):
        for mode in TREES:
            cases.append((mode,'native',size,repeat,FIX / f'native3-{size}'))
for kind,path in [('native',FIX/'native3-100000'),('migrated',FIX/'migrated-213cd77-100000'),('legacy',FIX/'legacy-256')]:
    for mode in TREES:
        cases.append((mode,kind,100000 if kind!='legacy' else 256,1,path))
assert len(cases)==24
for index,(mode,kind,size,repeat,home) in enumerate(cases,1):
    if not home.is_dir(): raise SystemExit('missing fixture '+str(home))
    label=f'{index:02}-{mode}-{kind}-{size}-{repeat}'
    before=fingerprint(home,label+'-before')
    tree=TREES[mode]
    data=run([PY,str(tree/'scripts/bench_rolling_open.py'),'measure',str(home),'--in-place','--cache'],label,dict(ENV,PYTHONPATH=str(tree/'src')))
    after=fingerprint(home,label+'-after')
    if before!=after:raise RuntimeError('fixture changed: '+label)
    assert data['events']==size and data['generations']==64
    assert data['open_source_checks']==data['warm_source_checks']==size
    assert str(tree/'src/hermes_ariadne')==data['source']
    if kind!='legacy':assert data['stored_identity_references']==2*size
    record={'mode':mode,'kind':kind,'size':size,'repeat':repeat,'measurement':data,'before':before,'after':after}
    with OUT.open('a') as stream:stream.write(json.dumps(record,sort_keys=True)+'\n')
    print(json.dumps({'completed':index,'of':len(cases),'label':label,'cold':data['open_s'],'warm':data['warm_open_s'],'reads':data['open_artifact_reads']}),flush=True)
records=[json.loads(line) for line in OUT.read_text().splitlines()]
assert len(records)==24
counts=Counter((r['mode'],r['kind'],r['size']) for r in records)
assert all(v==(3 if k[1]=='native' and k[2]!=100000 else 1) for k,v in counts.items())
(ROOT/'open-checkpoints-complete.json').write_text(json.dumps({'complete':True,'pairs':24,'groups':len(counts)}))
print('checkpoint campaign complete',flush=True)
