"""One supervised handoff of R28 evaluation queues; existing children finish."""
import argparse,hashlib,json,os,shlex,signal,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,'/home/oycy/MMDD/src')
from run_stage1_r28_evaluations import alive,jobs_for
from prepare_stage1_r28 import ROOT,OUT


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def launch(name: str, command: list[str], log: Path) -> None:
    shell_command=shlex.join(command)+' > '+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',name,'-c','/tmp/mmdd-r28-checks',shell_command],check=True)


def main(execute: bool) -> None:
    queues={s:json.loads((OUT/f'EVALUATION_QUEUE_seed{s}.json').read_text()) for s in (13,29)}
    current=json.loads((OUT/'CURRENT_STATUS.json').read_text())
    released={s:[r['arm'] for r in current['training'] if r['seed']==s and r['arm'].startswith('T-') and r['status']=='completed'] for s in queues}
    print(json.dumps({'released_teachers':released,'old_workers':{s:r['worker_pid'] for s,r in queues.items()},'execute':execute}),flush=True)
    if not execute:return
    assert all(released.values()), 'Wait for at least one completed Teacher per GPU'
    assert json.loads((OUT/'SPLIT_EVALUATION_QUEUE_PREFLIGHT.json').read_text())['status']=='pass'
    receipt_path=OUT/'EVALUATION_QUEUE_HANDOFF.json'
    assert not receipt_path.exists(), 'Inspect existing handoff before any further action'
    monitor=current['monitor_pid']
    assert b'run_stage1_r28_finish.py' in Path(f'/proc/{monitor}/cmdline').read_bytes()
    for seed,q in queues.items():
        pid=q['worker_pid']
        assert b'run_stage1_r28_evaluations.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
        assert Path(f'/proc/{pid}/wchan').read_text().strip()=='do_wait', 'Supervisor must be waiting for its child'
        for kind in ('student','teacher'):
            assert not (OUT/f'EVALUATION_QUEUE_seed{seed}_{kind}.json').exists()
            assert subprocess.run(['tmux','has-session','-t',f'r28-evaluation-{seed}-{kind}'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
    assert subprocess.run(['tmux','has-session','-t','r28-finalization-monitor-split'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
    stopped=[]
    try:
        for pid in [monitor]+[q['worker_pid'] for q in queues.values()]:
            os.kill(pid,signal.SIGSTOP); stopped.append(pid)
            for _ in range(100):
                if Path(f'/proc/{pid}/stat').read_text().rpartition(')')[2].split()[0]=='T':break
                time.sleep(.01)
            assert Path(f'/proc/{pid}/stat').read_text().rpartition(')')[2].split()[0]=='T'
        for seed,q in list(queues.items()):
            frozen=json.loads((OUT/f'EVALUATION_QUEUE_seed{seed}.json').read_text())
            assert frozen['worker_pid']==q['worker_pid']
            active=[j for j in frozen['jobs'] if j['status']=='running']
            children={int(p) for p in Path(f"/proc/{q['worker_pid']}/task/{q['worker_pid']}/children").read_text().split()}
            assert len(active)<=1 and children<={j['pid'] for j in active}, 'Untracked child: resume and inspect'
            registered={j['id']:j['command'] for j in jobs_for(seed,q['device'])}
            assert all(j['command']==registered[j['id']] for j in frozen['jobs'])
            queues[seed]=frozen
    except BaseException:
        for pid in reversed(stopped):os.kill(pid,signal.SIGCONT)
        raise
    receipt={'status':'old_supervisors_paused','released_teachers':released,'old_monitor':monitor,'old_queues':queues,
             'code':{'path':str(Path(__file__)),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},'new_queues':[]}
    save(receipt_path,receipt)
    archive=OUT/'queue_handoff'; archive.mkdir(exist_ok=True)
    for seed,q in queues.items():
        old_path=OUT/f'EVALUATION_QUEUE_seed{seed}.json'
        assert not (archive/old_path.name).exists()
        old_path.rename(archive/old_path.name)
        active=[j for j in q['jobs'] if j['status']=='running']
        for kind in ('student','teacher'):
            command=[sys.executable,str(ROOT/'src/run_stage1_r28_evaluations.py'),'--seed',str(seed),'--device',q['device'],'--kind',kind]
            inherited=next((j for j in active if j['kind']==kind),None)
            if inherited:command+=['--wait-pid',str(inherited['pid']),'--wait-receipt',inherited['receipt']]
            name=f'r28-evaluation-{seed}-{kind}'
            launch(name,command,OUT/'logs'/f'evaluation-{seed}-{kind}.log')
            receipt['new_queues'].append({'seed':seed,'kind':kind,'session':name,'command':command,'ledger':str(OUT/f'EVALUATION_QUEUE_seed{seed}_{kind}.json')})
    receipt['status']='split_workers_launched'; save(receipt_path,receipt)
    while not all(Path(r['ledger']).exists() for r in receipt['new_queues']):
        for r in receipt['new_queues']:
            assert subprocess.run(['tmux','has-session','-t',r['session']],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
        time.sleep(2)
    for r in receipt['new_queues']:
        q=json.loads(Path(r['ledger']).read_text()); assert alive(q['worker_pid'])
        r['worker_pid']=q['worker_pid']; r['expected_nodes']=len(q['jobs'])
    assert sum(r['expected_nodes'] for r in receipt['new_queues'])==66
    assert not Path(f'/proc/{monitor}/task/{monitor}/children').read_text().strip()
    launch('r28-finalization-monitor-split',[sys.executable,str(ROOT/'src/run_stage1_r28_finish.py'),'--t0-pid','314636'],OUT/'logs/finalization-monitor-split.log')
    os.kill(monitor,signal.SIGKILL)
    receipt['status']='new_queues_running_retiring_old_supervisors'; save(receipt_path,receipt)
    print(json.dumps({'status':receipt['status'],'new_workers':[(r['seed'],r['kind'],r['worker_pid']) for r in receipt['new_queues']]}),flush=True)
    for seed,q in queues.items():
        active=[j for j in q['jobs'] if j['status']=='running']
        while any(alive(j['pid']) for j in active):time.sleep(5)
        assert all(Path(j['receipt']).exists() and json.loads(Path(j['receipt']).read_text())['status']=='completed' for j in active)
        os.kill(q['worker_pid'],signal.SIGKILL)
        receipt.setdefault('retired_old_workers',[]).append(q['worker_pid']); save(receipt_path,receipt)
    receipt['status']='completed'; save(receipt_path,receipt)
    print(json.dumps({'status':'completed','training_jobs_restarted':0,'evaluation_nodes_added':0,'new_queue_workers':4}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--execute',action='store_true')
    main(parser.parse_args().execute)
