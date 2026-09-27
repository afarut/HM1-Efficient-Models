"""Supplemental real CUDA OOM search; main 132-point dataset is untouched.
Run: python oom_probe.py. Each configuration uses a fresh subprocess on GPU 0.
"""
import argparse, csv, json, math, os, subprocess, sys
from pathlib import Path


def worker(s, b, destination):
    import time
    import torch
    from models import SmallCNN
    from equations import memory
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    assert torch.cuda.is_available(), 'A real CUDA GPU is required'
    torch.cuda.set_device(0)
    p = torch.cuda.get_device_properties(0)
    row = dict(S=s, B=b, status='', memory=None, peak_before_failure=None,
               memory_prediction=float(memory(s,b)), gpu=p.name,
               total_memory=p.total_memory, torch=torch.__version__,
               cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
               phase='model', error='', experiment='supplemental_oom',
               is_validation=1, latency=None, energy=None)
    model = x = y = None
    try:
        with torch.inference_mode():
            model = SmallCNN().cuda().eval()
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            row['free_before_input'] = free
            row['allocated_before_input'] = torch.cuda.memory_allocated()
            # Fixed model is already allocated; compare input/activation estimate
            # with current free memory, in addition to nominal total capacity.
            row['available_tensor_budget'] = free + row['allocated_before_input']
            row['predicted_oom_total'] = int(row['memory_prediction'] > total)
            row['predicted_oom_available'] = int(row['memory_prediction'] > row['available_tensor_budget'])
            row['phase'] = 'input'
            x = torch.randn(b, 3, s, s, device='cuda')
            row['phase'] = 'warmup'
            for _ in range(3):
                y = model(x)
                del y
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            row['phase'] = 'forward'
            t = time.perf_counter()
            y = model(x)
            torch.cuda.synchronize()
            row['single_forward_seconds'] = time.perf_counter() - t
            row['memory'] = torch.cuda.max_memory_allocated()
            row['status'] = 'OK'
    except torch.cuda.OutOfMemoryError as e:
        row['status'] = 'OOM'
        row['peak_before_failure'] = torch.cuda.max_memory_allocated()
        row['error'] = str(e)
    # Any other error propagates; it must never be labelled as OOM.
    Path(destination).write_text(json.dumps(row, indent=2))
    print(json.dumps({k:row[k] for k in ['S','B','status','phase','memory','memory_prediction']}), flush=True)


def main():
    out = Path('results/oom');out.mkdir(parents=True,exist_ok=True)
    rows = []
    def probe(b, label='search'):
        path = out/f'probe_{len(rows):02d}_B{b}.json'
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker',
                        '--batch',str(b),'--output',str(path)], check=True)
        row = json.loads(path.read_text());row['trial']=len(rows);row['purpose']=label
        rows.append(row)
        fields = sorted(set().union(*(x.keys() for x in rows)))
        with (out/'measurements_oom.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
        return row
    # Establish a bracket by actual allocations, without lowering memory limits.
    low=0;high=256
    while True:
        row=probe(high)
        if row['status']=='OOM':break
        low=high;high*=2
        if high>8192:raise RuntimeError('No OOM found within safety cap; inspect hardware')
    initial=[low,high]
    while high-low>1:
        mid=(low+high)//2
        if probe(mid)['status']=='OK':low=mid
        else:high=mid
    confirmations=[]
    for _ in range(2):
        if low:confirmations.append(probe(low,'boundary_repeat'))
        confirmations.append(probe(high,'boundary_repeat'))
    stable=all(x['status']==('OK' if x['B']==low else 'OOM') for x in confirmations)
    from equations import PARAMETER_BYTES
    capacity=rows[0]['total_memory']
    predicted_first=math.floor((capacity-PARAMETER_BYTES)/(68*512**2))+1
    # Also test the analytic boundary itself; no extrapolated measurement claims.
    for b in [predicted_first-1,predicted_first]:
        if b>0 and not any(x['B']==b for x in rows):probe(b,'analytic_boundary')
    summary=dict(S=512,initial_bracket=initial,largest_success_in_search=low,
                 smallest_oom_in_search=high,boundary_repeats_stable=stable,
                 predicted_first_oom_total=predicted_first,total_memory=capacity,
                 gpu=rows[0]['gpu'],trials=len(rows),
                 real_oom_trials=sum(x['status']=='OOM' for x in rows),
                 false_negatives_total=sum(x['status']=='OOM' and not x['predicted_oom_total'] for x in rows),
                 false_positives_total=sum(x['status']=='OK' and x['predicted_oom_total'] for x in rows),
                 protocol='One fresh subprocess per trial, CUDA GPU 0, actual OOM only; eval/inference_mode FP32, 3 warmups then one measured forward. No artificial memory cap. No energy/median latency measured in this supplemental experiment.',
                 caveat='Binary search assumes local monotonic feasibility. The adjacent boundary is repeated; cuDNN algorithm changes can make global feasibility nonmonotonic. Results apply to this session and software.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('OOM EXPERIMENT COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true')
    parser.add_argument('--batch',type=int);parser.add_argument('--output')
    args=parser.parse_args()
    if args.worker:worker(512,args.batch,args.output)
    else:main()
