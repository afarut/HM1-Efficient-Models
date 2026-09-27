"""Run on one Kaggle GPU: python measure.py; python calibrate.py."""
import csv,gc,json,platform,time,threading,subprocess
from pathlib import Path
import numpy as np
import torch
from models import SmallCNN
from equations import flops,memory

FIELDS=['S','B','status','latency','memory','energy','is_validation','flops','memory_prediction','energy_method','energy_repeats','energy_window_seconds','power_samples','error']

def grid():
    rng=np.random.default_rng(2026)
    base_s=[32,64,128,224,256,384,512]; base_b=[1,2,4,8,16,32,64,128,256]
    extra_s=sorted(rng.choice([s for s in range(32,513,16) if s not in base_s],4,replace=False).tolist())
    extra_b=sorted(rng.choice([b for b in range(1,257) if b not in base_b],3,replace=False).tolist())
    return [(s,b,s in extra_s or b in extra_b) for s in sorted(base_s+extra_s) for b in sorted(base_b+extra_b)]

class Power:
    def __init__(self):
        import pynvml as nv
        nv.nvmlInit(); self.nv=nv
        props=torch.cuda.get_device_properties(0)
        try:self.h=nv.nvmlDeviceGetHandleByUUID(str(props.uuid))
        except Exception:self.h=nv.nvmlDeviceGetHandleByIndex(0)
        self.nv.nvmlDeviceGetPowerUsage(self.h)
    def watts(self):return self.nv.nvmlDeviceGetPowerUsage(self.h)/1000
    def counter(self):
        try:return self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h)/1000
        except Exception:return None
    def run(self,fn,n):
        samples=[]; stop=threading.Event()
        def poll():
            while not stop.is_set():
                samples.append((time.perf_counter(),self.watts()));stop.wait(.02)
        thread=threading.Thread(target=poll);thread.start()
        e0=self.counter();t0=time.perf_counter();samples.append((t0,self.watts()))
        try:
            for _ in range(n):fn()
            torch.cuda.synchronize()
            t1=time.perf_counter();e1=self.counter();samples.append((t1,self.watts()))
        finally:stop.set();thread.join()
        if e0 is not None and e1 is not None and e1>e0:
            return (e1-e0)/n,'nvml_total_energy',t1-t0,len(samples)
        samples=sorted(samples)
        times=np.array([x[0] for x in samples]);watts=np.array([x[1] for x in samples])
        inner=(times>t0)&(times<t1)
        tx=np.r_[t0,times[inner],t1];pw=np.interp(tx,times,watts)
        return float(np.trapz(pw,tx))/n,'nvml_power_integral',t1-t0,len(samples)

def main():
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(2026);torch.set_num_threads(2)
    assert torch.cuda.is_available(),'GPU required: select Kaggle T4 accelerator'
    torch.cuda.set_device(0)
    out=Path('results');out.mkdir(exist_ok=True)
    p=torch.cuda.get_device_properties(0)
    try:power=Power();power_error=None
    except Exception as e:power=None;power_error=str(e)
    meta={'gpu':p.name,'total_memory':p.total_memory,'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'python':platform.python_version(),'seed':2026,'flags':{'benchmark':False,'cudnn_allow_tf32':False,'matmul_allow_tf32':False},'latency_protocol':'median of 9 individual host wall-clock forwards, synchronize before/after; 3 warmups','energy_protocol':'whole GPU 0 only; target 1.2 sec sustained forwards (actual window saved in CSV); NVML counter, else sampled power integral; no idle subtraction','power_error':power_error,'grid':grid()}
    (out/'environment.json').write_text(json.dumps(meta,indent=2))
    with (out/'measurements.csv').open('w',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=FIELDS);writer.writeheader();fp.flush()
        with torch.inference_mode():
            model=SmallCNN().cuda().eval()
            for index,(s,b,valid) in enumerate(grid()):
                row=dict(S=s,B=b,is_validation=int(valid),flops=float(flops(s,b)),memory_prediction=float(memory(s,b)))
                x=None;y=None
                try:
                    x=torch.randn(b,3,s,s,device='cuda')
                    for _ in range(3):y=model(x);del y
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    y=model(x);torch.cuda.synchronize()
                    row['memory']=torch.cuda.max_memory_allocated();del y
                    times=[]
                    for _ in range(9):
                        torch.cuda.synchronize();t=time.perf_counter();y=model(x);torch.cuda.synchronize();times.append(time.perf_counter()-t);del y
                    row['latency']=float(np.median(times))
                    if power is not None:
                        n=max(3,int(np.ceil(1.2/row['latency'])))
                        e,method,window,count=power.run(lambda:model(x),n)
                        row.update(energy=e,energy_method=method,energy_repeats=n,energy_window_seconds=window,power_samples=count)
                    else:row.update(energy_method='unavailable',error=power_error)
                    row['status']='OK'
                except torch.cuda.OutOfMemoryError as e:
                    row.update(status='OOM',error=str(e).split('\n')[0]);y=None
                finally:
                    x=None;y=None;gc.collect();torch.cuda.empty_cache()
                writer.writerow(row);fp.flush()
                print(f'{index+1}/132 S={s} B={b} {row["status"]} t={row.get("latency")} E={row.get("energy")}',flush=True)
    print('MEASUREMENTS COMPLETE',flush=True)
if __name__=='__main__':main()
