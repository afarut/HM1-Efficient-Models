"""Fit on base grid only. Random sizes/batches are held out before measuring."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares,nnls
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from equations import flops,memory,latency,energy,bytes_moved,layer_costs

def main():
    out=Path('results');fig=out/'figures';fig.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(out/'measurements.csv');ok=df[df.status=='OK'].copy();train=ok[ok.is_validation==0]
    s=train.S.to_numpy();b=train.B.to_numpy();t=train.latency.to_numpy()
    def unpack(x):
        v=np.exp(x);return dict(launch_seconds=v[0],flops_per_second=v[1],bytes_per_second=v[2])
    fit=least_squares(lambda x:np.log(latency(s,b,unpack(x)))-np.log(t),np.log([1e-5,2e12,1e11]),bounds=(np.log([1e-9,1e8,1e7]),np.log([.1,1e15,1e14])))
    theta=unpack(fit.x)
    etrain=train[train.energy.notna() & (train.energy>0)]
    te=None
    if len(etrain)>=3:
        se,be=etrain.S.to_numpy(),etrain.B.to_numpy()
        X=np.column_stack([flops(se,be)/1e12,bytes_moved(se,be)/1e9,latency(se,be,theta)])
        target=etrain.energy.to_numpy();coef,_=nnls(X/target[:,None],np.ones(len(target)))
        te=dict(joules_per_flop=coef[0]/1e12,joules_per_byte=coef[1]/1e9,baseline_watts=coef[2],latency=theta)
    (out/'theta.json').write_text(json.dumps({'latency':theta,'energy':te,'training_rule':'base S x base B only; all random S or B held out'},indent=2))
    ss,bb=ok.S.to_numpy(),ok.B.to_numpy()
    ok['latency_pred']=latency(ss,bb,theta);ok['memory_pred']=memory(ss,bb);ok['flops_pred']=flops(ss,bb)
    if te:ok['energy_pred']=energy(ss,bb,te)
    metrics={}
    for name in ['latency','memory']+(['energy'] if te else []):
        metrics[name]={}
        for split,v in [('train',0),('validation',1)]:
            d=ok[(ok.is_validation==v)&ok[name].notna()];err=np.abs(d[name+'_pred']/d[name]-1)
            metrics[name][split]={'n':len(d),'median_absolute_relative_error':float(np.median(err)),'mean_absolute_relative_error':float(np.mean(err)),'p90_absolute_relative_error':float(np.quantile(err,.9))}
        factor,label={'latency':(1e3,'Latency (ms)'), 'memory':(1/2**20,'Peak allocated memory (MiB)'), 'energy':(1,'Whole-GPU energy per forward (J)')}[name]
        plt.figure(figsize=(6,5))
        for v,marker,lab in [(0,'o','Training'),(1,'x','Held out')]:
            d=ok[(ok.is_validation==v)&ok[name].notna()]
            plt.scatter(d[name]*factor,d[name+'_pred']*factor,label=lab,marker=marker,alpha=.7)
        vals=np.r_[ok[name].dropna()*factor,ok[name+'_pred'].dropna()*factor];lo=max(vals.min()*.8,1e-10);hi=vals.max()*1.2
        plt.plot([lo,hi],[lo,hi],'k--',label='Ideal');plt.xscale('log');plt.yscale('log');plt.xlabel('Measured '+label);plt.ylabel('Predicted '+label);plt.legend();plt.tight_layout();plt.savefig(fig/(name+'_parity.png'),dpi=160);plt.close()
        f,axes=plt.subplots(3,4,figsize=(15,10),sharex=True)
        for ax,batch in zip(axes.flat,sorted(df.B.unique())):
            d=ok[ok.B==batch].sort_values('S');xs=np.linspace(32,512,160)
            fn={'latency':lambda:latency(xs,batch,theta),'memory':lambda:memory(xs,batch),'energy':lambda:energy(xs,batch,te)}[name]
            ax.plot(xs,fn()*factor,label='Prediction')
            for v,m,lab in [(0,'o','Training'),(1,'x','Held out')]:
                part=d[d.is_validation==v];ax.scatter(part.S,part[name]*factor,marker=m,s=20,label=lab)
            ax.set_title(f'B={batch}');ax.set_yscale('log');ax.set_xlabel('S (pixels)');ax.set_ylabel(label);ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=7);f.tight_layout();f.savefig(fig/(name+'_grid.png'),dpi=140);plt.close(f)
    env=json.loads((out/'environment.json').read_text())
    df['predicted_oom']=df.memory_prediction>env['total_memory'];df.to_csv(out/'oom_analysis.csv',index=False)
    ok.to_csv(out/'predictions.csv',index=False)
    metrics['oom']={'observed':int((df.status=='OOM').sum()),'predicted_by_tensor_model':int(df.predicted_oom.sum()),'false_negatives':int(((df.status=='OOM')&~df.predicted_oom).sum()),'false_positives':int(((df.status=='OK')&df.predicted_oom).sum())}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    # FLOP accounting is checked with PyTorch profiler on 3 representative points.
    if (out/'flops_check.csv').exists():
        d=pd.read_csv(out/'flops_check.csv');plt.figure(figsize=(6,4));plt.plot(d.analytic_mac_flops,d.analytic_mac_flops,'k--',label='Analytic Conv/Linear');plt.scatter(d.analytic_mac_flops,d.profiler_flops,label='PyTorch profiler');plt.xlabel('Analytic Conv/Linear FLOPs');plt.ylabel('Profiled Conv/Linear FLOPs');plt.legend();plt.tight_layout();plt.savefig(fig/'flops_check.png',dpi=160);plt.close()
    f,q=layer_costs(ss,bb);launch=np.full(len(ok),17*theta['launch_seconds']);comp=(f/theta['flops_per_second']).sum(-1);mem=(q/theta['bytes_per_second']).sum(-1)
    plt.figure(figsize=(7,5));sc=plt.scatter(flops(ss,bb)/bytes_moved(ss,bb),flops(ss,bb)/ok.latency/1e12,c=ok.S,cmap='viridis',label='Measured');plt.colorbar(sc,label='S (pixels)');plt.scatter(flops(ss,bb)/bytes_moved(ss,bb),flops(ss,bb)/latency(ss,bb,theta)/1e12,c='tab:orange',marker='x',alpha=.6,label='Predicted');plt.legend();plt.xscale('log');plt.yscale('log');plt.xlabel('Arithmetic intensity (FLOPs/byte, traffic model)');plt.ylabel('Measured throughput (TFLOP/s)');plt.tight_layout();plt.savefig(fig/'regimes.png',dpi=160);plt.close()
    lines=['# Фактические результаты',f'GPU: {env["gpu"]}; PyTorch {env["torch"]}; CUDA {env["cuda"]}.',f'Всего {len(df)} точек, успешно {len(ok)}, OOM {metrics["oom"]["observed"]}.','', '| Величина | Train MdAPE | Validation MdAPE | Validation P90 |','|---|---:|---:|---:|']
    for name in ['latency','memory']+(['energy'] if te else []):
        m=metrics[name];lines.append(f'| {name} | {m["train"]["median_absolute_relative_error"]:.1%} | {m["validation"]["median_absolute_relative_error"]:.1%} | {m["validation"]["p90_absolute_relative_error"]:.1%} |')
    lines+=['', 'Параметры latency: `'+json.dumps(theta)+'`.', 'Параметры energy: `'+json.dumps(te)+'`.', '', '## Обсуждение', 'Модель latency суммирует roofline-оценку каждого оператора и постоянную стоимость запуска. Для малых тензоров Python, синхронизация и запуски занимают значимую долю; для больших растёт вклад вычислений. Разные cuDNN-алгоритмы и загрузка GPU вызывают отклонения, которые три глобальных параметра не описывают.', 'Memory — аналитическая оценка живых тензоров с временными индексами MaxPool. Она не включает неизвестный workspace cuDNN и округление аллокатора. Поэтому она не гарантирует точный порог OOM; память устройства также расходуют CUDA-контекст и библиотеки. Расхождение нужно рассматривать как границу применимости, а не подгонять коэффициент памяти.', 'Energy измерена для всей одной GPU в серии прямых проходов и поделена на число проходов. Это средняя энергия при устойчивой нагрузке, а не энергия изолированного запуска. Статическое потребление не вычиталось. Короткие проходы нельзя надёжно измерить одиночным чтением мощности. NVML-счётчик предпочтительнее; запасной способ — интеграл мощности с интервалом 20 мс. Вклад других процессов и дискретность датчика ограничивают точность.', 'Коэффициенты энергии — эмпирическая декомпозиция, а не физические константы: FLOPs и bytes сильно коррелируют, поэтому нулевой коэффициент NNLS не доказывает отсутствие соответствующих затрат. Все параметры подбирались только по базовой сетке, случайные размеры/батчи не участвовали в fit.', 'FLOPs учитывают Conv, Linear, bias и GAP. Сравнения ReLU/MaxPool считаются отдельно и не включаются в floating-point arithmetic; profiler проверяет только MAC-часть Conv/Linear.']
    (out/'SUMMARY_RU.md').write_text('\n'.join(line if line.startswith('|') else '\n'+line+'\n' for line in lines))
    print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
