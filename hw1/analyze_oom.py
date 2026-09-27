"""Plot the supplemental OOM experiment without fitting the memory equation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from equations import memory

def main():
    out=Path('results/oom')
    d=pd.read_csv(out/'measurements_oom.csv')
    s=json.loads((out/'summary.json').read_text())
    total=s['total_memory'];lo=s['largest_success_in_search'];hi=s['smallest_oom_in_search']
    predicted=s['predicted_first_oom_total']
    success=d[d.status=='OK'];failed=d[d.status=='OOM']
    available_first=np.floor((d.available_tensor_budget-4161296)/(68*512**2)).astype(int)+1
    analysis=dict(predicted_first_oom_available_min=int(available_first.min()),predicted_first_oom_available_max=int(available_first.max()),false_negatives_available=int(((d.status=='OOM') & (d.predicted_oom_available==0)).sum()),unique_batches=int(d.B.nunique()),boundary_batch_relative_gap=(predicted-hi)/hi)
    (out/'analysis.json').write_text(json.dumps(analysis,indent=2))
    bs=np.arange(max(1,int(d.B.min())-40),int(d.B.max())+41)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    ax=axes[0]
    ax.plot(bs,memory(512,bs)/2**30,label='Analytic tensor memory',color='tab:blue')
    ax.scatter(success.B,success.memory/2**30,label='Measured successful forward',color='black',zorder=3)
    ax.axhline(total/2**30,ls='--',color='gray',label='GPU total capacity')
    ax.axvspan(lo,hi,color='tab:red',alpha=.25,label='Observed adjacent OK/OOM bracket')
    ax.set(xlabel='Batch size B (S=512 pixels)',ylabel='Memory (GiB)',title='Memory model vs successful measurements')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    ax=axes[1]
    ax.step(bs,(memory(512,bs)>total).astype(int),where='mid',label='Predicted OOM: M > GPU total')
    ax.scatter(d.B,(d.status=='OOM').astype(int),marker='x',color='tab:red',label='Observed status (including repeats)',zorder=3)
    ax.axvline(int(available_first.median()),ls='--',color='tab:green',label=f'Prediction using free budget: {int(available_first.median())}')
    ax.axvline(hi,ls=':',color='tab:red',label=f'Observed first OOM in search: {hi}')
    ax.axvline(predicted,ls='--',color='tab:blue',label=f'Analytic first OOM: {predicted}')
    ax.set(xlabel='Batch size B (S=512 pixels)',ylabel='Forward outcome',yticks=[0,1],yticklabels=['OK','OOM'],ylim=(-.15,1.2),title='OOM boundary detail: measured vs predicted')
    ax.set_xlim(lo-12,predicted+12)
    ax.grid(alpha=.2);ax.legend(fontsize=8,loc='center right')
    fig.tight_layout();figpath=Path('results/figures/oom_boundary.png');figpath.parent.mkdir(exist_ok=True)
    fig.savefig(figpath,dpi=170);plt.close(fig)
    lines=['# Дополнительная проверка OOM','',
           'Это отдельный эксперимент за пределами основной сетки: S=512, B>256. Основные 132 измерения и калибровка не изменены.','',
           f'GPU: {s["gpu"]}, полная память {total/2**30:.2f} GiB. Каждый опыт выполнялся в новом процессе на cuda:0 с исходной моделью, FP32, eval(), inference_mode(), тремя прогревами и одним измеряемым forward. Все три флага benchmark/TF32 выключены. Искусственного ограничения памяти и посторонних тензоров не было.','',
           f'Поиск дал соседние точки: **B={lo} — OK; B={hi} — настоящий torch.cuda.OutOfMemoryError**. Обе точки дополнительно проверены дважды; стабильность подтверждена: {s["boundary_repeats_stable"]}.',
           f'Всего {len(d)} запусков ({d.B.nunique()} разных батчей), из них {len(failed)} с OOM.','',
           '| B | Исход | Пик успешного forward, GiB | M(S,B), GiB | Прогноз OOM по полной памяти |','|---:|---|---:|---:|---|']
    for b in sorted(d.B.unique()):
        part=d[d.B==b];state='/'.join(sorted(part.status.unique()));vals=part.memory.dropna()
        peak=f'{vals.min()/2**30:.3f}–{vals.max()/2**30:.3f}' if len(vals)>1 else (f'{vals.iloc[0]/2**30:.3f}' if len(vals) else '—')
        lines.append(f'| {b} | {state} | {peak} | {float(memory(512,b))/2**30:.3f} | {"OOM" if float(memory(512,b))>total else "OK"} |')
    lines += ['', '## Сравнение с формулой', '',
              'M(512,B)=4161296+68·512²·B байт. Коэффициенты формулы не подгонялись.',
              f'Первый предсказанный OOM по полной памяти: B=floor((GPU_total−4161296)/(68·512²))+1={predicted}. Фактическая соседняя граница поиска — {hi}. Таким образом, формула допускает слишком большой батч.',
              f'Если вместо полной памяти брать свободную память перед входом плюс уже выделенные тензоры модели, предсказанная первая OOM-точка лежит в диапазоне {available_first.min()}–{available_first.max()}; этот вариант также сохранён в CSV.',
              f'На всех пробах (включая повторы): пропущенных OOM по полной памяти — {s["false_negatives_total"]}, ложных OOM — {s["false_positives_total"]}. Это результаты адаптивного поиска, а не независимая оценка частоты ошибок.', '',
              f'Относительное смещение первого OOM по полной памяти: ({predicted}−{hi})/{hi}={100*(predicted-hi)/hi:.2f}%. Учёт свободного бюджета уменьшает смещение, но не устраняет его полностью. Это сравнение порога без подгонки формулы.', '',
              'Аналитика учитывает живые тензоры и индексы MaxPool, но не неизвестный workspace выбранных cuDNN kernels, округление/фрагментацию аллокатора и расходы вне max_memory_allocated. Поэтому OOM может возникнуть даже при M ниже полной памяти. Сообщения реальных исключений и этап отказа сохранены для каждой пробы в JSON и CSV. У неуспешного прохода полного пика памяти нет: поле memory пустое, peak_before_failure означает лишь максимум до исключения.', '',
              'Двоичный поиск использует предположение о локально монотонной вместимости. Повторы подтверждают соседнюю границу в этой сессии, но не доказывают глобальную монотонность: при других размерах cuDNN может выбирать другие алгоритмы. Измерения времени/энергии основной работы не заменяются дополнительными опытами; энергия и медианная latency здесь не измерялись.', '',
              '![Память и реальная граница OOM](../figures/oom_boundary.png)', '',
              'Воспроизведение из hw1/: `python oom_probe.py`, затем `python analyze_oom.py`.']
    (out/'README.md').write_text('\n'.join(lines))
    print(json.dumps(s,indent=2))

if __name__=='__main__':main()
