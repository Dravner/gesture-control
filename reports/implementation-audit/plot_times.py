"""Render measured processing distributions; never labels them full input latency."""
from pathlib import Path
import os,json,numpy as np
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/gesture-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parent/'vision-probe'
public=json.loads((out/'results.json').read_text());native=json.loads((out/'live-camera.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(11,4))
for method,title in [('mediapipe','MediaPipe'),('vision','Apple Vision')]:
    x=np.sort([r['route_ms'] for r in public['measurements'] if r['method']==method and not r['warmup']]);axes[0].plot(x,np.arange(1,len(x)+1)/len(x),label=f'{title}, n={len(x)}')
axes[0].set_title('Одинаковые публичные кадры · без GUI/OS');axes[0].legend();axes[0].set_xlabel('Маршрут извлечения точек, мс')
for key,title in [('compute_ms','Обработка + аннотация/копии'),('consume_age_ms','Возраст числового результата')]:
    x=np.sort([r[key] for r in native['packets']]);axes[1].plot(x,np.arange(1,len(x)+1)/len(x),label=title)
axes[1].set_title('Vision · камера1920×1080 · руки нет');axes[1].legend(fontsize=8);axes[1].set_xlabel('Время, мс · захват после read()')
for ax in axes:ax.set_ylabel('Доля отсчётов ≤ времени');ax.grid(alpha=.25);ax.set_ylim(0,1.02)
fig.text(.5,.01,'Разные входы двух панелей. Не измеряется задержка движения → экран/ОС и точность жестов.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.06,1,1]);fig.savefig(out/'processing-cdf.png',dpi=170);fig.savefig(out/'processing-cdf.svg');plt.close(fig)
