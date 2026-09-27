# HW1 — Аналитическая модель производительности CNN

Домашняя работа 1 по курсу Efficient Models, ITMO 2026. Вывод формул стоимости CNN и проверка на одной Tesla T4 в Kaggle: 132 конфигурации, FP32, без обучения.

## Материалы работы

| Требование | Файлы |
|---|---|
| Рукописное описание формул | [Рукопись, PDF](hw1/hw1_handwritten.pdf) |
| Код расчёта функций | [equations.py](hw1/equations.py): `flops`, `memory`, `latency`, `energy` |
| Графики | [Все 8 графиков](hw1/results/figures/) и ссылки ниже |

## Графики

На графиках сопоставлены аналитические предсказания и реальные измерения. Обучающие и отложенные точки обозначены отдельно на графиках времени, памяти и энергии.

| Величина | По сетке размеров и батчей | Предсказание против измерения |
|---|---|---|
| Время | [Latency grid](hw1/results/figures/latency_grid.png) | [Latency parity](hw1/results/figures/latency_parity.png) |
| Память | [Memory grid](hw1/results/figures/memory_grid.png) | [Memory parity](hw1/results/figures/memory_parity.png) |
| Энергия | [Energy grid](hw1/results/figures/energy_grid.png) | [Energy parity](hw1/results/figures/energy_parity.png) |

Дополнительно: [проверка FLOPs через PyTorch profiler](hw1/results/figures/flops_check.png) и [эффективная производительность](hw1/results/figures/regimes.png).

## Код и воспроизведение

- [Модель CNN](hw1/models.py), [измерения](hw1/measure.py), [калибровка и построение графиков](hw1/calibrate.py).
- [Автономный Kaggle-ноутбук](hw1/kaggle_run.ipynb), [запуск на Kaggle](https://www.kaggle.com/code/doifgoox/itmo-cnn-cost-lab-hw1).
- [Исходные замеры CSV](hw1/results/measurements.csv), [коэффициенты моделей](hw1/results/theta.json), [GPU и версии ПО](hw1/results/environment.json).
- [Подробный отчёт и инструкция запуска](hw1/README.md), [печатный вывод формул](hw1/DERIVATIONS_RU.md).

## Результаты

Измерены 132 из 132 конфигураций: 63 для калибровки и 69 для независимой проверки. OOM не наблюдался; максимальный пик выделенной памяти — 5.75 GiB.

| Величина | Медианная относительная абсолютная ошибка на отложенных точках |
|---|---:|
| Время | 33.41% |
| Память | 34.95% |
| Энергия | 30.29% |

Модели приближённые: память не включает workspace cuDNN, а постоянные параметры времени не описывают все скачки производительности. Причины расхождений и ограничения рассмотрены в [обсуждении результатов](hw1/results/DISCUSSION_RU.md).
