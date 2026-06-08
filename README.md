# itmo-hft-practice

Прогнозирование минутных лог-доходностей криптоактивов на данных Binance Spot с использованием Foundation Models для временных рядов.

## Аннотация проекта

Проект посвящен задаче краткосрочного прогнозирования поведения рынка криптоактивов на уровне минутных свечей. Основной фокус на вероятностном моделировании на основе квантилей, что позволяет получать интервальные оценки и использовать их в риск-ориентированных сценариях.

В качестве предметной области выбраны пары к `USDT` с высокой ликвидностью, поскольку именно они представляют практический интерес для алгоритмических стратегий и обычно имеют более устойчивую структуру данных. Целевая переменная - минутная лог-доходность на горизонтах `t+1, t+2, t+5, t+10, t+30`.

Исследование строится вокруг сравнения baslien подхода и foundation-моделей для временных рядов. Базовый контур реализован на `LightGBM` с `quantile loss`, а для внешнего сравнения используется `Chronos-2` в `zero-shot` режиме на одинаковых подвыборках.

Ключевой практический акцент сделан на воспроизводимости: подготовка данных, генерация признаков, обучение, валидация, расчет метрик и визуализации выполнены в виде единого pipeline с конфигурациями. Это позволяет повторять эксперименты, масштабировать их и корректно сопоставлять результаты между моделями.

По итогам проведенных запусков градиентный бустинг демонстрирует стабильную калибровку предсказательных интервалов (`q10-q90`) и ожидаемую динамику роста ошибок с увеличением горизонта. Сравнительный тест показал, что `Chronos-2` в текущем `zero-shot` сценарии уступает baseline по pinball/MAE и по калибровке интервалов, что формирует обоснованный план дальнейших улучшений через fine-tuning модели, а также применение фреймворка STRIDE.

## Постановка задачи

**Цель:** разработать и провалидировать воспроизводимый pipeline вероятностного прогнозирования минутных лог-доходностей криптоактивов и оценить применимость `Chronos-2` относительно `LightGBM` baseline.

**Исходные условия и ограничения:**

- Данные: `1m OHLCV` по ликвидным парам Binance Spot.
- Разбиение: train (`2024-01-01` - `2025-06-30`), valid (`2025-07-01` - `2025-12-31`).
- Горизонты: `1, 2, 5, 10, 30` минут.
- Квантили: `q10, q50, q90`.

**Критерии оценки:**

- pinball loss по квантилям;
- `MAE/RMSE` для `q50`;
- покрытие интервала `q10-q90` (целевой уровень около `0.8`);
- направление движения (`direction accuracy`) и вычислительная эффективность.

## Обзор литературы

### Foundation TSFM / LLM-подходы

| № | Authors | Title | Model | Year |
| --- | --- | --- | --- | --- |
| 1 | C. Liu, Aksu J., Liu et al. | Moirai 2.0: When Less Is More for Time Series Forecasting | Moirai 2.0 | 2025 |
| 2 | Jin, Wang, Ma et al. | Time-LLM: Time Series Forecasting by Reprogramming Large Language Models | Time-LLM | 2023 |
| 3 | Zhou, Niu, Wang et al. | One Fits All: Power General Time Series Analysis by Pretrained LM | GPT4TS | 2023 |

**Insights:**
1. **Moirai 2.0**
   - Decoder-only foundation model, обученная на большом корпусе ВР.
   - Умеет в quantile forecasting, рекурсивный multi-token prediction; заметно быстрее и компактнее предыдущей версии.
   - Использование pinball-loss для прямого прогнозирования квантилей обеспечивает более стабильную оптимизацию и более интерпретируемые интервальные прогнозы по сравнению с параметризацией смесевых распределений и NLL-оптимизацией.
2. **Time-LLM**
   - Backbone LLM (Llama/GPT-2) остается замороженным, обучаются только легкие prompter-и.
   - Используется техника Prompts-as-Prefix (PaP) для контекстуализации и «направления» трансформации.
   - Для low-latency сценариев тяжелый LLM-backbone может быть непрактичен по скорости.
3. **GPT4TS**
   - Универсальный TS-подход: одна pre-trained backbone для разных классов задач (forecasting, anomaly detection, few-shot).
   - Frozen Pretrained Transformer (FPT) сохраняет core-блоки attention/FFN без изменения, адаптируя input TS через fine-tuning.
   - Поведение attention интерпретируют как близкое к PCA-подобному извлечению структурных компонент.


### Chronos / reasoning-направление

| № | Authors | Title | Model | Year |
| --- | --- | --- | --- | --- |
| 1 | Ansari, Stella, Turkmen et al. | Chronos: Learning the Language of Time Series | Chronos | 2024 |
| 2 | Ansari, Shchur, Kuken et al. | Chronos-2: From Univariate to Universal Forecasting | Chronos-2 | 2025 |
| 3 | Ahamed, Parmar, Goyal et al. | Reasoning-Aware Training for Time Series Forecasting | STRIDE (реймворк для TSFM) | 2026 |

**Insights:**
1. **Chronos**
   - Предложен LLM-based подход к time-series forecasting через токенизацию BP (scaling + quantization) и обучение на CE-loss.
   - Решение проблемы отсутствия больших открытых datasets через генерацию синтетических ВР при помощи Data Augmentation.
   - Получено превосходство над статистическими подходами (Arima, ETS), классическими нейро-архитектурами (DLinear, N-HiTS, N-BEATS) и др.
2. **Chronos-2**
   - Переход от univariate к multivariate-covariates forecasting (в т.ч. в zero-shot режиме).
   - Добавлен механизм Group Attention для эффективного in-context learning между связанными рядами.
   - Выявлен заметный прирост в качестве по сравнению с предыдущей версией Chronos.
3. **STRIDE**
   - Идея STRIDE: встраивать reasoning не как text-токены, а через непрерывные embeddings в энкодере.
   - Механизм: reasoning-шаги дистиллируются в легковесную LLM, ее скрытые состояния проецируются в TSFM.
   - По сути plug-and-play надстройка над существующими TSFM (в т.ч. Chronos-2), а не замена им.
   - Практический смысл: улучшение точности и более интерпретируемое поведение модели.


## Описание технического решения

### 1) Data + feature pipeline

- Загрузка исторических свечей Binance через `scripts/download_binance.py`.
- Подготовка датасета и расчет признаков (`лаги`, `rolling`-статистики, производные по волатильности и объему) через `scripts/build_dataset.py`.
- Формирование таргетов для нескольких горизонтов и единый формат train/valid с контролем временной каузальности.

### 2) Baseline-модель (`LightGBM quantile`)

- Отдельные модели под каждый горизонт и квантиль (`q10, q50, q90`).
- Обучение и валидация через `scripts/train_baseline.py`.
- Логирование артефактов в `artifacts/metrics`, `artifacts/models`, `artifacts/predictions`.

### 3) Foundation baseline (`Chronos-2`, zero-shot)

- Оценка через `scripts/eval_chronos_baseline.py`.
- Протокол fair-сравнения на одинаковых точках (`BTCUSDT`, `ETHUSDT`) с агрегированием метрик в `artifacts/chronos/metrics_away`.
- Дополнительные визуализации для презентационного анализа.

## Полученные результаты

### 1) Итоги `LightGBM` на полном валидационном периоде (`TOTAL`)


| Horizon | Coverage q10-q90 | MAE (q50) | RMSE (q50) |
| ------- | ---------------- | --------- | ---------- |
| 1       | 0.8028           | 0.000791  | 0.001629   |
| 2       | 0.8084           | 0.001111  | 0.002408   |
| 5       | 0.8115           | 0.001747  | 0.003960   |
| 10      | 0.8127           | 0.002458  | 0.005431   |
| 30      | 0.8101           | 0.004232  | 0.008487   |


Ключевые выводы:

- покрытие интервалов стабильно держится около целевых `80%` на всех горизонтах;
- ошибка монотонно растет с горизонтом, что соответствует ожиданиям для минутных данных;
- `direction accuracy` по `q50` растет с горизонтом от `~0.28` (h1) до `~0.50` (h30).

### 2) Сравнение `Chronos-2 (zero-shot)` vs `LGBM` на одинаковых точках (`BTCUSDT+ETHUSDT`)


| Horizon | Pinball q50 LGBM | Pinball q50 Chronos | Ratio Chronos/LGBM | MAE q50 LGBM | MAE q50 Chronos | Coverage LGBM | Coverage Chronos |
| ------- | ---------------- | ------------------- | ------------------ | ------------ | --------------- | ------------- | ---------------- |
| 1       | 0.000232         | 0.001011            | 4.35x              | 0.000464     | 0.002022        | 0.8086        | 0.1204           |
| 2       | 0.000364         | 0.000996            | 2.74x              | 0.000727     | 0.001992        | 0.7977        | 0.1727           |
| 5       | 0.000586         | 0.001100            | 1.88x              | 0.001172     | 0.002199        | 0.7672        | 0.3171           |
| 10      | 0.000689         | 0.001281            | 1.86x              | 0.001378     | 0.002562        | 0.8009        | 0.4218           |


Вывод: в текущей `zero-shot` постановке `Chronos-2` заметно уступает baseline по качеству и калибровке интервалов, хотя относительный разрыв снижается на более длинных горизонтах.

## Дальнейшие планы по развитию

- Запустить controlled fine-tuning для `Chronos`/TSFM-подхода на доменных данных.
- Расширить контур признаков (микроструктурные и кросс-символьные фичи) и сравнить с текущим baseline.
- Ввести walk-forward валидацию и стресс-тесты по волатильным рыночным участкам.
- Добавить бэктест с учетом транзакционных издержек для оценки практической применимости прогнозов.
- Проверить reasoning-ориентированные подходы (например, идеи `STRIDE`) как надстройку к TSFM.

## Quick start

### 1) Environment

```powershell
uv venv --python 3.12
.\.venv\Scripts\Activate.ps1
uv sync
```

### 2) Download data

```powershell
uv run python scripts/download_binance.py --config configs/data.yaml
```

Optional manual symbol list:

```powershell
uv run python scripts/download_binance.py --symbols BTCUSDT,ETHUSDT,BNBUSDT
```

### 3) Build features and train/valid datasets

```powershell
uv run python scripts/build_dataset.py --config configs/features.yaml
```

### 4) Train quantile baseline

```powershell
uv run python scripts/train_baseline.py --config configs/train.yaml
```

## Notebook (ad-hoc testing)

- Open `notebooks/ad_hoc_pipeline.ipynb`.
- Set flags in the parameters cell:
  - `RUN_DOWNLOAD`
  - `RUN_BUILD`
  - `RUN_TRAIN`
  - `SMOKE_MODE`
- Start with `SMOKE_MODE=True` to validate the pipeline quickly, then switch to full run.

## Repository layout

```text
configs/   # yaml configs for data/features/train
scripts/   # runnable scripts
notebooks/ # interactive ad-hoc pipeline checks
src/       # package code
tests/     # tests
data/      # local datasets (ignored in git)
artifacts/ # models and metrics (ignored in git)
reports/   # figures/tables for defense
```

