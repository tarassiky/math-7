# Языковая модель на Elman RNN — лабораторная работа

Словарная модель следующего слова на русскоязычном корпусе.
Два ноутбука:

- `01_elman_rnn_language_model_example.ipynb` — полный пример на
  **Gazeta** (`IlyaGusev/gazeta@v1.0`, поле `text`).
- `02_elman_rnn_language_model_assignment.ipynb` — самостоятельная
  работа на **RuHeritage-Corpus**
  (`maxzt/RuHeritage-Corpus@b2d01eaad3ddb119fe10c4cb17a86e47415a77cb`,
  каталог `data`, поле `text`).

## Запуск

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python \
    --index-url https://download.pytorch.org/whl/cpu torch==2.13.0
uv pip install --python .venv/bin/python \
    datasets matplotlib jupyter numpy tqdm
.venv/bin/jupyter lab
```

Откройте первый ноутбук, выполните **Run All** — это обучающий
маршрут. Затем откройте второй, выполните сверху вниз и заполните
пропущенные ячейки (см. коммиты / README).

## Что внутри

- **Этап 1.** Константы, seed, устройство.
- **Этап 2.** Потоковое чтение корпуса, нормализация, train/val split.
- **Этап 3.** Словарь только по обучающим документам (без утечки).
- **Этап 4.** Сдвиг на одну позицию, Dataset и `collate_fn` с PAD.
- **Этап 5.** Ручная Elman RNN: `Embedding(V,128)` → 2 слоя
  `tanh(W_x x + U_h h + b_h)` → `Linear(128, V)`.
- **Этап 6.** Обучение по сдвинутым целям, маска PAD,
  ранняя остановка.
- **Этап 7.** Валидационная перплексия, график, top-k и
  жадная генерация.
- **Этап 8.** Самопроверка.

## Что мы обещаем проверить

| Проверка | Ожидание |
|---|---|
| `input_ids.shape`, `target_ids.shape` | `(B, T)` с `T ≤ 32` |
| эмбеддинги / скрытые состояния | `(B, T, 128)` |
| логиты | `(B, T, V)` |
| PAD (индекс 0) | исключён из средней кросс-энтропии |
| потери и перплексия | конечные положительные |
| `topk` | индексы из `itos`, вероятности конечны |
| `greedy_generate` | останавливается на EOS или лимите |

## Пересборка ноутбуков

```bash
python build_notebook.py
```