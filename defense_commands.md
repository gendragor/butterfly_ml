# Команды для скриншотов и защиты

Открой PowerShell в `C:\Users\dzigr\Downloads\pr3`.

## 1. Структура DVC-пайплайна

```powershell
$env:PYTHONPATH="$PWD/vendor_dvc"
python -m dvc dag
```

Скриншотировать результат со схемой `prepare → augment → train → evaluate`.

## 2. Проверка воспроизводимости

```powershell
python -m dvc status
```

Ожидаемый результат: `Data and pipelines are up to date.`

## 3. История Git

```powershell
git log --oneline --all
```

Показать коммиты разработки и финальный commit `34ec2a0`.

## 4. Метрики без повторного обучения

```powershell
python -m dvc metrics show --json
Get-Content nn_comparison.md
```

## 5. Что показать в проводнике

- `data/processed` и `data/augmented`;
- `models/baseline_nn` и `models/augmented_nn`;
- `reports/baseline_nn` и `reports/augmented_nn`;
- `models/nn` и `reports/nn` как текущие DVC-выходы.

## Важное ограничение для защиты

В проекте настроен локальный DVC-пайплайн и создан `dvc.lock`. Удалённый DVC remote и полноценная история `dvc exp apply` не настроены. Не следует утверждать обратное. GitHub содержит код, конфигурацию, DVC-файлы и сводные метрики; большие данные и checkpoint-файлы остаются локально.
