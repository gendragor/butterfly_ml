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

## 5. Переключение между готовыми экспериментами без обучения

```powershell
python -m dvc exp list
python -m dvc exp diff augmented-final baseline-final --json
python -m dvc exp apply baseline-final
Get-Content reports/nn/test_metrics.json
python -m dvc exp apply augmented-final
Get-Content reports/nn/test_metrics.json
```

Ожидаемые значения ResNet18: `0.546` для `baseline-final` и `0.606` для `augmented-final`. После `dvc exp apply` checkpoint-файлы в `models/nn` восстанавливаются из DVC cache; повторное обучение не запускается.

## 6. Что показать в проводнике

- `data/processed` и `data/augmented`;
- `models/baseline_nn` и `models/augmented_nn`;
- `reports/baseline_nn` и `reports/augmented_nn`;
- `models/nn` и `reports/nn` как текущие DVC-выходы.

## Важное ограничение для защиты

В проекте настроен локальный DVC-пайплайн и создан `dvc.lock`. Эксперименты `baseline-final` и `augmented-final` сохранены локально в DVC и проверены через `dvc exp diff` и `dvc exp apply` без переобучения. Удалённый DVC remote не подключён, поэтому `dvc push/pull` через облако не демонстрируются. GitHub содержит код, конфигурацию, DVC-файлы и сводные метрики; большие данные и checkpoint-файлы остаются локально.
