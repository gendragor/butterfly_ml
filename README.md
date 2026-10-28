# Butterfly MLOps

Практическая работа №3: инженерия машинного обучения для классификации 100 видов бабочек.

## Пайплайн

```text
prepare → augment → train → evaluate
```

- `prepare` создаёт воспроизводимую выборку из исходного датасета;
- `augment` сохраняет дополнительные изображения с поворотом, отражением и изменением яркости;
- `train` одним запуском обучает SimpleCNN, MobileNetV3 Small и ResNet18;
- `evaluate` оценивает сохранённые checkpoint-файлы на неизменённом test.

DVC описывает стадии в `dvc.yaml`, а параметры находятся в `params.yaml`. Большие данные и модели исключены из Git.

## Запуск

```powershell
$env:PYTHONPATH="$PWD/vendor_nn;$PWD/vendor_dvc"
python src/prepare.py --config params.yaml
python src/augment.py --config params.yaml
python src/train_nn.py --config params.yaml
python src/evaluate.py --config params.yaml
```

Или запуск пайплайна DVC:

```powershell
$env:PYTHONPATH="$PWD/vendor_nn;$PWD/vendor_dvc"
python -m dvc repro
```

Измените `params.yaml`, например `training.epochs`, `training.learning_rate` или `training.use_augmentation`, затем повторите запуск для нового эксперимента.

## Проверенный CPU-запуск

Использовано до 40 train-изображений на класс, 5 эпох и seed 42. Лучшая модель — ResNet18: test accuracy 0.546. Полная таблица находится в `nn_results.md`.
