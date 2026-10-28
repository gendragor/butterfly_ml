# План скриншотов для отчёта

1. GitHub: репозиторий `gendragor/butterfly_ml`, ветка `main`, файлы проекта.
2. `params.yaml`: seed, image_size, batch_size, epochs, patience, learning_rate и три модели.
3. `dvc.yaml` в редакторе: стадии `prepare`, `augment`, `train`, `evaluate`.
4. Терминал с командой `dvc dag` и схемой четырёх стадий.
5. Проводник: `data/processed` и `data/augmented`; в одном классе видны исходные и `_aug.jpg` файлы.
6. Папка `models/nn`: `simplecnn.pt`, `mobilenet_v3_small.pt`, `resnet18.pt`.
7. Терминал с выводом обучения: строки JSON с `epoch`, `train_accuracy`, `valid_accuracy`.
8. `reports/nn/test_metrics.json` и `nn_comparison.md` со сводной таблицей без/с аугментацией.
9. Терминал после успешного `dvc repro`: `Stage ...` и сообщение об обновлении `dvc.lock`.
10. GitHub commit history: отдельный коммит нейросетевого пайплайна и DVC.

Подписи к рисункам можно сделать так: «Рисунок N — ...». Не включать скриншоты с чужим отчётом; использовать только собственный проект и собственные результаты.
