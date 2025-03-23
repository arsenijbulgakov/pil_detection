## Окружение 

```
conda env list
conda activate torch_env
```

## Запуск обучения

```
python train.py model.output_model_path="deberta3base_512.lr_7e-5.n_epochs_1"  output_dir=test_logs training.report_to=mlflow mlflow.run_name="deberta3base_512.lr_7e-5.n_epochs_1" training.learning_rate=7e-5 training.num_train_epochs=1
```

## Запуск инференса

```
python inference.py model_path="./deberta3base_512.lr_7e-5.n_epochs_1" predictions_path=submission_v4_hydra.csv
```

## Запуск mlflow сервера


```
mlflow server --host 0.0.0.0 --port 19878 --backend-store-uri ./mlruns
```