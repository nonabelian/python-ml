# Batch Pipeline

```
mlflow server --file-store tracking_server/file_store --default-artifact-root tracking_server/artifacts/ -h 0.0.0.0
```

```
$ MLFLOW_TRACKING_URI=http://localhost:5000 python train.py
```

