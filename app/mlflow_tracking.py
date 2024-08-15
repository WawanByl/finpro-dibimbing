import mlflow
import mlflow.tensorflow

def log_metrics(metrics, stage):
    with mlflow.start_run() as run:
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(f"{stage}_{metric_name}", metric_value)
        mlflow.tensorflow.log_model(model, "model")
