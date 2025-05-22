from metaflow import FlowSpec, step


class XGBoostTrainingFlow(FlowSpec):
    stage = "staging"
    project = "mlops-study"
    model_dependencies_group_name = "model"

    @step
    def start(self):
        """Generate synthetic data."""
        from sklearn.datasets import make_classification

        self.X, self.y = make_classification(
            n_samples=200, n_features=5, n_informative=3, n_redundant=0, random_state=42
        )
        self.next(self.split_data)

    @step
    def split_data(self):
        """Split data into train and test sets."""
        from sklearn.model_selection import train_test_split
        import pandas as pd

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.25, random_state=42
        )
        self.feature_names = [f"feature_{i}" for i in range(self.X_train.shape[1])]
        self.X_train_df = pd.DataFrame(self.X_train, columns=self.feature_names)
        self.X_test_df = pd.DataFrame(self.X_test, columns=self.feature_names)
        self.next(self.train_model)

    @step
    def train_model(self):
        """Train XGBoost model and log with MLflow."""

        import mlflow
        import xgboost as xgb
        from sklearn.metrics import accuracy_score
        import pandas as pd
        import os
        from mlflow.models.signature import infer_signature
        import subprocess

        def get_uv_package_versions():
            """Get package versions using UV."""
            try:
                # Use UV to get the resolved versions
                result = subprocess.run(
                    [
                        "uv",
                        "pip",
                        "compile",
                        "--no-header",
                        "--no-emit-index-url",
                        "--group",
                        self.model_dependencies_group_name,
                    ],
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0:
                    # Parse the output to get package versions
                    requirements = result.stdout.strip().split("\n")
                    return [req for req in requirements if req]
                else:
                    print(f"Warning: UV command failed: {result.stderr}")
                    return None
            except Exception as e:
                print(f"Warning: Failed to get package versions using UV: {str(e)}")
                return {}

        # Get package versions using UV
        pip_reqs = get_uv_package_versions()

        # Set MLflow tracking URI and S3 endpoint if needed
        os.environ.setdefault(
            "MLFLOW_TRACKING_URI", "http://mlflow.mlops-study.svc.cluster.local:5000"
        )
        os.environ.setdefault(
            "MLFLOW_S3_ENDPOINT_URL",
            "http://mlflow-minio.mlops-study.svc.cluster.local:9000",
        )
        os.environ.setdefault("AWS_ACCESS_KEY_ID", "mlflow_minio_user")
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "mlflow_minio_password")

        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(self.X_train_df, label=self.y_train)
        dtest = xgb.DMatrix(self.X_test_df, label=self.y_test)

        experiment_name = "XGBoost_Metaflow_Example"
        registered_model_name = "xgboost-metaflow-model"

        try:
            experiment = mlflow.set_experiment(experiment_name)
            experiment_id = experiment.experiment_id
        except Exception as e:
            print(f"Error {e}")
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name="Metaflow XGBoost Run",
            tags={"stage": self.stage, "project": self.project},
        ):
            # Log datasets
            train_df = self.X_train_df.copy()
            train_df["target"] = self.y_train
            test_df = self.X_test_df.copy()
            test_df["target"] = self.y_test

            train_dataset = mlflow.data.from_pandas(
                train_df,
                source="synthetic_data",
                name="training_data",
                targets="target",
            )
            test_dataset = mlflow.data.from_pandas(
                test_df, source="synthetic_data", name="test_data", targets="target"
            )

            # Log dataset metadata
            mlflow.log_input(train_dataset, context="training")
            mlflow.log_input(test_dataset, context="testing")

            # Log dataset statistics
            mlflow.log_metric("train_samples", len(self.X_train_df))
            mlflow.log_metric("test_samples", len(self.X_test_df))
            mlflow.log_metric("n_features", self.X_train_df.shape[1])

            params = {
                "objective": "binary:logistic",
                "max_depth": 2,
                "eta": 0.1,
                "eval_metric": "logloss",
            }
            num_boost_round = 20
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=[(dtest, "validation")],
                early_stopping_rounds=5,
                verbose_eval=False,
            )
            y_pred_proba = model.predict(dtest)
            y_pred = (y_pred_proba > 0.5).astype(int)
            self.accuracy = accuracy_score(self.y_test, y_pred)

            mlflow.log_params(params)
            mlflow.log_metric("accuracy", self.accuracy)
            mlflow.log_metric("num_boost_round", num_boost_round)

            signature = infer_signature(
                self.X_test_df, pd.Series(y_pred_proba, name="prediction_score")
            )

            registered_model = mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path="xgboost_model",
                signature=signature,
                registered_model_name=registered_model_name,
                input_example=self.X_test_df.head(5),
                pip_requirements=pip_reqs,
            )

            # Set an alias for the model version
            client = mlflow.tracking.MlflowClient()
            client.set_registered_model_alias(
                name=registered_model_name,
                alias=self.stage,
                version=registered_model.registered_model_version,
            )

            print(
                f"Model '{registered_model_name}' has been "
                f"registered with alias '{self.stage}'."
            )
        self.next(self.end)

    @step
    def end(self):
        print(f"Flow completed. Accuracy: {self.accuracy:.4f}")


if __name__ == "__main__":
    XGBoostTrainingFlow()
