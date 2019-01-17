import os
import pickle
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import mlflow
import mlflow.sklearn


ARTIFACT_ROOT = os.path.abspath(os.path.join('tracking_server',
                                             'artifacts',
                                             'iris_knn'))


if __name__ == '__main__':
    experiment_name = 'iris_knn' + '|' + datetime.utcnow().isoformat()
    experiment_id = mlflow.create_experiment(artifact_location=ARTIFACT_ROOT,
                                             name=experiment_name)

    with mlflow.start_run(experiment_id=experiment_id):
        data = load_iris()
        X = data['data']
        y = data['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        print(knn.score(X_test, y_test))

        mlflow.log_metric("Accuracy", knn.score(X_test, y_test))

        y_pred = knn.predict(X_test)
        print(confusion_matrix(y_test, y_pred))

        mlflow.sklearn.log_model(knn, 'knn.pkl')
