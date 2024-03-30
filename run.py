from package.feature.data_processing import get_feature_dataframe
from package.ml_training.preprocessing_pipeline import get_pipeline
from package.utils.utils import get_performance_plots, set_or_create_experiment, get_classification_metrics, register_model_with_client
from package.ml_training.retrieval import get_train_test_score_set
from package.ml_training.training import train_model
import mlflow 

if __name__ == "__main__":
    experiment_name = "house_pricing_classifer"
    run_name = "training_classifier"
    model_name = "registered_model"
    artifact_path = "model"
    df = get_feature_dataframe()
    # print(df.head())
    X_train, X_test, X_score, y_train, y_test, y_score = get_train_test_score_set(df)
    features = [f for f in X_train.columns if f not in ["id","target","MedHouseVal"]]
    pipeline = get_pipeline(numerical_features=features,categorical_features=[])
    experiment_id = set_or_create_experiment(experiment_name=experiment_name)
    run_id, model = train_model(pipeline=pipeline,run_name=run_name,model_name=model_name,x=X_train,y=y_train)
    y_pred = model.predict(X_test)
    performance_plots = get_performance_plots(y_true=y_test,y_pred=y_pred,prefix="test")
    classification_metrics = get_classification_metrics(y_true=y_test,y_pred=y_pred,prefix="test")
    # register_model_with_client(model_name,run_id,artifact_path)
    # log performance metrics
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(classification_metrics)
        mlflow.log_params(model[-1].get_params())
        mlflow.set_tags({"type":"classifier"})
        mlflow.set_tag("mlflow.note.content","This is a classifier for the house pricing dataset")
        for plot_name, fig in performance_plots.items():
            mlflow.log_figure(fig, plot_name+".png")