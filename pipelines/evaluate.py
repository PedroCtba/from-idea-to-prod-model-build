import numpy as np
import pandas as pd
import xgboost as xgb
from time import gmtime, strftime
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import tarfile
import pickle as pkl
import boto3

# helper function to load XGBoost model into xgboost.Booster
def load_model(model_data_s3_uri):
    model_file = "./xgboost-model.tar.gz"
    bucket, key = model_data_s3_uri.replace("s3://", "").split("/", 1)
    boto3.client("s3").download_file(bucket, key, model_file)
    
    with tarfile.open(model_file, "r:gz") as t:
        t.extractall(path=".")
    
    # Load model
    model = xgb.Booster()
    model.load_model("xgboost-model")

    return model

def plot_roc_curve(fpr, tpr):
    fn = "roc-curve.png"
    fig = plt.figure(figsize=(6, 4))
    
    # Plot the diagonal 50% line
    plt.plot([0, 1], [0, 1], 'k--')
    
    # Plot the FPR and TPR achieved by our model
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig(fn)

    return fn
  
def evaluate(
    test_x_data_s3_path,
    test_y_data_s3_path,
    model_s3_path,
    output_s3_prefix,
    run_id=None,
):
    try:        
        # Read test data
        X_test = xgb.DMatrix(pd.read_csv(test_x_data_s3_path, header=None).values)
        y_test = pd.read_csv(test_y_data_s3_path, header=None).to_numpy()
    
        # Run predictions
        probability = load_model(model_s3_path).predict(X_test)
    
        # Evaluate predictions
        fpr, tpr, thresholds = roc_curve(y_test, probability)
        auc_score = auc(fpr, tpr)
        eval_result = {"evaluation_result": {
            "classification_metrics": {
                "auc_score": {
                    "value": auc_score,
                },
            },
        }}
        
        prediction_baseline_s3_path = f"{output_s3_prefix}/prediction_baseline/prediction_baseline.csv"
    
        # Save prediction baseline file - we need it later for the model quality monitoring
        pd.DataFrame({"prediction":np.array(np.round(probability), dtype=int),
                      "probability":probability,
                      "label":y_test.squeeze()}
                    ).to_csv(prediction_baseline_s3_path, index=False, header=True)
        
        return {
            **eval_result,
            "prediction_baseline_data":prediction_baseline_s3_path,
        }
            
    except Exception as e:
        print(f"Exception in processing script: {e}")
        raise e