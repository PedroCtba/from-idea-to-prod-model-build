import json
import sagemaker
import boto3
from time import gmtime, strftime
from sagemaker.estimator import Estimator
from sagemaker.model_metrics import (
    MetricsSource, 
    ModelMetrics,  
    FileSource
)

def register(
    training_job_name,
    model_package_group_name,
    model_approval_status,
    evaluation_result,
    output_s3_prefix,
    model_statistics_s3_path=None,
    model_constraints_s3_path=None,
    model_data_statistics_s3_path=None,
    model_data_constraints_s3_path=None,
    pipeline_run_id=None,
    run_id=None,
):
    try:
        evaluation_result_path = f"evaluation.json"
        with open(evaluation_result_path, "w") as f:
            f.write(json.dumps(evaluation_result))
            
        estimator = Estimator.attach(training_job_name)
        
        model_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri=model_statistics_s3_path,
                content_type="application/json",
            ) if model_statistics_s3_path else None,
            model_constraints=MetricsSource(
                s3_uri=model_constraints_s3_path,
                content_type="application/json",
            ) if model_constraints_s3_path else None,
            model_data_statistics=MetricsSource(
                s3_uri=model_data_statistics_s3_path,
                content_type="application/json",
            ) if model_data_statistics_s3_path else None,
            model_data_constraints=MetricsSource(
                s3_uri=model_data_constraints_s3_path,
                content_type="application/json",
            ) if model_data_constraints_s3_path else None,
        )
    
        model_package = estimator.register(
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=["ml.m5.xlarge", "ml.m5.large"],
            transform_instances=["ml.m5.xlarge", "ml.m5.large"],
            model_package_group_name=model_package_group_name,
            approval_status=model_approval_status,
            model_metrics=model_metrics,
            model_name="from-idea-to-prod-pipeline-model",
            domain="MACHINE_LEARNING",
            task="CLASSIFICATION", 
        )

        return {
            "model_package_arn":model_package.model_package_arn,
            "model_package_group_name":model_package_group_name,
        }

    except Exception as e:
        print(f"Exception in processing script: {e}")
        raise e
