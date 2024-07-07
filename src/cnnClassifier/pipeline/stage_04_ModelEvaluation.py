from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_evaluation_com4 import Evaluation
from cnnClassifier import logger



STAGE_NAME = "Evaluation stage"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.log_into_mlflow()