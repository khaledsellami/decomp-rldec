import logging
import os
from typing import Dict
from concurrent import futures

import grpc
from google.protobuf.internal.well_known_types import Struct

from rldec.decompose import generate_decomposition
from rldec.model_handler import ModelHandler
from rldec.models.rldec import DecompositionRequest, Decomposition, Partition, Granularity, SelectionStrategy, \
    ModelsResponse, ModelDetails, ModelsRequest, ModelDetailsRequest, TrainedAppsRequest, TrainedAppsResponse, \
    AppDetails, RLDecApproach, AnalysisFeatures, DecompositionResponse, Status, add_RLDecServicer_to_server, RLDecServicer
from rldec.config import MODELS_PATH


SUPPORTED_LANGUAGES = ["java"]


class DecompServer(RLDecServicer):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def decompose(self, request: DecompositionRequest, context):
        app_name = request.appName
        language = request.language
        # For now only Java is supported
        if language not in SUPPORTED_LANGUAGES:
            self.logger.warning(f"The programming language '{language}' is not supported! Supported languages are "
                                f"'{SUPPORTED_LANGUAGES}'. Defaulting back to Java")
            language = "java"
        level = request.level
        # For now only Class level is supported
        if level != Granularity.CLASS:
            self.logger.warning(f"The granularity level language '{level}' is not supported! "
                                f"Defaulting back to Class level")
            level = Granularity.CLASS
        experiment_id = request.experimentID if request.HasField("experimentID") else None
        models_path = request.path if request.HasField("path") else MODELS_PATH
        num_episodes = request.numEpisodes if request.HasField("numEpisodes") else 1
        select_strategy = request.selectionStrategy if request.HasField("selectionStrategy") else SelectionStrategy.BEST
        verbose = False
        output_path = None
        decomposition = generate_decomposition(app_name, models_path, experiment_id, verbose, output_path, num_episodes,
                                               select_strategy)
        decomposition = Decomposition(name=experiment_id, appName=app_name,
                             language=language, level=Granularity.keys()[level].lower(), appRepo=models_path,
                             partitions=[Partition(name=k, classes=v) for k, v in decomposition.items()])
        response = DecompositionResponse(decomposition=decomposition, status=Status.DONE,
                                         message="Generated decomposition")
        return response

    def getModels(self, request: ModelsRequest, context):
        model_handler = ModelHandler()
        app_name = request.appName
        language = request.language
        models = model_handler.list_models(app_name, language)
        response = ModelsResponse(models=[to_model_details(model) for model in models])
        return response

    def getModelDetails(self, request: ModelDetailsRequest, context):
        model_handler = ModelHandler()
        app_name = request.appName
        language = request.language
        level = request.level
        model_name = request.experimentID
        model_details = model_handler.get_model_details(app_name, model_name, language, level)
        if model_details is None:
            raise ValueError("Model does not exist!")
        response = to_model_details(model_details)
        return response

    def getTrainedApps(self, request: TrainedAppsRequest, context):
        model_handler = ModelHandler()
        apps = model_handler.get_apps()
        # Warning: for now only Java is supported
        response = TrainedAppsResponse(appDetails=[AppDetails(appName=app, language="java") for app in apps])
        return response


def to_model_details(model: Dict) -> ModelDetails:
    approach = RLDecApproach.Value(model["approach"].upper())
    features = [AnalysisFeatures.Value(feature.upper()) for feature in model["features"]]
    hyperparams_file = Struct()
    hyperparams_file.update(model["hyperparamsFile"])
    return ModelDetails(appName=model["appName"], language=model["language"], level=model["level"],
                        experimentID=model["experimentID"], approach=approach, features=features,
                        hyperparamsFile=hyperparams_file)


def serve():
    rldec_port = os.getenv('SERVICE_RLDEC_PORT', 50150)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_RLDecServicer_to_server(DecompServer(), server)
    server.add_insecure_port(f"[::]:{rldec_port}")
    server.start()
    logging.info(f"RLDec server started, listening on {rldec_port}")
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    serve()
