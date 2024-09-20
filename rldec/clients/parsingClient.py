import io
import logging
import os
from typing import List, Callable

import grpc
import pandas as pd

from ..models.parse import NamesRequest, Granularity, ParseRequest, Status, Format, ParserStub


class ParsingClient:
    TEMP_PATH = "./temp/"
    # PARSING_PORT = 50500
    FORMAT = Format.PARQUET
    SERVICE_NAME = os.getenv('SERVICE_PARSING', "localhost")
    PARSING_PORT = os.getenv('SERVICE_PARSING_PORT', 50500)

    def __init__(self, app: str, app_repo: str = "", language: str = "java", granularity: str = "class",
                 is_distributed: bool = False, *args, **kwargs):
        assert granularity in ["class", "method"]
        self.app_name = app
        self.app_repo = app_repo
        self.language = language
        self.granularity = granularity
        self.is_distributed = is_distributed

    def parse_all(self):
        with grpc.insecure_channel(f'{self.SERVICE_NAME}:{self.PARSING_PORT}') as channel:
            stub = ParserStub(channel)
            level = Granularity.Value(self.granularity.upper())
            request = ParseRequest(appName=self.app_name, appRepo=self.app_repo, language=self.language, level=level,
                                   isDistributed=self.is_distributed)
            response = stub.parseAll(request)
            return response.status

    def get_names(self, granularity: Granularity = None) -> List[str]:
        logging.debug(f"getting names from {self.SERVICE_NAME}:{self.PARSING_PORT}")
        with grpc.insecure_channel(f'{self.SERVICE_NAME}:{self.PARSING_PORT}') as channel:
            stub = ParserStub(channel)
            granularity = Granularity.Value(self.granularity.upper()) if granularity is None else granularity
            request = NamesRequest(appName=self.app_name, appRepo=self.app_repo, language=self.language,
                                   level=granularity, isDistributed=self.is_distributed)
            names = stub.getNames(request)
            return names.names

    def get_calls(self):
        logging.debug(f"getting calls from {self.SERVICE_NAME}:{self.PARSING_PORT}")
        with grpc.insecure_channel(f'{self.SERVICE_NAME}:{self.PARSING_PORT}') as channel:
            stub = ParserStub(channel)
            function = stub.getCalls
            return self.get_matrix(function)

    def get_interactions(self):
        with grpc.insecure_channel(f'{self.SERVICE_NAME}:{self.PARSING_PORT}') as channel:
            stub = ParserStub(channel)
            function = stub.getInteractions
            return self.get_matrix(function)

    def get_tfidf(self):
        with grpc.insecure_channel(f'{self.SERVICE_NAME}:{self.PARSING_PORT}') as channel:
            stub = ParserStub(channel)
            function = stub.getTFIDF
            return self.get_matrix(function)

    def get_word_counts(self):
        with grpc.insecure_channel(f'{self.SERVICE_NAME}:{self.PARSING_PORT}') as channel:
            stub = ParserStub(channel)
            function = stub.getWordCounts
            return self.get_matrix(function)

    def get_matrix(self, function: Callable):
        level = Granularity.Value(self.granularity.upper())
        request = ParseRequest(appName=self.app_name, appRepo=self.app_repo, language=self.language,
                               format=self.FORMAT, level=level, isDistributed=self.is_distributed)
        bytes_data = bytearray()
        for response in function(request):
            one_of = response.WhichOneof('response')
            if one_of == "metadata":
                metadata = response.metadata
                logging.debug("File transfer status: {}".format(Status.Name(metadata.status)))
            else:
                bytes_data += bytearray(response.file.content)
        df = self.load(io.BytesIO(bytes_data), format=self.FORMAT)
        return df

    def load(self, bytes, format) -> pd.DataFrame:
        if format == Format.PARQUET:
            data = pd.read_parquet(bytes)
        elif format == Format.CSV:
            data = pd.read_csv(bytes)
        elif format == Format.PICKLE:
            data = pd.read_pickle(bytes)
        elif format == Format.JSON:
            data = pd.read_json(bytes)
        else:
            raise ValueError("Unrecognized data_format {}!".format(format))
        return data
