from collections import namedtuple
import glob
import json
import logging
import os
import re

import numpy as np
import torch_neuron
from sagemaker_inference import content_types, decoder, default_inference_handler, encoder

# Original implementation from SM 1P Pytorch inference container
# https://github.com/aws/sagemaker-pytorch-inference-toolkit/blob/master/src/sagemaker_pytorch_serving_container/default_inference_handler.py

class ModelHandler(default_inference_handler.DefaultInferenceHandler):
    def default_model_fn(self, model_dir):
        model_files = []
        for f in os.listdir(model_dir):
            if os.path.isfile(f):
                name, ext = os.path.splitext(f)
                if ext == ".pt" or ext == ".pth":
                    model_files.append(f)
        if len(model_files) != 1:
            raise ValueError("Exactly one .pth or .pt file is required for PyTorch models: {}".format(model_files))
        return torch.jit.load(model_files[0])

    # The Pydocs are def wrong for this
    def default_input_fn(self, input_data, content_type):
        """A default input_fn that can handle JSON, CSV and NPZ formats.
        Args:
            input_data: the request payload serialized in the content_type format
            content_type: the request content_type
        Returns: input_data deserialized into torch.FloatTensor or torch.cuda.FloatTensor,
            depending if cuda is available.
        """
        np_array = decoder.decode(input_data, content_type)
        # tensor = torch.FloatTensor(np_array) if content_type in content_types.UTF8_TYPES else torch.from_numpy(np_array)
        # return tensor.to(device)
        return torch.tensor(np_array)

    def default_predict_fn(self, data, model):
        """A default predict_fn for PyTorch. Calls a model on data deserialized in input_fn.
        Runs prediction on GPU if cuda is available.
        Args:
            data: input data (torch.Tensor) for prediction deserialized by input_fn
            model: PyTorch model loaded in memory by model_fn
        Returns: a prediction
        """
        return model(data)

    def default_output_fn(self, prediction, accept):
        """A default output_fn for PyTorch. Serializes predictions from predict_fn to JSON, CSV or NPY format.
        Args:
            prediction: a prediction result from predict_fn
            accept: type which the output data needs to be serialized
        Returns: output data serialized
        """
        if type(prediction) == torch.Tensor:
            prediction = prediction.detach().cpu().numpy().tolist()
        encoded_prediction = encoder.encode(prediction, accept)
        if accept == content_types.CSV:
            encoded_prediction = encoded_prediction.encode("utf-8")

        return encoded_prediction

_service = ModelHandler()

def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)

