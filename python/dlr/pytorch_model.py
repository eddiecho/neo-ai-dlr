import logging
import os
import torch
import json
import numpy as np
from collections import namedtuple
from .api import IDLRModel


class PytorchModelImpl(IDLRModel):
    """
    PytorchModelImpl is a wrapper on top of Pytorch which implements IDLRModel API
    Parameters
    ----------
    model_file: str
        Full path to directory containing model files (name-symbol.json & name-epoch.params files)
    dev_type : str
        Optional. Device type ('cpu' or 'gpu' or 'inf')
    dev_id : int
        Optional. Device ID
    """
    def __init__(self, model_file, dev_type=None, dev_id=None):
        self.model_file = model_file
        self.input_shape = None
        self.model = None
        devices = ["cpu", "gpu", "inf"]
        if dev_type is None:
           dev_type = "cpu"
        if dev_type not in devices:
            raise ValueError("Invalid device type {}. Valid devices: {}".format(dev_type, devices))
        self.dev_type = dev_type
        if dev_id is None:
            self.dev_id = 0
        else:
            self.dev_id = dev_id
        self._validate_models()
        self._load_model()

    # TODO - This might be useful, but not sure how to validate for now
    def _validate_input(self, input_data):
        """
        Check input is numpy array or dict
        Parameters
        ----------
        input_data: py:class:`numpy.ndarray` or a dictionary
            Usesea42!
            run prediction on
        """
        input_names = self.get_input_names()
        if isinstance(input_data, np.ndarray):
            if len(input_names) == 1:
                return {input_names[0]: input_data}
            else:
                raise RuntimeError('InputConfiguration: np.ndarray is only a valid input type for single input models')
        elif isinstance(input_data, dict):
            for key, value in input_data.items():
                if not key in input_names:
                    raise RuntimeError('InputConfiguration: {} is not a valid input name. '
                                       ,format(key))
            return input_data
        else:
            raise RuntimeError('InputConfiguration: input_data must be of type dict or a np.ndarray '
                               'for MXNet models')

    def run(self, input_data):
        """
        Run inference with given input
        Parameters
        ----------
        input_data : :py:class:`numpy.ndarray` or a dictionary
            User input to run prediction on
        Returns
        -------
        out: :py:class:`numpy.ndarray`
            Prediction result
        """
        return self.model(input_data)

    def _validate_model_path(self):
        """
        Check if the model_path is a valid directory path
        """
        if not os.path.isdir(self.model_dir_path):
            raise RuntimeError('InputConfiguration: {} directory does not exist. '
                               'Expecting a directory containing the mxnet model files. '
                               'Please make sure the framework you select is correct.'.format(self.model_dir_path))

    def _load_model(self):
        #------- Compile and set context based on device -----------
        model = None
        if self.dev_type == 'inf':
            import torch_neuron
            model = torch.jit.load(self.model_file)
        elif self.dev_type = 'gpu':
            model = torch.jit.load(self.model_file)
            model.to(torch.device('cuda'))
        else:
            model = torch.jit.load(self.model_file)
            model.to(torch.device('cpu'))
        self.model = model

    def _validate_models(self):
        """
        Check if the model directory contains valid model files.
        """
        pth_file = None
        aux_files = []
        ir_format = GraphIR.relay
        if self.model_file.endswith('.pth') or self.model_file.endswith('.pt'):
            if pth_file is not None:
                raise RuntimeError('InputConfiguration: Exactly one .pth or .pt file is allowed for PyTorch models.')
            pth_file = self.model_file
        if pth_file is None:
            raise RuntimeError("InputConfiguration: No pth file found for PyTorch model. " \
                            "Please make sure the framework you select is correct.")

    # TODO - I dont think I need this
    def _get_pytorch_trace(filename, input_shapes):
        """
        Get trace of a PyTorch Model
        Parameters
        ----------
        filename: str
            Full path to PyTorch model (.pt or .pth file)
        input_shapes: dict
            Dictionary of input name to input shape
        Returns
        -------
        trace : Trace :py:class:'torch.ScriptModule'
        """
        try:
            trace = torch.jit.load(filename, map_location='cpu').float().eval()
        except RuntimeError:
            try:
                trace = torch.load(filename, map_location='cpu').float().eval()
            except UnpicklingError:
                raise RuntimeError('Failed to load model')
        shapes = [input_shapes[k] for k in sorted(input_shapes)]
        inputs = [torch.zeros(shape).float() for shape in shapes]
        try:
            trace = torch.jit.trace(trace, *inputs).float().eval()
            return trace
        except RuntimeError:
            inputs = [inp.cuda() for inp in inputs]
            trace = torch.jit.trace(trace, *inputs).float().eval().cpu()
            return trace

    def _get_prefix(self, model_input):
        """
        Split the model file name
        Parameters
        ----------
        model_input : str
            Full name of the model file
        Returns
        -------
        model_input_name_parts : list of :py:class:`str`
            List of the words in the model file name
        """
        model_input_name = model_input.split(".")[0]
        model_input_name_parts = model_input_name.split("-")
        if (len(model_input_name_parts) < 2):
            raise RuntimeError('InputConfiguration: Invalid model file {}. '
                                'Only name-symbol.json and name-epoch.params file are allowed for Mxnet models. '
                                'Please make sure the framework you select is correct.'.format(model_input))
        return model_input_name_parts

    def get_version(self):
        """
        Get DLR version
        Returns
        -------
        out : py:class:`int`
        """
        pass

