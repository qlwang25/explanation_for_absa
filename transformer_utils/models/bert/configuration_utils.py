# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import (absolute_import, division, print_function, unicode_literals)

import copy
import json
import logging
import os
from io import open
from typing import Any, Dict, List, Optional, Tuple, Union


logger = logging.getLogger(__name__)
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"

class PretrainedConfig(object):
    attribute_map: Dict[str, str] = {}
    
    def __setattr__(self, key, value):
        if key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        super().__setattr__(key, value)

    def __getattribute__(self, key):
        if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        return super().__getattribute__(key)

    def __init__(self, **kwargs):
        self.finetuning_task = kwargs.pop('finetuning_task', None)
        self.num_labels = kwargs.pop('num_labels', 2)
        self.output_attentions = kwargs.pop('output_attentions', False)
        self.output_hidden_states = kwargs.pop('output_hidden_states', False)
        self.output_past = kwargs.pop('output_past', True)  # Not used by all models
        self.torchscript = kwargs.pop('torchscript', False)  # Only used by PyTorch models
        self.use_bfloat16 = kwargs.pop('use_bfloat16', False)
        self.pruned_heads = kwargs.pop('pruned_heads', {})
        self.is_decoder = kwargs.pop('is_decoder', False)

    def save_pretrained(self, save_directory):
        assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"
        output_config_file = os.path.join(save_directory, CONFIG_NAME)

        self.to_json_file(output_config_file)
        logger.info("Configuration saved in {}".format(output_config_file))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        cache_dir = kwargs.pop('cache_dir', None)
        force_download = kwargs.pop('force_download', False)
        proxies = kwargs.pop('proxies', None)
        return_unused_kwargs = kwargs.pop('return_unused_kwargs', False)

        config_file = pretrained_model_name_or_path
        if os.path.isdir(config_file):
            config_file = os.path.join(config_file, CONFIG_NAME)

        # redirect to the cache, if necessary
        try:
            # resolved_config_file = cached_path(config_file, cache_dir=cache_dir, force_download=force_download, proxies=proxies)
            resolved_config_file = config_file
        except EnvironmentError:
            msg = "Couldn't reach server at '{}' to download pretrained model configuration file.".format(config_file)
            raise EnvironmentError(msg)

        if resolved_config_file == config_file:
            logger.info("loading configuration file {}".format(config_file))
        else:
            logger.info("loading configuration file {} from cache at {}".format(config_file, resolved_config_file))

        config = cls.from_json_file(resolved_config_file)
        if hasattr(config, 'pruned_heads'):
            config.pruned_heads = dict((int(key), value) for key, value in config.pruned_heads.items())

        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        # logger.info("Model config %s", str(config))
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def from_dict(cls, json_object):
        config = cls(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            setattr(config, key, value)
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())

    def update(self, config_dict):
        """
        Updates attributes of this class with attributes from `config_dict`.
        Args:
            config_dict (`Dict[str, Any]`): Dictionary of attributes that should be updated for this class.
        """
        for key, value in config_dict.items():
            setattr(self, key, value)
