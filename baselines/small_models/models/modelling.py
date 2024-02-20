import os
from typing import Optional, Union

import torch
from transformers import PreTrainedModel, PretrainedConfig, AutoModelForCausalLM, AutoConfig
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast


class FormulaLLM(PreTrainedModel):
    def __init__(self, config: PretrainedConfig, tokenizer, formula_token_id, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.tokenizer = tokenizer
        self.formula_token_id = formula_token_id

        self.hidden_states_linear = nn.Linear(self.model_config.hidden_size, self.model_config.hidden_size)

        self.criterion = nn.CrossEntropyLoss()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], init_formula_embs=False, *model_args, **kwargs):
        if init_formula_embs:
            cls.model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            cls.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, device_map="auto", trust_remote_code=True)
            cls.formula_embs = nn.Embedding(kwargs.get('n_formula'), cls.model_config.hidden_size)
        else:
            raise NotImplementedError

        cls.__init__(PretrainedConfig(), tokenizer=kwargs.get('tokenizer'), formula_token_id=kwargs.get('formula_token_id'))
        return cls

    def forward(self, input_ids, attention_mask, labels, formula_labels):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             labels=labels,
                             output_hidden_states=True,
                             return_dict=True)

        last_hidden_state = self.hidden_states_linear(outputs.hidden_states[-1])
        indices = torch.nonzero(labels == self.formula_token_id)
        formula_token_hidden_states = last_hidden_state[indices[:, 0], indices[:, 1], :]

        assert formula_token_hidden_states.size(0) == formula_labels.size(0)
        formula_scores = torch.mm(formula_token_hidden_states, self.formula_embs.weight.t())

        formula_loss = self.criterion(formula_scores, formula_labels)

        return {'loss': outputs.loss + formula_loss}
