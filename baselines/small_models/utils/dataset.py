import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict

from filelock import FileLock
from tqdm import tqdm
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    input: str
    output: str
    id: str
    original_data: Dict


@dataclass
class InputFeatureTest:
    input_ids: List[int]
    attention_mask: List[int]
    id: Optional[str] = None
    original_data: Optional[Dict] = None


@dataclass
class InputFeature:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]


class ProcessorFormulaLLM:

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_examples(self, tokenizer, data_dir, data_type):
        logger.info("LOOKING AT {} train".format(data_dir))
        _examples = self._create_examples(f'{data_dir}/{data_type}.json', data_type)
        return _examples

    def _read_json(self, datafile):
        datas = json.load(open(datafile, 'r', encoding='utf-8'))
        return datas

    def _create_examples(self, datafile, data_type) -> List[InputExample]:
        examples = []
        raw_datas = self._read_json(datafile)
        for data in raw_datas:
            input = f"{data['instruction']}\n\n{data['input']}" if 'instruction' in data else data['input']
            output = data['output']
            examples.append(InputExample(input=input,
                                         output=output,
                                         id=data['original_data']['id'] if data_type != 'train' else None,
                                         original_data=data['original_data'] if data_type != 'train' else None, ))
        return examples


class DatasetFormulaLM:
    features: List[InputFeature]

    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int, data_dir, data_type='train'):
        super().__init__()
        self.processor = ProcessorFormulaLLM()

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        with FileLock(f'.tmp_filelock.lock'):
            logger.info(f"Creating features from dataset file at {data_dir}")
            if data_type == 'train':
                examples = self.processor.get_examples(tokenizer, data_dir, data_type='train')
                self.features = convert_examples_to_features(examples, tokenizer=tokenizer, max_length=max_seq_length)
            elif data_type == 'id_test':
                examples = self.processor.get_examples(tokenizer, data_dir, data_type='id_test')
                self.features = convert_examples_to_features(examples, tokenizer=tokenizer, max_length=max_seq_length, is_train=False)
            elif data_type == 'ood_test':
                examples = self.processor.get_examples(tokenizer, data_dir, data_type='ood_test')
                self.features = convert_examples_to_features(examples, tokenizer=tokenizer, max_length=max_seq_length, is_train=False)
            else:
                raise NotImplementedError(f'{data_type=}')

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i].__dict__


def convert_examples_to_features(
        examples: List[InputExample],
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        is_train=True,
):
    features = []
    t = tqdm(enumerate(examples), total=len(examples), desc=f'Converting examples to features', ncols=125)

    for data_index, example in t:
        if is_train:
            input = tokenizer(example.input, truncation=True, max_length=max_length)
            output = tokenizer(example.output, truncation=True, max_length=max_length)
            features.append(InputFeature(input_ids=input.input_ids + output.input_ids + [tokenizer.eos_token_id],
                                         attention_mask=input.attention_mask + output.attention_mask + [1],
                                         labels=[-100] * len(input.input_ids) + output.input_ids + [tokenizer.eos_token_id]))
        else:
            input = tokenizer(example.input, truncation=True, max_length=max_length)
            output = tokenizer(example.output, truncation=True, max_length=max_length)
            features.append(InputFeatureTest(input_ids=input.input_ids,
                                             attention_mask=input.attention_mask,
                                             id=example.id,
                                             original_data=example.original_data))

    return features
