import fusekit.Common.env as env
import fusekit.Common.utils as utils

import torch, copy

from fusekit.Datasets import GenericSample
from fusekit.Common.Memory import MemoryManager, BatchDimensions, TensorDimensions

from transformers import BatchEncoding, PreTrainedModel

VERBOSE = False

'''
Before modifying, please refer to the PEP 8 Style Guide:
https://peps.python.org/pep-0008/

Limit all lines to a maximum of 79 characters.
'''

class ExtendedBatchEncoding(BatchEncoding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def input_ids(self) -> torch.Tensor:
        return self["input_ids"]
    
    def mask(self) -> torch.Tensor:
        return self["attention_mask"]

class BatchSamples:
    def __init__(self,
                 samples: list[(GenericSample, int)], 
                 max_new_tokens=0):
        self.samples: list[(GenericSample, int)] = samples
        self.max_new_tokens = max_new_tokens
        self.inputs = None
        self.labels = None
        self._index = 0

    def get_inputs(self):
        if self.inputs is None:
            self.inputs = self.encode(lambda sample: sample.get_inputs(), fn_type="get_inputs")
        return self.inputs
    
    def get_labels(self, training=False):
        if self.labels is None:
            self.labels = self.encode(lambda sample: sample.get_labels())
        
        answers = copy.deepcopy(self.labels)
        tokens = copy.deepcopy(self.get_inputs())
        if not training:
            tokens = answers
            tokens['input_ids'][self.labels['attention_mask'] == 0] = -100
        else:
           tokens['input_ids'] = torch.cat([tokens['input_ids'], answers['input_ids']],1)
           tokens['attention_mask'] = torch.cat([tokens['attention_mask'], answers['attention_mask']],1)
        return tokens

    def encode(self, fn_getter, fn_type=None) -> ExtendedBatchEncoding:
        tokens = []
        mask = []
        image_inputs = {}

        for (sample, subsample_idx) in self.samples:
            #print(sample, subsample_idx)
            pad = sample.tokenizer.pad_token_id
            sample_tokens = fn_getter(sample)
            if sample_tokens.shape[0] > 1:
                sample_tokens = sample_tokens[subsample_idx].tolist()
            else:
                sample_tokens = sample_tokens[0].tolist()
            mask.append([1 if token > 0 else 0 for token in sample_tokens])
            tokens.append([token if token > 0 else pad for token in sample_tokens])

            if sample.processor and fn_type == "get_inputs":
                filtered_ctx = sample.get_image_ctx()
                for (key, value) in filtered_ctx:
                    if key not in image_inputs:
                        image_inputs[key] = []
                    image_inputs[key].append(value.squeeze(dim=0).tolist())

        tokens = utils.padding.right(tokens, pad=pad)
        mask = utils.padding.right(mask)

        return ExtendedBatchEncoding({"input_ids": tokens,
                                      "attention_mask": mask, 
                                      **image_inputs},
                                      tensor_type="pt")
    
    def to(self, *args, **kwargs):
        self.get_inputs()
        self.inputs = self.inputs.to(*args, **kwargs)
        # Unnecessary and wastes GPU memory
        # self.labels = self.labels.to(*args, **kwargs)
        return self
    
    def add_next_token(self, next_tokens: torch.Tensor):
        device = self.get_inputs().input_ids().device
        next_tokens = next_tokens.to(device)

        rows, cols = self.get_inputs().input_ids().size()
        seq_lens = self.get_inputs().mask().sum(dim=1)

        if cols < max(seq_lens) + 1:
            input_ids = torch.ones((rows, cols + 1),
                                    dtype=self.get_inputs().input_ids().dtype,
                                    device=device)
            mask = torch.zeros_like(input_ids)

            input_ids[:, :-1] = self.get_inputs().input_ids()
            mask[:, :-1] = self.get_inputs().mask()

            self.inputs["input_ids"] = input_ids
            self.inputs["attention_mask"] = mask

        self.inputs["input_ids"][torch.arange(rows), seq_lens] = next_tokens
        self.inputs["attention_mask"][torch.arange(rows), seq_lens] = 1

    def __iter__(self):
        self._index = 0
        return self
    
    def __next__(self) -> GenericSample:
        if self._index < len(self.samples):
            result = self.samples[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, index: int) -> GenericSample:
        return self.samples[index]
    
class DynamicBatchLoader(object):
    """
    Custom loader that supports variable batch sizing based on the number of
    tokens in each sample.\n
    Helps prevent CUDA Out of Memory errors when batches contain samples that
    are too long.

    Attributes:
        batch_size (int): The maximum number of tokens allowed per batch:
            batch_size = batch_length * maximum_token_length_of_longest_sample.
    """

    def __init__(self,
                 model: PreTrainedModel,
                 dataset: list[GenericSample],
                 memoryManager: MemoryManager,
                 batch_images=True,
                 batch_size=0,
                 num_adapters=1,
                 training=False):
        self.model = model
        self.dataset = dataset
        self.batch_images = batch_images
        self.samples: list[(GenericSample, int)] = []
        self.mm = memoryManager
        #self.batch_memory = 0
        self.batch_size = batch_size
        self.num_adapters=num_adapters
        self.training = training

    def __len__(self):
        return len(self.dataset)

    def _active_adapter_count(self) -> int:
        """
        Best-effort adapter count from the underlying model.
        Falls back to configured num_adapters when unavailable.
        """
        try:
            if hasattr(self.model, "active_adapters"):
                active = self.model.active_adapters()
                if isinstance(active, str):
                    return 1
                if active is None:
                    return 1
                return max(1, len(active))
        except Exception:
            pass
        return max(1, int(self.num_adapters))

    def _uses_logit_composition(self) -> bool:
        """
        Logit composition patches model.forward and keeps model._orig_forward.
        """
        try:
            orig_forward = getattr(self.model, "_orig_forward", None)
            cur_forward = getattr(self.model, "forward", None)
            return (orig_forward is not None) and (cur_forward is not None) and (cur_forward is not orig_forward)
        except Exception:
            return False

    def _adapter_memory_factor(self) -> int:
        """
        Only logit composition performs a full forward per adapter, so KV-cache
        and activation-like memory should scale with active adapters in that case.
        """
        if self._uses_logit_composition():
            return self._active_adapter_count()
        return max(1, int(self.num_adapters))

    def batch_full(self, next_sample_size) -> bool:
        if next_sample_size > self.mm.get_unallocated_memory():
            return True
        elif self.batch_size != 0 and len(self.samples) + 1 > self.batch_size:
            return True
        else:
            #self.batch_memory = next_sample_size
            return False
    
    def __iter__(self):
        self.samples: list[GenericSample] = []
        #self.batch_memory = 0
        batch_dims = BatchDimensions()
        max_train_input_len = 0
        max_new_tokens = 0
        adapter_factor = self._adapter_memory_factor()
        if VERBOSE:
            print(f'Unallocated Memory: {self.mm.get_unallocated_memory()} MB')
            print(f'Adapter Memory Factor: {adapter_factor}')
        for sample in self.dataset:
            inputs = sample.get_inputs()
            assert inputs.dim() == 2, (
                f"Expected 2D tensor, but got {inputs.dim()}D tensor."
            )
            input_len = inputs.shape[1]

            if hasattr(sample, 'max_new_tokens'):
                input_len += sample.max_new_tokens

            for subsample_idx, _ in enumerate(inputs):
                temp = BatchDimensions('input_ids', TensorDimensions([1,input_len]))
                temp.add_tensor('attention_mask', TensorDimensions([1,input_len]))
                max_new_tokens = max(max_new_tokens, sample.max_new_tokens)
                if sample.processor:
                    filtered_ctx = sample.get_image_ctx()
                    for key, tensor in filtered_ctx:
                        temp.add_tensor(key, TensorDimensions(list(tensor.shape)))

                if self.training:
                    B = len(self.samples) + 1
                    T = max(max_train_input_len, input_len)
                    sample_size = self.mm.required_train_memory(1, input_len)
                    new_batch_size = self.mm.required_train_memory(B, T) * adapter_factor
                else:
                    sample_size = self.mm.required_memory(temp)
                    batch_dims.merge(temp)
                    new_batch_size = self.mm.required_memory(batch_dims) * adapter_factor

                if VERBOSE:
                    print(f'Sample Size: {sample_size:.2f} MB')
                assert sample_size <= self.mm.get_unallocated_memory(), (
                    f'Unallocated CUDA memory of size {self.mm.get_unallocated_memory()} MB '
                    f'is too small for Sample of length {input_len} tokens (required space: {sample_size} MB)'
                )
                if sample_size * adapter_factor > self.mm.get_unallocated_memory():
                    print(f'Unallocated CUDA memory of size {self.mm.get_unallocated_memory()} MB might be too small for Sample of length {input_len} tokens (recommended space: {sample_size * adapter_factor} MB)')

                if not self.batch_full(new_batch_size):
                    self.samples.append((sample, subsample_idx))
                    if self.training:
                        max_train_input_len = max(max_train_input_len, input_len)
                else:
                    if VERBOSE:
                        print(f'Batch Memory Required: {new_batch_size:.2f} MB')
                    yield BatchSamples(self.samples, max_new_tokens=max_new_tokens)
                    self.samples = [(sample, subsample_idx)]
                    batch_dims = BatchDimensions()
                    max_new_tokens = sample.max_new_tokens
                    if self.training:
                        max_train_input_len = input_len

                # print(f'Batch needs : {new_batch_size}MB')
                # Some models do not support multimodal batching
                if not self.batch_images and sample.image and self.samples:
                    #print(f'Batch Memory Required: {self.batch_size:.2f} MB')
                    
                    yield BatchSamples(self.samples, max_new_tokens=max_new_tokens)
                    self.samples: list[GenericSample] = []
                    batch_dims = BatchDimensions()
                    max_train_input_len = 0
                    max_new_tokens = 0
        if self.samples:
            #print(f'Batch Memory Required: {self.batch_size:.2f} MB')
            yield BatchSamples(self.samples, max_new_tokens=max_new_tokens)
