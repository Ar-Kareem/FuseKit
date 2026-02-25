import os, datetime, warnings
import fusekit.Common.env as env

import types
import torch
import torch.nn.functional as F
from torch.cuda import CudaError

from typing import List

from tqdm import tqdm

from peft import LoraConfig, PeftMixedModel
from peft.tuners.lora.layer import LoraLayer

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

from .composition import Composition, WeightComposition, LogitComposition
from .base import CausalModelBase, APIModelBase, GenericModel
from fusekit.Datasets import IterableDataset, APICost, GenerationSample, TextVisionSample
from fusekit.Common.EvalType import EvalType as EvalType
from fusekit.Common.Batching import BatchSamples, DynamicBatchLoader
from fusekit.Common.Memory import MemoryManager, print_memory_allocation, get_available_gpus, clear_cuda
from fusekit.Modeling.model_hooks import InputsToHook, attach_hook, detach_hook

from fusekit.Modeling.training import TrainingMixin

VERBOSE = False

class CausalModel(TrainingMixin, CausalModelBase):
    """
    A CausalModel class that extends the Hugging Face `PreTrainedModel` to 
    support model evaluation on a GPU device with memory management and custom
    methods to interface with fusekit.Datasets in this package.

    Attributes:
        model (PreTrainedModel): The underlying pretrained model instance.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.
        device (list[int]|int|None): The GPU device indexes available for the model.
        memory_limit (int): The maximum GPU memory (in MB) allocated for the model.
        force_sharding: Boolean to force sharding even if model fits on one GPU.   
        batch_images: batches images to pass into model. #warning batching currently does not work on images
    """

    def __init__(self,
                 model: PreTrainedModel, 
                 processor: ProcessorMixin|None,
                 tokenizer: PreTrainedTokenizer,
                 device: list[int]|int=None,
                 memory_limit: int=None,
                 vision_embed_size=0,
                 force_sharding=True,
                 batch_images=False):
        super().__init__(model, processor, tokenizer, device)
        if model.dtype in (torch.bfloat16,torch.float16):
            model.half()

        clear_cuda(verbose=False)
        available_devices, detected_memory_limit = get_available_gpus()
        if device is None:
            warnings.warn(f'Device not specified! Using detected available GPUs: {available_devices}', UserWarning)
            device = available_devices

        if memory_limit is None:
            warnings.warn(f'memory_limit not specified! Use detected available memory: {detected_memory_limit} MB', UserWarning)
            memory_limit = detected_memory_limit
            
        self.model_name = "CausalModel"
        self.vision_embed_size = vision_embed_size
        self.shard_model = force_sharding
        self.batch_images = batch_images
        self.device_args = device
        self.memory_limit = memory_limit
        self.device_list = GenericModel.parse_device(device)
        self.device_map = None
        self.mm = self.update_memory(memory_limit, 
                                     force_sharding=self.shard_model)
        
    def sanitize(self):
        super().sanitize()
        self.remove_hooks()

    def update(self):
        print(f"Updating Memory Manager for {self.model_name}")
        super().update()
        self.mm = self.update_memory(self.memory_limit, force_sharding=self.shard_model)
    
    def update_memory(self, memory_limit, force_sharding=True):
        if isinstance(self.model, torch.nn.DataParallel):
            print("Disabling DataParallel")
            self.model = self.model.module

        self.mm = MemoryManager(self.model, device_list=self.device_list,
                                memory_limit=memory_limit, 
                                vision_embed_size=self.vision_embed_size)
        self._use_sharding(force_sharding)
        print()
        return self.mm

    def _use_sharding(self, force_sharding):
        assert self.mm is not None, 'self.mm has not been initialized.'\
            'Do not call update_model() before initializing MemoryManager'
        
        print(f'Memory Required for Inference ' + 
            f'{self.mm.get_model_memory(entire_model=True):.2f} MB')

        # if isinstance(self.model, torch.nn.DataParallel):
        #     print("Disabling DataParallel")
        #     self.model = self.model.module
        # else:
        #     print("Removing Sharding hooks")
        #     self.remove_hooks()
        if force_sharding \
            and len(self.device_list) > 1 \
                or (self.mm.get_model_memory(entire_model=True) 
                    > self.mm.get_memory_limit()
                    ):
                self.shard_model = True 
                self.device_map = self.mm.get_cuda_assignment()
                self.attach_hooks()
                print(f"Using GPU Sharding: {self.device_list}")
        else:
            self.shard_model = False
            if self.device_list and len(self.device_list) > 1:
                print(f"Using multiple GPUs: {self.device_list}")
                self.model = torch.nn.DataParallel(self.model, 
                                               device_ids=self.device_list)
            elif self.device_list:
                print(f"Using single GPU: {self.device_list[0]}")

        if not self.shard_model \
            and self.mm.get_unallocated_memory() < 0 \
                and self.device_list:
            raise CudaError(
                f"Insufficient CUDA memory for model initialization: "
                f"required {self.mm.get_model_memory(entire_model=True)} MB, "
                f"available {self.mm.get_memory_limit()} MB on device(s) "
                f"{self.device_list}."
            )

        return self.model
        
    def get_module(self, name):
        return eval(f'self.model{name}')

    def attach_hooks(self):
        if self.shard_model:
            for submodule_name, device in self.device_map.items():
                submodule = self.get_module(submodule_name) 
                if isinstance(submodule, torch.nn.Module):               
                    submodule = attach_hook(submodule, InputsToHook(device))

    def remove_hooks(self):
        if self.device_map:
            for module_name, device in self.device_map.items():
                module = self.get_module(module_name)
                module = detach_hook(module)
            self.device_map = None

    def dispatch_model(self):
        if self.shard_model:
            print("Sharding model across multiple GPUs")
            for module_name, device in self.device_map.items():
                module = self.get_module(module_name)
                if isinstance(module, torch.nn.Module): 
                    module.to(device)
                elif isinstance(module, torch.nn.Parameter): 
                    module.data = module.data.to(device)
        elif self.device_list:
            print("Sending model to GPU")
            self.model.to(self.device_list[0])
        else:
            print("GPU Device not specified--Running on CPU")
            self.model.to("cpu")

    def __call__(self, encoding: BatchSamples) -> CausalLMOutputWithPast:
        return self.model(**encoding.get_inputs(), use_cache=False)

    def get_next_token(self, encoding: BatchSamples):
        outs: CausalLMOutputWithPast = \
            self.model(**encoding.get_inputs(), use_cache=False)

        token_indices = encoding.get_inputs().mask().sum(dim=-1) - 1
        next_logits = outs.logits[torch.arange(outs.logits.size(0)),
                                  token_indices, :]
        next_tokens = torch.argmax(next_logits, dim=-1)

        return next_tokens
    
    def loglikelihood(self, encoding: BatchSamples):
        with torch.no_grad():
            outs = self(encoding)
            logits = outs.logits
            logit_prob = F.log_softmax(logits, dim=-1)
            if VERBOSE:
                print_memory_allocation()

        for batch_idx, (sample, candidate_idx) in enumerate(encoding):
            candidate = sample.options[candidate_idx][0]
            candidate_ids = sample.encode(candidate)
            attn_len = encoding.inputs.mask()[batch_idx].sum().item()
            candidate_ids = encoding.inputs.input_ids()[
                batch_idx, attn_len-len(candidate_ids):attn_len]
            relevant_logits = logit_prob[
                batch_idx, attn_len-len(candidate_ids)-1:attn_len-1, :]

            assert len(relevant_logits) == len(candidate_ids), f'Wrong dimensions'
            candidate_loglikelihood = 0.0
            for i in range(len(candidate_ids)):
                candidate_loglikelihood += relevant_logits[
                    i, candidate_ids[i]].item()
            sample.loglikelihood[candidate_idx] = candidate_loglikelihood/len(candidate_ids)

    def generate(self, encoding: BatchSamples):
        model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        outs = model.generate(**encoding.get_inputs(), 
                                   max_new_tokens=encoding.max_new_tokens,
                                   top_p=0, do_sample=True)
        
        for batch_idx, (sample, candidate_idx) in enumerate(encoding):
            input_len = sample.get_inputs().shape[-1]
            sample.outs = outs.tolist()[batch_idx][input_len:-1]
            sample.pred_text = self.tokenizer.decode(sample.outs, skip_special_tokens=True)
            sample.update_tokens_used(input_len + len(sample.outs))

            #print(self.tokenizer.decode(sample.outs))
            #print(self.tokenizer.decode(sample.label))

            # print(sample.outs)
            # print(sample.label)

    def evaluate(self, datasets: list[IterableDataset] | IterableDataset, llm_judge=None, save_file=None, skip_generation=False):
        if not isinstance(datasets, list):
            datasets = [datasets]
            
        if not skip_generation:
            self.eval()
            self.dispatch_model()

        super().evaluate(datasets, llm_judge=llm_judge, save_file=save_file, skip_generation=skip_generation)

        if not skip_generation:
            self.to('cpu')

        datasets_metrics = {}
        datasets_raw_metrics = {}
        for dataset in datasets:
            if len(datasets) > 1:
                print(f'--- {dataset.name} ---')
            metrics, raw = dataset.accuracy()
            datasets_metrics[dataset.name] = metrics
            datasets_raw_metrics[dataset.name] = raw

        return datasets_metrics, datasets_raw_metrics

    def run_model(self, dataset: IterableDataset):
        num_adapters = len(self.model.active_adapters()) if self._hf_peft_config_loaded else 1    
        dataloader = DynamicBatchLoader(self.model, dataset, 
                                        memoryManager=self.mm,
                                        batch_images=self.batch_images,
                                        num_adapters=num_adapters)

        pbar = tqdm(total=len(dataset), desc="Generating Samples", unit="sample")
        for batch in dataloader:
            if self.device_list:
                batch.to(self.device_list[0])

            if dataset.type is EvalType.LOGLIKELIHOOD:
                self.loglikelihood(batch)
            elif dataset.type is EvalType.SIMILARITY:
                self.generate(batch)
            else:
                raise ValueError(f"Unsupported dataset type: {dataset.type}.")
            
            pbar.update(len(batch))
            batch.to('cpu')

    def evaluate_options(self, dataset: IterableDataset, skip_generation, llm_judge):
        if not skip_generation:
            self.run_model(dataset)

        if llm_judge is not None and isinstance(llm_judge, APIModelBase):
            llm_judge.judge_answers(dataset, llm_judge)
            dataset.cost()

    def finetune(self, lr, weight_decay, epochs,
                 train_dataset: IterableDataset, 
                 val_datasets: IterableDataset,
                 batch_size=0,
                 save_path=None,
                 overwrite=False):
        
        train_dataloader = DynamicBatchLoader(self.model, train_dataset, memoryManager=self.mm, 
                                              batch_images=self.batch_images, batch_size=batch_size, training=True)

        self.dispatch_model()
        super().finetune(lr, weight_decay, epochs, train_dataloader, val_datasets, save_path=save_path, overwrite=overwrite)
        self.to('cpu')

    def load_adapters(self, lora_checkpoints: List[str], composition: Composition = None):
        if not isinstance(composition, Composition):
            raise TypeError("composition must be a Composition instance.")

        if not isinstance(lora_checkpoints, list):
            lora_checkpoints = [lora_checkpoints]

        self.adapter_name_prefix = "lora_"

        if hasattr(self.model, "peft_config"):
            for name in list(self.model.peft_config.keys()):
                self.model.delete_adapter(name)

        if hasattr(self.model, "_orig_forward"):
            self.model.forward = self.model._orig_forward
        if hasattr(self.model, "_logit_composition_past_key_values"):
            delattr(self.model, "_logit_composition_past_key_values")

        # --- Load reference config from first adapter dir ---
        ref_cfg = LoraConfig.from_pretrained(lora_checkpoints[0])
        required_keys = ["r", "lora_alpha", "lora_dropout"]

        # --- Sanity check: make sure all configs match ---
        for path in lora_checkpoints[1:]:
            cfg = LoraConfig.from_pretrained(path)
            for k in required_keys:
                if getattr(cfg, k) != getattr(ref_cfg, k):
                    raise ValueError(f"Inconsistent {k}: {getattr(cfg, k)} != {getattr(ref_cfg, k)}")

        # base = self.model if isinstance(self.model, (PeftMixedModel)) else self.model
        # first_name  = f"{self.adapter_name_prefix}0"
        # first_path  = str(lora_checkpoints[0])
        # mixed = PeftMixedModel.from_pretrained(base, first_path, adapter_name=first_name)

        adapter_names = []
        for i, path in enumerate(lora_checkpoints):
            adapter_name = f"{self.adapter_name_prefix}{i}"
            self.model.load_adapter(path, adapter_name=adapter_name)
            adapter_names.append(adapter_name)

        self.model.set_adapter(adapter_names)

        for module in self.model.modules():
            if isinstance(module, LoraLayer) and isinstance(composition, WeightComposition):
                module.forward = types.MethodType(composition.make_forward(), module)

        if isinstance(composition, LogitComposition):
            if not hasattr(self.model, "_orig_forward"):
                self.model._orig_forward = self.model.forward
            self.model.forward = types.MethodType(composition.make_forward(), self.model)

        return self

