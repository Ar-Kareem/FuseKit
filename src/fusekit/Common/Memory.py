import torch
import torch.nn as nn
import math
import gc
from transformers import PreTrainedModel

import subprocess
import re
import sys
import time
import warnings

def clear_cuda(verbose=True):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        if verbose:
            print()
            print("GPU Memory Emptied")
            print(f"Memory reserved: {torch.cuda.memory_reserved() / 1e6} MB")
            print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e6} MB")
            print()
    else:
        print("No GPU available. Memory not cleared.")

def print_memory_allocation():
    if torch.cuda.is_available():
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e6} MB")
    else:
        print("No GPU available. Memory allocation not printed.")

def get_gpu_status():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            idx, name, total, used, free = re.split(r",\s*", line)
            gpus.append({
                "index": int(idx),
                "name": name,
                "total_MB": math.floor(int(total) / 1024) * 1000,
                "used_MB":  math.floor(int(used)  / 1024) * 1000,
                "free_MB":  math.floor(int(free)  / 1024) * 1000,
            })
        return gpus
    except Exception as e:
        print(f"Failed to query nvidia-smi: {e}", file=sys.stderr)
        return []

def get_available_gpus(threshold=0.9, retries=1):
    """
    Returns (device_indices, memory_limit_MB)
    - Retries if GPU detection returns None.
    - Sleeps 1s between retries.
    - Raises RuntimeError if no qualifying GPUs are found after all attempts
    """
    result = None

    for attempt in range(retries + 1):
        gpus = get_gpu_status()
        if gpus is None:
            # get_gpu_status itself failed; treat as empty list
            gpus = []

        last_seen = gpus
        qualifying = [g for g in gpus if g["free_MB"] >= threshold * g["total_MB"]]
        if qualifying:
            memory_limit = min(qualifying, key=lambda g: g["free_MB"])["free_MB"]
            devices = [g["index"] for g in qualifying]
            return devices, memory_limit

        if attempt < retries:
            time.sleep(1)

    raise RuntimeError(
        f"No GPUs met the {int(threshold*100)}% free-memory threshold after {retries+1} attempt(s). " + \
        f"Last observed GPUs: {[{'index': g.get('index'), 'free_MB': g.get('free_MB'), 'total_MB': g.get('total_MB')} for g in (last_seen or [])]}"
    )

class TensorDimensions:
    def __init__(self, shape: list):
        self.shape = shape

class BatchDimensions:
    def __init__(self, key=None, tensor_object=None):
        self.objects = {}

        if key:
            self.objects[key] = tensor_object

    def add_tensor(self, key, tensor_dimensions: TensorDimensions):
        if key not in self.objects:
            self.objects[key] = tensor_dimensions
            return
        
        self._add_tensor(key, tensor_dimensions)

    def _add_tensor(self, key, tensor_dimensions: TensorDimensions):
        dim = len(self.objects[key].shape)
        assert dim == len(tensor_dimensions.shape), f"dimension mismatch: Initialized with {dim} " + \
             f"dimensions, but attempted to add object of dimension {len(tensor_dimensions.shape)}"
        
        new_shape = self.objects[key].shape
        new_shape[0] += 1
        for idx, dim in enumerate(new_shape):
            new_shape[idx] = max(dim, tensor_dimensions.shape[idx])

        self.objects[key].shape = new_shape

    def merge(self, batch_dimensions):
        # TODO: When stabilized revert to isinstance()
        # Jupyter fails isinstance checks when Memory.py is modified
        # if not isinstance(batch_dimensions, BatchDimensions):
        # if batch_dimensions.__class__ is not BatchDimensions:
        #     raise TypeError("Can only merge with another BatchDimensions object.")

        for key, tensor_dimensions in batch_dimensions.objects.items():
            if key not in self.objects:
                self.objects[key] = tensor_dimensions
            else:
                self._add_tensor(key, tensor_dimensions)

    def get_num_by_key(self, key):
        total_values = 0
        if key in self.objects:
            num_values = 1
            for dim in self.objects[key].shape:
                num_values *= dim
            total_values += num_values
        return total_values

    def get_num_values(self):
        total_values = 0
        for key, tensor in self.objects.items():
            num_values = 1
            for dim in tensor.shape:
                num_values *= dim
            total_values += num_values
        return total_values

class MemoryManager:
    BYTES_TO_MB = (1024 ** 2)
    MEMORY_OVERHEAD_FACTOR = 1.00
    INFERENCE_OVERHEAD_FACTOR = 4 # Key, Query, Value, Output matrices

    def __init__(self, model: PreTrainedModel, device_list: list[int],
                 memory_limit=1024, vision_embed_size=None):
        self.device_list = device_list
        self.memory_limit = memory_limit
        self.precision = model.dtype
        self.model = model
        self.vision_embed_size = vision_embed_size

        self.device_assignments = {}
        self.used_memory = 0
        self.gpu_idx = 0

        self._model_memory = None
        self._logit_size = None
        self._pixel_size = None

        print(f'Using {self.memory_limit} MB per GPU on {len(self.device_list)} GPUs')
        print(f'Additional Memory Required to Train: {self.trainable_param_mb()} MB')
        #print(f'Estimated Training Overhead Memory: {self.constant_overhead_mb()} MB')

    def get_cuda_assignment(self):
        self.device_assignments = {}
        self.used_memory = 0
        self.gpu_idx = 0

        rounding_factor = 1000
        self._model_memory = math.ceil(self.get_model_memory(entire_model=True) / len(self.device_list)/rounding_factor) * rounding_factor

        assert self.get_model_memory() < self.get_memory_limit(), f'Not enough CUDA memory is available for model of {self.get_model_memory():.2f} MB across {len(self.device_list)} GPUs with limit of {self.memory_limit} MB each'

        # base, prefix = self.get_base_and_prefix()
        # return self.module_assignment(base, prefix=prefix)
        return self.module_assignment(self.model)
    
    def get_base_and_prefix(self):
        """
        Recursively unwraps a wrapped HF PreTrainedModel (e.g., PeftModel, LoraModel)
        and returns the base model and its attribute prefix (e.g., '.model', '.base_model.model').
        """
        prefix_parts = []

        def unwrap(m):
            for attr in ['base_model', 'model']:
                if hasattr(m, attr):
                    child = getattr(m, attr)
                    # Check for recursive unwrapping
                    if isinstance(child, PreTrainedModel) or hasattr(child, 'forward'):
                        prefix_parts.append(attr)
                        return unwrap(child)
            return m

        base_model = unwrap(self.model)
        prefix = '.' + '.'.join(prefix_parts) if prefix_parts else ''
        return base_model, prefix
    
    def module_parameters(self, module: torch.nn.Module):
        """Collect parameters that belong directly to this module (not its children)."""
        children = [name for name, _ in module.named_children()]
        param_dict = {}
        for name, param in module.named_parameters(recurse=False):
            # top-level params that aren't part of submodules
            num_params = param.numel()
            param_dict[name] = num_params
        return param_dict

    def _module_assignment(self, name, num_params, submodule=None, prefix=None):
        module_memory = MemoryManager.compute_memory_from_params(num_params=num_params, dtype=self.precision)

        # Build the display key (keeps your existing bracket/numeric behavior)
        disp_name = f'[{name}]' if str(name).isdigit() else f'.{name}'
        key = f'{prefix}{disp_name}' if prefix else disp_name

        # --- NEW: always recurse into ModuleList (e.g., "layers") to expose [0]..[N-1]
        if submodule is not None and isinstance(submodule, torch.nn.ModuleList):
            # Force recursion so blocks shard individually
            self.module_assignment(submodule, prefix=key)
            return

        # If too big to fit in the remaining budget of this GPU, recurse into children
        if submodule is not None and module_memory > self.get_model_memory():
            self.module_assignment(submodule, prefix=key)
            return

        # Greedy pack as before
        if module_memory < self.get_model_memory() - self.used_memory:
            self.device_assignments[key] = f'cuda:{self.device_list[self.gpu_idx]}'
            self.used_memory += module_memory
        else:
            if self.gpu_idx < len(self.device_list) - 1:
                self.gpu_idx += 1
            self.used_memory = module_memory
            self.device_assignments[key] = f'cuda:{self.device_list[self.gpu_idx]}'

            # Normal greedy fill
            if module_memory < self._model_memory - self.used_memory:
                self.device_assignments[key] = f'cuda:{self.device_list[self.gpu_idx]}'
                self.used_memory += module_memory
            else:
                # Move to next GPU when over the limit
                if self.gpu_idx < len(self.device_list) - 1:
                    self.gpu_idx += 1
                self.used_memory = module_memory
                self.device_assignments[key] = f'cuda:{self.device_list[self.gpu_idx]}'

    def module_assignment(self, module: torch.nn.Module, prefix=None):
        # If this node itself is a ModuleList, iterate its indexed children directly.
        if isinstance(module, torch.nn.ModuleList):
            for idx, child in module.named_children():            # idx is '0','1',...
                num_params = sum(p.numel() for p in child.parameters())
                self._module_assignment(idx, num_params, submodule=child, prefix=prefix)
            return self.device_assignments

        # 1) direct parameters of this module (no recursion)
        param_dict = self.module_parameters(module)
        for name, num_params in param_dict.items():
            self._module_assignment(name, num_params, submodule=None, prefix=prefix)

        # 2) recurse into children (including "layers", "embed_tokens", etc.)
        for name, submodule in module.named_children():
            num_params = sum(p.numel() for p in submodule.parameters())
            self._module_assignment(name, num_params, submodule=submodule, prefix=prefix)

        return self.device_assignments
    
    # def module_parameters(self, module: torch.nn.Module):
    #     children = [name for name, _ in module.named_children()]
    #     param_dict = {}

    #     for name, params in module.named_parameters():
    #         if name.split('.')[0] not in children:
    #             num_params = sum(p.numel() for p in params)
    #             param_dict[name] = num_params

    #     return param_dict

    # def _module_assignment(self, name, num_params, submodule=None, prefix=None):
    #     module_memory = MemoryManager.compute_memory_from_params(num_params=num_params, dtype=self.precision)
    #     # print(f'{name} uses {module_memory:.2f} MB')

    #     name = f'[{name}]' if name.isdigit() else f'.{name}'
    #     key = f'{prefix}{name}' if prefix else name

    #     if submodule is not None and module_memory > self.get_model_memory():
    #         self.module_assignment(submodule, key)
    #     elif module_memory < self.get_model_memory() - self.used_memory:
    #         self.device_assignments[key] = f'cuda:{self.device_list[self.gpu_idx]}'
    #         self.used_memory += module_memory
    #     else:
    #         if self.gpu_idx < len(self.device_list) - 1:
    #             self.gpu_idx += 1
    #         self.used_memory = module_memory
    #         self.device_assignments[key] = f'cuda:{self.device_list[self.gpu_idx]}'

    # def module_assignment(self, module: torch.nn.Module, prefix=None):
    #     param_dict = self.module_parameters(module)
    #     for name, num_params in param_dict.items():
    #         self._module_assignment(name, num_params, submodule=None, prefix=prefix)

    #     for name, submodule in module.named_children():
    #         num_params = sum(p.numel() for p in submodule.parameters())
    #         self._module_assignment(name, num_params, submodule=submodule, prefix=prefix)
    #     return self.device_assignments

    def compute_model_memory(self):
        num_params = sum(p.numel() for p in self.model.parameters())
        return MemoryManager.compute_memory_from_params(num_params=num_params, 
                                                        dtype=self.precision)
    
    @staticmethod
    def compute_memory_from_params(num_params, dtype):
        bytes_per_param = torch.tensor([], dtype=dtype).element_size()
        total_memory = num_params * bytes_per_param
        
        total_memory_MB = total_memory / MemoryManager.BYTES_TO_MB

        model_memory = total_memory_MB * MemoryManager.MEMORY_OVERHEAD_FACTOR
        
        return model_memory
    
    def get_bytes_per_element(self):
        return torch.tensor([], dtype=self.precision).element_size()

    def compute_logit_size(self):
        # Prefer the model's output head (works even if vocab was resized/tied)
        try:
            oe = self.model.get_output_embeddings()
            if oe is not None and hasattr(oe, "weight"):
                vocab_size = int(oe.weight.shape[0])
            else:
                raise AttributeError
        except Exception:
            # Simple config fallbacks for multimodal models (e.g., Mllama)
            cfg = getattr(self.model, "config", None)
            vocab_size = (
                int(getattr(cfg, "vocab_size", 0))
                or int(getattr(getattr(cfg, "text_config", None), "vocab_size", 0))
            )
            if not vocab_size:
                # last-ditch: common head names
                head = getattr(self.model, "lm_head", None)
                if head is not None and hasattr(head, "out_features"):
                    vocab_size = int(head.out_features)
                elif head is not None and hasattr(head, "weight"):
                    vocab_size = int(head.weight.shape[0])
                else:
                    raise RuntimeError("Could not determine vocab size")
            
        bytes_per_logit = vocab_size * self.get_bytes_per_element()
        return  bytes_per_logit / MemoryManager.BYTES_TO_MB * MemoryManager.INFERENCE_OVERHEAD_FACTOR

    def compute_pixel_size(self):
        bytes_per_pixel = self.vision_embed_size * self.get_bytes_per_element()
        return  bytes_per_pixel / MemoryManager.BYTES_TO_MB
    
    def _is_gc_enabled(model) -> bool:
        # HF models expose different flags; check a few
        if hasattr(model, "is_gradient_checkpointing") and model.is_gradient_checkpointing:
            return True
        if getattr(getattr(model, "config", None), "gradient_checkpointing", False):
            return True
        return False

    def _uses_flash_attn(self) -> bool:
        impl = getattr(getattr(self.model, "config", None), "_attn_implementation", None)
        return impl in ("flash_attention_2", "flash_attention")

    def trainable_param_mb(self) -> float:
        """LoRA-aware: only params that require grad."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        grad_bytes = trainable * self.get_bytes_per_element()               # grads ≈ param dtype
        adam_bytes = trainable * 8                       # Adam m+v in fp32
        return (grad_bytes + adam_bytes) / (1024**2)
    
    def constant_overhead_mb(self):
        return 300.0

    def required_train_memory(self, B, T):
        cfg = self.model.config
        if 'num_hidden_layers' not in cfg and 'text_config' in cfg:
            cfg = cfg.text_config
        L   = getattr(cfg, 'num_hidden_layers', None) or cfg.num_hidden_layers
        H   = getattr(cfg, 'hidden_size', None) or cfg.hidden_size
        I   = getattr(cfg, 'intermediate_size', None) or 4 * H  # common FFN width
        V   = getattr(cfg, 'vocab_size', None) or getattr(self.model, 'vocab_size', 32000)

        # Attention saved activations per token per layer:
        # roughly: Q, K, V, attn_out  => ~4H
        attn_saved = 4 * H

        # FlashAttention keeps additional work buffers ~O(B*T*H) (not T*T)
        attn = 2 * H if self._uses_flash_attn() else 0

        # MLP saved activations: up-proj + nonlinearity + down-proj inputs => ~2I
        mlp_saved = 2 * I

        per_token_per_layer = attn_saved + attn + mlp_saved  # ~ (6H + 2I)

        # Gradient checkpointing reduces what is saved. Crude but effective factor:
        gradient_checkpointing = 0.5 if self._is_gc_enabled() else 1.0

        activations_bytes = B * T * L * per_token_per_layer * self.get_bytes_per_element() * gradient_checkpointing

        # logits [B, T, V] and 1 temp buffer of same shape during softmax/log-softmax
        ce_bytes = 2 * B * T * V * self.get_bytes_per_element()

        total_mb = (activations_bytes + ce_bytes) / self.BYTES_TO_MB + self.trainable_param_mb() + self.constant_overhead_mb()
        return total_mb

    def required_memory(self, batch_object: BatchDimensions):
        input_ids = batch_object.get_num_by_key("input_ids") 
        pixel_values = batch_object.get_num_by_key("pixel_values") 
        all_values = batch_object.get_num_values() 

        #print(input_ids, pixel_values, all_values)
        #print(self.get_logit_size(), self.get_pixel_size())
        
        token_size = input_ids * self.get_logit_size() + pixel_values * self.get_pixel_size()
        remaining_size = ((all_values - input_ids - pixel_values) * 
                          torch.tensor([], dtype=self.precision).element_size() / 
                          MemoryManager.BYTES_TO_MB)

        return token_size + remaining_size

    def get_pixel_size(self):
        if self._pixel_size is None:
            self._pixel_size = self.compute_pixel_size()
        return self._pixel_size

    def get_logit_size(self):
        if self._logit_size is None:
            self._logit_size = self.compute_logit_size()
        return self._logit_size

    # Returns the size of the model or the size of each shard
    def get_model_memory(self, entire_model=False):
        if entire_model or not self._model_memory:
            return self.compute_model_memory()
        else: 
            return self._model_memory
    
    def get_memory_limit(self):
        return self.memory_limit
    
    def get_unallocated_memory(self):
        return self.get_memory_limit() - self.get_model_memory()
    