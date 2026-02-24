import math
import warnings
import copy

from torch import Tensor
from typing import Dict, Callable, Any
import torch

from peft import PeftMixedModel
from peft.tuners.lora.layer import LoraLayer

class Composition:
    def __call__(self, layer, base_out: Tensor, x: Tensor) -> Tensor:
        raise NotImplementedError
    
class WeightComposition(Composition):
    def make_forward(self) -> Callable:
        def forward(layer: LoraLayer, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
            layer._check_forward_args(x, *args, **kwargs)
            adapter_names = kwargs.pop("adapter_names", None)

            # keep PEFT control flow intact
            if layer.disable_adapters:
                if layer.merged:
                    layer.unmerge()
                return layer.base_layer(x, *args, **kwargs)
            elif adapter_names is not None:
                return layer._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
            elif layer.merged:
                return layer.base_layer(x, *args, **kwargs)

            # collect base output
            base_out = layer.base_layer(x, *args, **kwargs)
            torch_result_dtype = base_out.dtype

            result = self(layer, base_out, x)
            return result.to(torch_result_dtype)

        return forward
    
class SumOfDeltas(WeightComposition):
    def __call__(self, layer: LoraLayer, base_out: Tensor, x: Tensor):
        result = base_out
        for active_adapter in layer.active_adapters:
            if active_adapter not in layer.lora_A:
                continue

            lora_A = layer.lora_A[active_adapter]
            lora_B = layer.lora_B[active_adapter]
            dropout = layer.lora_dropout[active_adapter]
            scaling = layer.scaling[active_adapter]

            x_cast = layer._cast_input_dtype(x, lora_A.weight.dtype).contiguous()
            result = result + lora_B(lora_A(dropout(x_cast))) * scaling
        return result

class AverageOfDeltas(WeightComposition):
    def __call__(self, layer: LoraLayer, base_out: Tensor, x: Tensor):
        result = base_out
        for active_adapter in layer.active_adapters:
            if active_adapter not in layer.lora_A:
                continue

            lora_A = layer.lora_A[active_adapter]
            lora_B = layer.lora_B[active_adapter]
            dropout = layer.lora_dropout[active_adapter]
            scaling = layer.scaling[active_adapter] / len(layer.active_adapters)

            x_cast = layer._cast_input_dtype(x, lora_A.weight.dtype).contiguous()
            result = result + lora_B(lora_A(dropout(x_cast))) * scaling
        return result
    
class PEMAddition(WeightComposition):
    def __call__(self, layer: LoraLayer, base_out: Tensor, x: Tensor):
        result = base_out
        
        lora_A = layer.lora_A[layer.active_adapters[0]].weight.clone()
        lora_B = layer.lora_B[layer.active_adapters[0]].weight.clone()
        dropout = layer.lora_dropout[layer.active_adapters[0]]
        scaling = layer.scaling[layer.active_adapters[0]]

        for active_adapter in layer.active_adapters[1:]:
            if active_adapter not in layer.lora_A:
                continue

            lora_A += layer.lora_A[active_adapter].weight
            lora_B += layer.lora_B[active_adapter].weight
            
        x_cast = layer._cast_input_dtype(x, lora_A.dtype).contiguous()    
        lora_A_out = torch.nn.functional.linear(dropout(x_cast), lora_A)
        lora_B_out = torch.nn.functional.linear(lora_A_out, lora_B)
        result = result + lora_B_out * scaling
        return result

class LoraHub(WeightComposition):
    def __call__(self, layer: LoraLayer, base_out: Tensor, x: Tensor):
        result = base_out

        adapters = [
            a for a in layer.active_adapters
            if (a in layer.lora_A) and (a in layer.lora_B)
        ]
        if not adapters:
            return result

        n = len(adapters)
        w = 1.0 / n

        lora_A = layer.lora_A[adapters[0]].weight.clone().mul_(w)
        lora_B = layer.lora_B[adapters[0]].weight.clone().mul_(w)

        for a in adapters[1:]:
            lora_A.add_(layer.lora_A[a].weight, alpha=w)
            lora_B.add_(layer.lora_B[a].weight, alpha=w)

        first = adapters[0]
        dropout = layer.lora_dropout[first]
        scaling = layer.scaling[first]

        x_cast = layer._cast_input_dtype(x, lora_A.dtype).contiguous()
        lora_A_out = torch.nn.functional.linear(dropout(x_cast), lora_A)
        lora_B_out = torch.nn.functional.linear(lora_A_out, lora_B)

        return result + lora_B_out * scaling

class AdapterSoup(WeightComposition):
    def __call__(self, layer: LoraLayer, base_out: Tensor, x: Tensor):
        result = base_out

        adapters = [
            a for a in layer.active_adapters
            if (a in layer.lora_A) and (a in layer.lora_B)
        ]
        if not adapters:
            return result

        n = len(adapters)
        inv_n = 1.0 / n

        A_bar = layer.lora_A[adapters[0]].weight.clone().mul_(inv_n)
        B_bar = layer.lora_B[adapters[0]].weight.clone().mul_(inv_n)
        s_bar = float(layer.scaling[adapters[0]]) * inv_n

        for a in adapters[1:]:
            A_bar.add_(layer.lora_A[a].weight, alpha=inv_n)
            B_bar.add_(layer.lora_B[a].weight, alpha=inv_n)
            s_bar += float(layer.scaling[a]) * inv_n

        first = adapters[0]
        dropout = layer.lora_dropout[first]

        x_cast = layer._cast_input_dtype(x, A_bar.dtype).contiguous()
        A_out = torch.nn.functional.linear(dropout(x_cast), A_bar)
        B_out = torch.nn.functional.linear(A_out, B_bar)

        return result + B_out * s_bar
    
class LogitComposition(Composition):
    def make_forward(self) -> Callable:
        def forward(model: PeftMixedModel, *args, **kwargs):
            adapters = model.active_adapters()
            outs = []
            template_past = None
            if kwargs.get("past_key_values") is not None:
                # KV cache objects are mutable; reusing them across adapter calls
                # corrupts attention shapes during generation.
                template_past = copy.deepcopy(kwargs["past_key_values"])
            for idx, adapter in enumerate(adapters):
                model.set_adapter(adapter)
                adapter_kwargs = dict(kwargs)
                if idx > 0 and template_past is not None:
                    adapter_kwargs["past_key_values"] = copy.deepcopy(template_past)
                outs.append(model._orig_forward(*args, **adapter_kwargs))
            model.set_adapter(adapters)

            logits = [o.logits for o in outs]
            base_out = outs[0]
            base_out.logits = self(logits)  
            return base_out
        return forward     

class LogitSum(LogitComposition):
    def __call__(self, logits: list):
        acc = logits[0].clone()
        for t in logits[1:]:
            acc.add_(t)
        return acc

        return torch.stack(logits, dim=0).sum(dim=0)
    
class LogitMax(LogitComposition):
    def __call__(self, logits: list):
        acc = logits[0].clone()
        for t in logits[1:]:
            torch.maximum(acc, t, out=acc)
        return acc
    
        return torch.stack(logits, dim=0).max(dim=0)[0]
    
class LogitMean(LogitComposition):
    def __call__(self, logits: list):
        n = len(logits)
        acc = logits[0].clone()
        for t in logits[1:]:
            acc.add_(t)
        acc.div_(n)
        return acc

        return torch.stack(logits, dim=0).mean(dim=0)
