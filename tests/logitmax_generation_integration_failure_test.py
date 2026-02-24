from fusekit import Modeling, Datasets
from fusekit.Common import env, Memory
from fusekit.Common.EvalType import EvalType

import random
import numpy as np
import torch

random.seed(0)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

import warnings
import shutil

warnings.filterwarnings("ignore", category=UserWarning)


class FixedLenGenerationDataset(Datasets.IterableDataset):
    def __init__(self, tokenizer, target_input_len=47, data_limit=1, max_new_tokens=8):
        super().__init__()
        self.type = EvalType.SIMILARITY
        self.name = "LogitMaxGenerationDataset"

        prompt = self.find_prompt_for_length(tokenizer, target_input_len, max_new_tokens=max_new_tokens)
        self.samples = [
            Datasets.GenerationSample(
                tokenizer=tokenizer,
                prompt=prompt,
                answer="ok",
                max_new_tokens=max_new_tokens,
                uid=i,
            )
            for i in range(data_limit)
        ]

    @staticmethod
    def find_prompt_for_length(tokenizer, target_input_len, max_new_tokens=8):
        prompt = ""
        for _ in range(512):
            prompt = (prompt + " x").strip()
            sample = Datasets.GenerationSample(
                tokenizer=tokenizer,
                prompt=prompt,
                answer="ok",
                max_new_tokens=max_new_tokens,
            )
            cur_len = int(sample.get_inputs().shape[1])
            if cur_len == target_input_len:
                return prompt
            if cur_len > target_input_len:
                break

        raise RuntimeError(
            f"Could not synthesize prompt with token length {target_input_len}. "
            "Adjust target_input_len for your tokenizer/model."
        )


def create_local_test_adapters(device=None, memory_limit=None):
    adapter_root = env.adapters / "logitmax_generation_integration_failure_test"
    adapter_paths = [
        adapter_root / "adapter_s1",
        adapter_root / "adapter_s2",
        adapter_root / "adapter_s3",
    ]

    if all((p / "adapter_config.json").exists() for p in adapter_paths):
        return adapter_root, adapter_paths, False

    adapter_root.mkdir(parents=True, exist_ok=True)

    model = Modeling.Llama2_7b(device=device, memory_limit=memory_limit)
    model = model.init_lora(rank=8, alpha=32, dropout=0.1)
    model.model.save_pretrained(adapter_paths[0])
    model.to("cpu")
    Memory.clear_cuda(verbose=False)

    shutil.copytree(adapter_paths[0], adapter_paths[1])
    shutil.copytree(adapter_paths[0], adapter_paths[2])

    return adapter_root, adapter_paths, True


def logitmax_generation_integration_failure_test(device=None, memory_limit=None):
    adapter_root, adapter_paths, created = create_local_test_adapters(
        device=device, memory_limit=memory_limit
    )

    model = Modeling.Llama2_7b(device=device, memory_limit=memory_limit)
    dataset = FixedLenGenerationDataset(
        model.tokenizer, target_input_len=47, data_limit=10, max_new_tokens=8
    )

    print("\n\n=== LogitMax Generation Repro ===")
    model.load_adapters(adapter_paths, Modeling.LogitMax())
    metrics, _ = model.evaluate(dataset)
    if metrics:
        print(metrics)


Memory.clear_cuda()
print("LogitMax Generation Integration Failure Test")
logitmax_generation_integration_failure_test()
Memory.clear_cuda()
