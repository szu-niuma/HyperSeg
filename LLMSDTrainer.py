from transformers import Trainer
from typing import Optional
import torch.nn as nn
import os
import torch
import transformers


# train_LLaMALoRA.py
def maybe_zero_3(param):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


# hugging face model for hugging face trainer
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ["mm_projector"]
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(["embed_tokens", "embed_in"])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f"{current_folder}.bin"))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


# remove "module"
def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).
    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


class LLMSDTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        重写了 _save 方法，用于保存模型的不同组件
        分别保存两类权重：
            - 连接层权重 (weight_to_save)
            - LLM模型权重 (weight_to_save_LLM)
        """
        # Save the model
        _state_dict = state_dict
        if _state_dict is None:
            # Only save the model itself if we are using distributed training
            model_to_save = unwrap_model(self.model)
            _state_dict = model_to_save.state_dict()

        # Original checkpoint save: 'mm_projector' + 'llm_proj' + ('sd_query_tokens' + 'sd_qformer') + 'lm_head'
        weight_to_save = {}
        connections_keys_to_match = ["mm_projector", "lm_head"]
        for k, v in _state_dict.items():
            if any(key_match in k for key_match in connections_keys_to_match):
                # lm_head.weight torch.Size([32035, 4096])
                # mm_projector.weight torch.Size([4096, 1024]) + mm_projector.bias torch.Size([4096])
                print(k, v.size())
                weight_to_save[k] = v

        # LLM training... -> LLM checkpoint save
        weight_to_save_LLM = {}
        LLM_keys_to_match = ["model.base_model.model"]
        for k, v in _state_dict.items():
            if any(key_match in k for key_match in LLM_keys_to_match):
                weight_to_save_LLM[k] = v

        # len(weight_to_save.keys())=133, len(weight_to_save_unet.keys())=686, len(weight_to_save_LLM.keys())=450
        # checkpoint saving -> save_steps + training_finish
        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        if current_folder.startswith("checkpoint-"):
            current_step = int(current_folder[len("checkpoint-") :])
            # Original checkpoint save
            mm_projector_folder = os.path.join(parent_folder, "embeddings_qformer")
            os.makedirs(mm_projector_folder, exist_ok=True)
            torch.save(weight_to_save, os.path.join(mm_projector_folder, f"{current_folder}_embeddings_qformer.bin"))

            # LLM checkpoint save
            LLM_folder = os.path.join(parent_folder, "LLM-%d" % current_step)
            os.makedirs(LLM_folder, exist_ok=True)
            torch.save(weight_to_save_LLM, os.path.join(LLM_folder, "adapter_model.bin"))

            # optimizer and scheduler...
            now_folder = parent_folder + "/" + current_folder
            os.makedirs(now_folder, exist_ok=True)
        else:
            # Original checkpoint save
            mm_projector_folder = os.path.join(output_dir, "embeddings_qformer")
            os.makedirs(mm_projector_folder, exist_ok=True)
            torch.save(weight_to_save, os.path.join(mm_projector_folder, "checkpoint-last_embeddings_qformer.bin"))
            # LLM checkpoint save
            LLM_folder = os.path.join(output_dir, "LLM-last")
            os.makedirs(LLM_folder, exist_ok=True)
            torch.save(weight_to_save_LLM, os.path.join(LLM_folder, "adapter_model.bin"))
