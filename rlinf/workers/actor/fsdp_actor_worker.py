# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
from contextlib import contextmanager
from functools import partial
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.distributed.tensor import DTensor
from torch.multiprocessing.reductions import reduce_tensor
from torch.utils._pytree import tree_map

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.algorithms.utils import (
    kl_penalty,
)
from rlinf.config import SupportedModel, torch_dtype_from_precision
from rlinf.data.embodied_io_struct import Trajectory, convert_trajectories_to_batch
from rlinf.data.io_struct import BatchResizingIterator, RolloutResult
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import (
    FSDPModelManager,
)
from rlinf.hybrid_engines.fsdp.utils import (
    pack_fsdp_input,
    prepare_pack_fsdp,
    unpack_fsdp_logprobs,
    unpack_sequences,
)
from rlinf.hybrid_engines.weight_syncer import WeightSyncer
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.data_iter_utils import (
    get_iterator_k_split,
    get_reverse_idx,
    get_seqlen_balanced_partitions,
    split_dynamic_batch_size,
)
from rlinf.utils.distributed import (
    RolloutDataBalance,
    all_reduce_dict,
    all_reduce_int,
    masked_normalization,
)
from rlinf.utils.distributed import (
    compute_rollout_metrics as compute_math_rollout_metrics,
)
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_loss_mask,
    compute_rollout_metrics,
    compute_split_num,
)
from rlinf.utils.nested_dict_process import (
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.utils.placement import (
    HybridComponentPlacement,
    ModelParallelComponentPlacement,
)
from rlinf.utils.pytree import register_pytree_dataclasses
from rlinf.utils.utils import (
    clear_memory,
    compute_entropy_from_logits,
    compute_logprobs_from_logits,
    cpu_weight_swap,
    get_loss_agg_func,
    masked_mean,
    reshape_entropy,
    retrieve_model_state_dict_in_cpu,
)
from rlinf.workers.rollout.utils import RankMapper


def process_nested_dict_for_adv(nested_dict, rollout_epoch):
    """
    original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
    target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
    """
    ret_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, torch.Tensor):
            new_value = value.reshape(
                rollout_epoch, -1, *value.shape[1:]
            )  # [rollout_epoch, n_chunk_step, bsz, ...]
            new_value = new_value.transpose(
                0, 1
            )  # [n_chunk_step, rollout_epoch, bsz, ...]
            new_value = new_value.reshape(new_value.shape[0], -1, *new_value.shape[3:])
            ret_dict[key] = new_value
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_adv(value, rollout_epoch)
    return ret_dict


def process_nested_dict_for_train(nested_dict, shuffle_id):
    ret_dict = {}
    for key, value in nested_dict.items():
        if key in ["dones", "terminations", "truncations", "prev_values"]:
            value = value[:-1]
        if "env_info" in key:
            raise NotImplementedError
        if value is None:
            ret_dict[key] = None
        if isinstance(value, torch.Tensor):
            ret_dict[key] = value.reshape(-1, *value.shape[2:])[shuffle_id]
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_train(value, shuffle_id)
    return ret_dict


class FSDPActor(FSDPModelManager, Worker):
    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        cfg_fsdp: Optional[DictConfig] = None,
    ) -> None:
        """
        FSDPActor worker used to train the model with data from rollout workers.

        Args:
            cfg (DictConfig): The global yaml configuration.
            placement (ModelParallelComponentPlacement): The accelerator placement for actor worker.
        """
        if cfg_fsdp is None:
            cfg_fsdp = cfg.actor
        Worker.__init__(self)
        super().__init__(cfg_fsdp, self._world_size, self._rank)

        self.cfg = cfg

        self.response_len = (
            cfg.actor.model.encoder_seq_length - cfg.data.max_prompt_length
        )
        self.calculate_entropy = cfg.algorithm.calculate_entropy
        self.calculate_entropy_loss = (
            cfg.algorithm.entropy_bonus > 0 and self.calculate_entropy
        )
        self.kl_beta = cfg.algorithm.kl_beta
        self.kl_penalty_type = cfg.algorithm.kl_penalty_type
        self.reinpp_kl_beta = cfg.algorithm.get("reinpp_kl_beta", 0.0)
        self.combine_reference_model = cfg.actor.get("combine_reference_model", True)

        self.total_batch_size_per_dp = (
            cfg.data.rollout_batch_size * cfg.algorithm.group_size // self._world_size
        )

        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = placement
        self.is_pipeline = self._component_placement.is_disaggregated
        self.ref_policy_state_dict = None
        if self.is_pipeline:
            self._inference_group_name = cfg.inference.group_name
            self._inference_world_size = self._component_placement.get_world_size(
                "inference"
            )
            self._inference_dst_map: dict[int, list[str]] = {}
        else:
            self._inference_group_name = None
            self._inference_world_size = 0
            self._inference_dst_map = None
        self.loss_agg_func = get_loss_agg_func(cfg.algorithm.loss_agg_func)
        self.enable_offload = not self.is_pipeline and cfg.actor.get(
            "enable_offload", False
        )
        self.micro_batch_size = cfg.actor.micro_batch_size
        self.n_mini_batches = cfg.algorithm.n_minibatches
        self.task_type = cfg.runner.task_type
        self.entropy_op_type = cfg.algorithm.get("entropy_op_type", "flash_attn")
        self.enable_dp_load_balance = cfg.actor.get("enable_dp_load_balance", False)
        self.lr_sched_sync_with_optim = cfg.actor.get("lr_sched_sync_with_optim", True)
        self.enable_dynamic_batch_size = cfg.runner.get(
            "enable_dynamic_batch_size", False
        )
        if self.is_pipeline:
            assert not self.enable_dp_load_balance, (
                "DP load balance is not supported in pipeline mode."
            )
            assert not self.enable_dynamic_batch_size, (
                "Dynamic batch size is not supported in pipeline mode."
            )
        self.max_tokens_per_mbs = cfg.runner.get("max_tokens_per_mbs", 2048)

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend
        (FSDP/FSDP2) to wrap it. If needed, offload model parameters and optimizer states to CPU.
        If kl_beta > 0, retrieve the reference policy model state dict to CPU.
        If mode is disaggregated, setup which inference ranks it needs to sync weights to by
        doing a handshake with inference workers.
        """
        self.setup_model_and_optimizer()
        if (
            self.kl_beta > 0 or self.reinpp_kl_beta > 0
        ) and self.combine_reference_model:
            self.ref_policy_state_dict = retrieve_model_state_dict_in_cpu(self.model)
            self.offload_model_buffer = {}

        if self.enable_offload and not self.is_pipeline:
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()

    def _setup_rollout_weight_dst_ranks(self) -> None:
        """Setup destination ranks for token and weight communication."""
        rank_map = RankMapper.get_actor_rank_to_rollout_rank_map(
            self._component_placement
        )
        self._weight_dst_rank_in_rollout = rank_map[self._rank]
        self.log_info(
            f"Actor rank {self._rank} will send weights to {self._weight_dst_rank_in_rollout}"
        )

    def del_reshard_state_dict(self) -> None:
        """Just for interface compatibility with MegatronActor."""
        pass

    def sync_model_to_inference(self) -> None:
        """
        Sync the model's full state dict to the inference worker.
        The model state_dict is the reference of actor's model
        parameters(by setting cpu_offload=False).
        """
        if not self._inference_dst_map:
            self._strategy.setup_actor_sync_inference_ranks(self)

        if self.enable_offload and not self.is_optimizer_offloaded:
            self.offload_optimizer()

        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device, False)

        inference_state_dict = self.get_model_state_dict(
            cpu_offload=False, full_state_dict=False
        )
        # NOTE: we have already know which inference rank needs which params
        # by calling _strategy.setup_actor_sync_inference_ranks() to do handshake
        # with each inference rank. just send them accordingly.
        for rank, needed_params in self._inference_dst_map.items():
            sended_params = {}
            for name in needed_params:
                if name in inference_state_dict:
                    # mentioned again, no ShardedTensor here.
                    sended_params[name] = (
                        inference_state_dict[name].to_local()
                        if isinstance(inference_state_dict[name], DTensor)
                        else inference_state_dict[name]
                    )
            self.send(
                object=sended_params,
                dst_group_name=self._inference_group_name,
                dst_rank=rank,
                async_op=True,
            )

        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()

        torch.distributed.barrier()

    def sync_model_to_rollout(self):
        """
        Sync the model's full state dict to the rollout worker.
        """
        if self.enable_offload:
            if not self.is_optimizer_offloaded:
                self.offload_optimizer()

            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device, False)

        rollout_dtype = None
        if self._cfg.get("sync_precision", None) is not None:
            rollout_dtype = torch_dtype_from_precision(self._cfg.sync_precision)

        rollout_state_dict = self.get_model_state_dict(
            cpu_offload=False, full_state_dict=False
        )
        has_visual = any("visual." in k for k in rollout_state_dict.keys())
        model_bucket_list = self.divide_model_to_bucket(rollout_state_dict, has_visual)
        del rollout_state_dict
        send_handles = []
        buffer = {}
        for bucket_idx, model_bucket in enumerate(model_bucket_list):
            for k, v in model_bucket.items():
                if isinstance(v, DTensor):
                    v = v.full_tensor()
                if rollout_dtype is not None:
                    v = v.to(rollout_dtype)
                if not self.is_pipeline:
                    v = reduce_tensor(v)
                buffer[k] = v
            if bucket_idx == 0:
                buffer["bucket_length"] = len(model_bucket_list)

            for send_handle in send_handles:
                send_handle.wait()
            send_handles = []

            if not self.is_pipeline:
                send_handle = self.send(
                    buffer,
                    self._rollout_group_name,
                    self._weight_dst_rank_in_rollout,
                    async_op=True,
                )
                send_handles.append(send_handle)
            else:
                for rank in self._weight_dst_rank_in_rollout:
                    send_handle = self.send(
                        buffer,
                        self._rollout_group_name,
                        rank,
                        async_op=True,
                    )
                    send_handles.append(send_handle)
            buffer = {}

        for send_handle in send_handles:
            send_handle.wait()

        if self.enable_offload:
            assert not self.is_weight_offloaded, (
                "weight should be offloaded in sync_model_to_rollout"
            )
            self.offload_param_and_grad()

        clear_memory(sync=False)

    def get_batch(
        self, channel: Channel
    ) -> tuple[dict[str, torch.Tensor], RolloutResult]:
        result: RolloutResult = channel.get()

        batch = result.to_actor_batch(
            self.cfg.data.max_prompt_length,
            self.cfg.actor.model.encoder_seq_length,
            self.tokenizer.eos_token_id,
        )
        return batch, result

    def get_dynamic_batch_as_much(
        self,
        input_channel: Channel,
        min_result_len: int,
        max_result_len: int,
        cliped_results=[],
        unfinished_result=None,
    ):
        assert not input_channel.is_local
        rollout_results = cliped_results
        # get min_result_len
        while len(rollout_results) < min_result_len:
            if unfinished_result is not None:
                rollout_result: RolloutResult = unfinished_result.wait()
                unfinished_result = None
            else:
                rollout_result: RolloutResult = input_channel.get()
            rollout_results.append(rollout_result)

        # try to get result as much
        # get result in every 0.1s and do all reduce to get the min result between dp (result_len)
        # stop at: the min result between dp (result_len) is same as the last min result
        last_result_len = 0
        result_len = len(rollout_results)
        time_until = time.time() + 0.1
        while last_result_len < result_len:
            if len(rollout_results) < max_result_len:
                if unfinished_result is None:
                    unfinished_result = input_channel.get(async_op=True)
                else:
                    time.sleep(0.001)
                if unfinished_result.done():
                    rollout_results.append(unfinished_result.wait())
                    unfinished_result = None
                if time.time() >= time_until:
                    last_result_len = result_len
                    result_len = all_reduce_int(len(rollout_results))
                    if last_result_len < result_len:
                        time_until = time.time() + 0.1
            else:
                last_result_len = result_len
                result_len = all_reduce_int(len(rollout_results))

        cliped_results = list(rollout_results[result_len:])
        rollout_results = rollout_results[:result_len]

        batches = []
        for rollout_result in rollout_results:
            batch = rollout_result.to_actor_batch(
                self.cfg.data.max_prompt_length,
                self.cfg.actor.model.encoder_seq_length,
                self.tokenizer.eos_token_id,
            )
            batches.append(batch)

        batch = RolloutResult.merge_batches(batches)
        rollout_result = RolloutResult.merge_result_list(rollout_results)
        return batch, rollout_result, result_len, cliped_results, unfinished_result

    @staticmethod
    def _split_to_micro_batch(
        batch,
        enable_dynamic_batch_size: bool,
        *,
        max_tokens_per_mbs: Optional[int] = None,
        split_num,
    ):
        if enable_dynamic_batch_size:
            (
                micro_batches_iter,
                _,
                micro_batch_cnt,
                dbs_indices,
            ) = split_dynamic_batch_size(
                batch=batch,
                cp_world_size=1,
                vpp_world_size=1,
                max_tokens_per_mbs=max_tokens_per_mbs,
                microbatch_group_size_per_vp_stage=1,
            )
        else:
            micro_batch_cnt = split_num
            micro_batches_iter = get_iterator_k_split(batch, micro_batch_cnt)
            dbs_indices = None
        return micro_batches_iter, micro_batch_cnt, dbs_indices

    def _load_weight_and_optimizer(self) -> None:
        # Acquire the GPUs to ensure that no one is using them before loading models
        # Otherwise, it may lead to OOM
        with self.device_lock:
            if not self.enable_offload:
                return
            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device)
            if self.is_optimizer_offloaded:
                self.load_optimizer(self.device)

    def compute_logprobs(self, logits, target):
        return compute_logprobs_from_logits(
            logits,
            target,
            op_type=self.entropy_op_type,
        )

    def forward_batch(
        self, m_batch: dict[str, torch.Tensor], calculate_entropy: bool = False
    ) -> torch.Tensor:
        input_ids = m_batch["input_ids"]
        attention_mask = m_batch["attention_mask"]
        position_ids = m_batch["position_ids"]

        multi_modal_inputs = {}
        if "multi_modal_inputs" in m_batch.keys():
            for key in m_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in m_batch["multi_modal_inputs"]],
                    dim=0,
                ).to(Worker.torch_device_type)

        if self.enable_dynamic_batch_size:
            max_seq_len_pack = self.max_tokens_per_mbs
            max_seq_len_unpack = self.cfg.actor.model.encoder_seq_length
            max_prompt_len = self.cfg.data.max_prompt_length
            max_response_len = max_seq_len_unpack - max_prompt_len
            idx_starts, idx_ends = prepare_pack_fsdp(m_batch, max_prompt_len)

            input_ids, position_ids, attention_mask = pack_fsdp_input(
                input_ids,
                position_ids,
                idx_starts=idx_starts,
                idx_ends=idx_ends,
                max_seq_len_pack=max_seq_len_pack,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        with self.amp_context:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                **multi_modal_inputs,
            )

        logits: torch.Tensor = outputs.logits
        logits.div_(self.cfg.algorithm.sampling_params.temperature)
        if self.enable_dynamic_batch_size:
            logprobs = unpack_fsdp_logprobs(
                logits,
                input_ids,
                idx_starts=idx_starts,
                idx_ends=idx_ends,
                max_seq_len_unpack=max_seq_len_unpack,
                eos_token_id=self.tokenizer.eos_token_id,
                compute_logprobs_fn=self.compute_logprobs,
            )
            logprobs = logprobs[:, -max_response_len:]
        else:
            # (bsz, response_length, vocab_size)
            logits = logits[:, -self.response_len - 1 : -1, :]
            responses = input_ids[:, -self.response_len :]
            logprobs = self.compute_logprobs(logits, responses)
        if calculate_entropy:
            entropy = compute_entropy_from_logits(logits)
            if self.enable_dynamic_batch_size:
                entropy = unpack_sequences(
                    entropy, idx_starts, idx_ends, max_seq_len_unpack, pad_val=0
                )[:, -self.response_len :]
            return logprobs, entropy
        return logprobs

    def inference_step(
        self,
        batch: dict[str, torch.Tensor],
        rollout_result: RolloutResult,
        compute_ref_logprobs: bool,
    ):
        micro_batches_iter, _, dbs_indices = self._split_to_micro_batch(
            batch,
            self.enable_dynamic_batch_size,
            max_tokens_per_mbs=self.max_tokens_per_mbs,
            split_num=rollout_result.num_sequence
            // self.cfg.algorithm.logprob_forward_micro_batch_size,
        )
        if self.enable_dynamic_batch_size:
            indices = sum(dbs_indices, [])
            revert_indices = torch.tensor(
                get_reverse_idx(indices),
                dtype=torch.long,
            )
        micro_batches = list(micro_batches_iter)

        prev_logprobs, ref_logprobs = None, None

        # Prev logprobs
        prev_logprobs = torch.cat(
            [self.forward_batch(batch) for batch in micro_batches]
        ).cpu()

        if self.enable_dynamic_batch_size:
            assert len(indices) == prev_logprobs.size(0), (
                f"Dynamic batch size indices length {len(indices)} does not equal "
                f"output length {prev_logprobs.size(0)}"
            )
            prev_logprobs = prev_logprobs[revert_indices]

        # Ref logprobs
        if compute_ref_logprobs:
            assert self.ref_policy_state_dict is not None, (
                "Reference policy state dict is None but compute_ref_logprobs is True"
            )
            with cpu_weight_swap(
                self.model,
                self.ref_policy_state_dict,
                self.offload_model_buffer,
            ):
                ref_logprobs = torch.cat(
                    [self.forward_batch(batch) for batch in micro_batches]
                ).cpu()

                if self.enable_dynamic_batch_size:
                    assert len(indices) == ref_logprobs.size(0), (
                        f"Dynamic batch size indices length {len(indices)} does not equal "
                        f"output length {ref_logprobs.size(0)}"
                    )
                    ref_logprobs = ref_logprobs[revert_indices]

        return prev_logprobs, ref_logprobs

    def run_inference(
        self,
        input_channel: Channel,
        output_channel: Channel,
        compute_ref_logprobs: bool,
        do_offload=False,
    ):
        """
        Compute prev/ref logprobs using the actor Model's forward.

        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
            compute_ref_logprobs: Whether to compute reference logprobs.
            do_offload: Whether offload weights after inference is done
        """
        assert not do_offload, (
            "do_offload argument of run_inference/run_training is not supported in FSDP for now"
        )

        inference_split = self.cfg.actor.get("inference_split", None)
        if inference_split is None:
            if not self.is_pipeline:
                inference_split = 1
            else:
                inference_split = self.cfg.algorithm.n_minibatches
        assert self.total_batch_size_per_dp % inference_split == 0, (
            f"FSDPActor: total_batch_size_per_dp[{self.total_batch_size_per_dp}] should be divisible by inference_split[{inference_split}]"
        )

        min_result_len = 1
        max_result_len = (
            self.cfg.data.rollout_batch_size // self._world_size // inference_split
        )
        if not self.is_pipeline:
            min_result_len = max_result_len
            coll_rollout_results = []
        total_result_len = 0
        total_result_len_per_dp = self.cfg.data.rollout_batch_size // self._world_size
        cliped_results, unfinished_result = [], None
        while total_result_len < total_result_len_per_dp:
            batch, rollout_result, result_len, cliped_results, unfinished_result = (
                self.get_dynamic_batch_as_much(
                    input_channel,
                    min(min_result_len, total_result_len_per_dp - total_result_len),
                    min(max_result_len, total_result_len_per_dp - total_result_len),
                    cliped_results,
                    unfinished_result,
                )
            )
            total_result_len += result_len
            self.log_debug(
                f"[dynamic inference rank-{self._rank}] inference result_len={result_len}, total_result_len={total_result_len}/{total_result_len_per_dp}"
            )
            self._load_weight_and_optimizer()
            self.model.eval()

            with self.worker_timer():
                with torch.no_grad():
                    prev_logprobs, ref_logprobs = self.inference_step(
                        batch, rollout_result, compute_ref_logprobs
                    )

                if rollout_result.rollout_logprobs is not None:
                    # Rollout has returned logprobs, store the recomputed logprobs in recompute_prev_logprobs
                    rollout_result.recompute_prev_logprobs = prev_logprobs
                else:
                    # Otherwise, directly store the logprobs in prev_logprobs (the final logprobs used for training)
                    rollout_result.prev_logprobs = prev_logprobs

                # Ref logprobs
                if compute_ref_logprobs:
                    rollout_result.ref_logprobs = ref_logprobs

            if self.is_pipeline:
                # for pipeline mode, send after inference to reduce latency.
                # should do split to ensure actor won't get too much batches.
                split_results = RolloutResult.split_results(rollout_result, result_len)
                for split_result in split_results:
                    output_channel.put(split_result, async_op=True)
            else:
                coll_rollout_results.append(rollout_result)

        if not self.is_pipeline:
            # for coll mode, merge results to reduce send time.
            rollout_result = RolloutResult.merge_result_list(coll_rollout_results)
            split_results = RolloutResult.split_results(
                rollout_result,
                min(total_result_len, self.cfg.algorithm.n_minibatches),
            )
            for split_result in split_results:
                output_channel.put(split_result)
        assert total_result_len == total_result_len_per_dp, (
            f"Expected {total_result_len_per_dp} sequences from channel, but got {total_result_len}"
        )

    def training_step(
        self, batch: dict[str, torch.Tensor] | BatchResizingIterator
    ) -> tuple[dict[str, torch.Tensor], float, list[float]]:
        if isinstance(batch, dict):
            global_batch_size = batch["input_ids"].shape[0]
            assert global_batch_size % self.micro_batch_size == 0, (
                f"global batch size {global_batch_size} can not divide micro_batch_size {self.micro_batch_size}"
            )
            micro_batches_iter, micro_batch_cnt, _ = self._split_to_micro_batch(
                batch,
                self.enable_dynamic_batch_size,
                max_tokens_per_mbs=self.max_tokens_per_mbs,
                split_num=global_batch_size // self.micro_batch_size,
            )
            self.gradient_accumulation = micro_batch_cnt
        else:
            global_batch_size = self.total_batch_size_per_dp // self.n_mini_batches
            micro_batch_cnt = global_batch_size // self.micro_batch_size
            self.gradient_accumulation = micro_batch_cnt

            def iterator_wrapper():
                for _ in range(micro_batch_cnt):
                    yield next(batch)

            micro_batches_iter = iterator_wrapper()
        self.optimizer.zero_grad()
        mbs_metrics_list = {}
        for idx, m_batch in enumerate(micro_batches_iter):
            backward_ctx = self.before_micro_batch(
                self.model,
                is_last_micro_batch=(idx + 1) == micro_batch_cnt,
            )
            for k, v in m_batch.items():
                m_batch[k] = (
                    v.to(Worker.torch_device_type) if isinstance(v, torch.Tensor) else v
                )

            # batch for forward
            logprobs, entropy = self.forward_batch(m_batch, True)

            # batch for backward
            prev_logprobs = m_batch["prev_logprobs"]
            advantages = m_batch["advantages"]
            ref_logprobs = None
            if "ref_logprobs" in m_batch:
                ref_logprobs = m_batch["ref_logprobs"]

            loss_mask = m_batch["response_mask"][:, -self.response_len :]

            clip_ratio = self.cfg.algorithm.ratio_clip_eps
            clip_ratio_low = self.cfg.algorithm.get("clip_ratio_low", None)
            clip_ratio_high = self.cfg.algorithm.get("clip_ratio_high", None)
            clip_ratio_low = (
                clip_ratio_low if clip_ratio_low is not None else clip_ratio
            )
            clip_ratio_high = (
                clip_ratio_high if clip_ratio_high is not None else clip_ratio
            )
            clip_ratio_c = self.cfg.algorithm.get("clip_ratio_c", 3.0)

            if self.cfg.algorithm.get("importance_sampling_fix", False):
                rollout_prev_logprobs = prev_logprobs
                recompute_prev_logprobs = m_batch["recompute_prev_logprobs"]
                advantages = advantages * torch.clamp(
                    (recompute_prev_logprobs - rollout_prev_logprobs).exp(),
                    min=self.cfg.algorithm.importance_sampling_clip,
                )

            loss, mbs_metrics_data = policy_loss(
                task_type=self.task_type,
                loss_type=self.cfg.algorithm.loss_type,
                loss_agg_func=self.loss_agg_func,
                logprobs=logprobs,
                old_logprobs=prev_logprobs,
                advantages=advantages,
                clip_ratio_c=clip_ratio_c,
                clip_ratio_low=clip_ratio_low,
                clip_ratio_high=clip_ratio_high,
                loss_mask=loss_mask,
                clip_log_ratio_min=self.cfg.algorithm.get("clip_log_ratio_min", None),
                clip_log_ratio_max=self.cfg.algorithm.get("clip_log_ratio_max", None),
                fast_path_zero_loss_mask=True,
            )

            entropy_loss = torch.tensor(
                0.0, device=Worker.torch_platform.current_device()
            )
            if self.calculate_entropy:
                entropy_loss = self.loss_agg_func(entropy, mask=loss_mask)
                if self.calculate_entropy_loss:
                    loss = loss - self.cfg.algorithm.entropy_bonus * entropy_loss

            kl_loss = torch.tensor(0.0, device=Worker.torch_platform.current_device())
            if self.kl_beta > 0 and ref_logprobs is not None:
                kld = kl_penalty(ref_logprobs, logprobs, self.kl_penalty_type)
                kl_loss = self.loss_agg_func(kld, loss_mask)
                loss = loss + kl_loss * self.kl_beta

            # add to log
            # scale loss for gradient accumulation and backprop
            final_loss_metric = loss.detach()
            loss = loss / self.gradient_accumulation
            with backward_ctx:
                self.grad_scaler.scale(loss).backward()

            mbs_metrics_data.update(
                {
                    "actor/final_loss": final_loss_metric,
                    "actor/entropy_loss": entropy_loss.detach(),
                    "actor/kl_loss": kl_loss.detach(),
                }
            )

            append_to_dict(mbs_metrics_list, mbs_metrics_data)

        grad_norm, lr_list = self.optimizer_step()

        if self.lr_sched_sync_with_optim:
            self.lr_scheduler.step()

        # aggregate metrics across micro-batches
        mean_metric_dict = {
            key: torch.mean(torch.stack(value))
            for key, value in mbs_metrics_list.items()
        }
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        mean_metric_dict["actor/grad_norm"] = float(grad_norm)
        mean_metric_dict["actor/lr"] = lr_list[0]
        return mean_metric_dict

    def run_training_pipeline(self, input_channel: Channel) -> tuple[dict, list]:
        self.model.train()
        train_batch_iterator = BatchResizingIterator(
            cfg=self.cfg,
            get_batch_fn=partial(self.get_batch, input_channel),
            micro_batch_size=self.micro_batch_size,
            total_batch_size=self.total_batch_size_per_dp,
            num_global_batches=self.n_mini_batches,
            forward_only=False,
        )
        train_batch_iterator.register_get_batch_handler(
            self.compute_advantages_and_returns
        )

        if self.cfg.algorithm.normalize_advantages:

            def normalize_advantages(batch: dict[str, torch.Tensor]):
                mask = batch["response_mask"][:, -self.response_len :]
                batch["advantages"] = masked_normalization(batch["advantages"], mask)
                return batch

            train_batch_iterator.register_global_batch_handler(normalize_advantages)

        self._load_weight_and_optimizer()
        training_metrics_list = []
        with self.worker_timer("run_training"):
            for _ in range(self.n_mini_batches):
                mean_metric_dict = self.training_step(batch=train_batch_iterator)
                training_metrics_list.append(mean_metric_dict)
            if not self.lr_sched_sync_with_optim:
                self.lr_scheduler.step()

        # Rollout metrics
        batch = train_batch_iterator.get_all_batches()
        rollout_metrics, _, _ = compute_math_rollout_metrics(
            batch, self.cfg.data.max_prompt_length, self.response_len
        )

        return rollout_metrics, training_metrics_list

    def _dp_load_balance(self, batch: dict[str, torch.Tensor]):
        batch_size = batch["input_ids"].shape[0]
        assert batch_size == self.total_batch_size_per_dp, (
            f"DP Load balance is only available when a single batch contains all data, e.g., in collocated mode. But got {batch_size=} and {self.total_batch_size_per_dp=}."
        )
        batch = RolloutDataBalance.from_rollout_batches(
            rollout_batches=batch,
            dp_world_size=torch.distributed.get_world_size(),
            dp_rank=torch.distributed.get_rank(),
            dp_group=torch.distributed.group.WORLD,
            partitioning_tool=get_seqlen_balanced_partitions,
        )
        return batch

    def run_training(
        self, input_channel: Channel, do_offload=False
    ) -> tuple[dict, list]:
        # Get all batches for this DP
        assert not do_offload, (
            "do_offload argument of run_inference/run_training is not supported in FSDP for now"
        )

        if self.is_pipeline:
            return self.run_training_pipeline(input_channel)

        batches = []
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            batch, rollout_result = self.get_batch(input_channel)
            batches.append(batch)
            recv_batch_size += rollout_result.num_sequence
        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )
        global_batch = RolloutResult.merge_batches(batches)

        # Compute advantages and returns
        global_batch = self.compute_advantages_and_returns(global_batch)

        if self.enable_dp_load_balance:
            global_batch = self._dp_load_balance(global_batch)

        if self.cfg.algorithm.normalize_advantages:
            mask = global_batch["response_mask"][:, -self.response_len :]
            global_batch["advantages"] = masked_normalization(
                global_batch["advantages"], mask
            )

        # Must be called after batch is retrieved, which is when rollout has stopped
        # Otherwise, loading model might cause OOM
        self._load_weight_and_optimizer()

        mini_batches = get_iterator_k_split(
            global_batch,
            num_splits=self.cfg.algorithm.n_minibatches,
            shuffle=self.cfg.algorithm.get("shuffle_rollout", True),
            shuffle_seed=self.cfg.actor.seed,
        )

        self.model.train()
        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )

        training_metrics_list = []
        # Global batch iterations
        with self.worker_timer():
            for mini_batch in mini_batches:
                mean_metric_dict = self.training_step(batch=mini_batch)
                training_metrics_list.append(mean_metric_dict)
            if not self.lr_sched_sync_with_optim:
                self.lr_scheduler.step()

        # Rollout metrics
        rollout_metrics, _, _ = compute_math_rollout_metrics(
            global_batch, self.cfg.data.max_prompt_length, self.response_len
        )

        return rollout_metrics, training_metrics_list

    # Advantages and returns
    def compute_advantages_and_returns(self, batch: dict[str, torch.Tensor]):
        """Compute the advantages and returns.

        Args:
            batch (Dict[str, torch.Tensor]): The rollout batch.
        """
        with self.worker_timer():
            if batch.get("advantages", None) is None:
                mask = batch["response_mask"][:, -self.response_len :]
                advantages, _ = calculate_adv_and_returns(
                    task_type=self.task_type,
                    adv_type=self.cfg.algorithm.adv_type,
                    rewards=batch["rewards"].to(Worker.torch_device_type),
                    loss_mask=mask.to(Worker.torch_device_type),
                    group_size=self.cfg.algorithm.group_size,
                    kl_beta=self.reinpp_kl_beta,
                    kl_penalty_type=self.kl_penalty_type,
                    logprob=batch["prev_logprobs"].to(Worker.torch_device_type)
                    if "prev_logprobs" in batch
                    else None,
                    ref_logprob=batch["ref_logprobs"].to(Worker.torch_device_type)
                    if "ref_logprobs" in batch
                    else None,
                    use_reinpp_baseline=self.cfg.algorithm.get(
                        "use_reinpp_baseline", False
                    ),
                )
                batch["advantages"] = advantages

        return batch


class EmbodiedFSDPActor(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)
        self.cfg = cfg
        self._env_group_name = cfg.env.group_name
        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        # stage_num: default to 2, use for pipeline rollout process
        self.stage_num = cfg.rollout.pipeline_stage_num

        self.enable_offload = self.cfg.actor.get("enable_offload", False)
        self.entropy_op_type = self.cfg.algorithm.get("entropy_op_type", "torch")

        self.enable_sft_co_train = cfg.actor.get("enable_sft_co_train", False)
        self.version = 0
        if self.enable_sft_co_train:
            self._build_sft_data_loader()

        # create weight syncer
        weight_syncer_cfg = OmegaConf.select(cfg, "weight_syncer")
        self.weight_syncer = WeightSyncer.create(weight_syncer_cfg)
        self._sync_weight_comm_options = self.weight_syncer.comm_options

        self._is_weight_sender = self._rank == 0
        self._actor_world_size = self._world_size
        self._rollout_all_ranks = list(
            range(self._component_placement.get_world_size("rollout"))
        )

    def _debug_weight_sync(self, message: str) -> None:
        line = (
            message
            if message.startswith("[DEBUG-WEIGHT-SYNC]")
            else f"[DEBUG-WEIGHT-SYNC] {message}"
        )
        print(line, flush=True)
        debug_paths = []
        repo_path = os.environ.get("REPO_PATH")
        if repo_path:
            debug_paths.append(os.path.join(repo_path, "debug_weight_sync.log"))
        try:
            log_dir = os.path.join(
                self.cfg.runner.logger.log_path,
                self.cfg.runner.logger.experiment_name,
            )
            os.makedirs(log_dir, exist_ok=True)
            debug_paths.append(os.path.join(log_dir, "debug_weight_sync.log"))
            for debug_path in debug_paths:
                os.makedirs(os.path.dirname(debug_path), exist_ok=True)
                with open(debug_path, "a", encoding="utf-8") as debug_file:
                    debug_file.write(line + "\n")
        except Exception as exc:
            print(f"[DEBUG-WEIGHT-SYNC] failed to write debug log: {exc}", flush=True)
        self.log_info(line)

    def _debug_weight_sync_marker(self, marker: str) -> None:
        line = (
            "[DEBUG-WEIGHT-SYNC-MARKER] "
            f"pid={os.getpid()} cwd={os.getcwd()} "
            f"rank={self._rank} version={self.version} {marker}\n"
        )
        try:
            debug_dir = os.environ.get("RLINF_DEBUG_SYNC_DIR", "/tmp")
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, "actor_sync_markers.log")
            with open(debug_path, "a", buffering=1) as f:
                f.write(line)
        except Exception as e:
            try:
                with open("actor_sync_markers.log", "a", buffering=1) as f:
                    f.write(
                        "[DEBUG-WEIGHT-SYNC-MARKER] "
                        f"fallback_write_error={repr(e)} "
                        f"pid={os.getpid()} cwd={os.getcwd()} "
                        f"rank={self._rank} version={self.version} {marker}\n"
                    )
            except Exception:
                pass

    def _debug_actor_stage_marker(self, marker: str) -> None:
        line = (
            "[DEBUG-ACTOR-STAGE-MARKER] "
            f"pid={os.getpid()} cwd={os.getcwd()} "
            f"rank={self._rank} version={self.version} {marker}\n"
        )
        try:
            debug_dir = os.environ.get("RLINF_DEBUG_SYNC_DIR", "/tmp")
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, "actor_stage_markers.log")
            with open(debug_path, "a", buffering=1) as f:
                f.write(line)
        except Exception as e:
            try:
                with open("actor_stage_markers.log", "a", buffering=1) as f:
                    f.write(
                        "[DEBUG-ACTOR-STAGE-MARKER] "
                        f"fallback_write_error={repr(e)} "
                        f"pid={os.getpid()} cwd={os.getcwd()} "
                        f"rank={self._rank} version={self.version} {marker}\n"
                    )
            except Exception:
                pass

    @contextmanager
    def _debug_inner_model_forward_markers(self, marker_prefix: str):
        target = getattr(self.model, "module", None)
        target_name = "model.module.forward"
        if not isinstance(target, nn.Module):
            target = self.model
            target_name = "model.forward"

        original_forward = getattr(target, "forward", None)
        if original_forward is None:
            self._debug_actor_stage_marker(
                f"{marker_prefix} unable to install inner forward marker "
                f"target={target_name}"
            )
            yield
            return

        target_cls = target.__class__.__name__

        def wrapped_forward(*args, **kwargs):
            self._debug_actor_stage_marker(
                f"{marker_prefix} inner forward enter "
                f"target={target_name} module={target_cls}"
            )
            try:
                return original_forward(*args, **kwargs)
            finally:
                self._debug_actor_stage_marker(
                    f"{marker_prefix} inner forward exit "
                    f"target={target_name} module={target_cls}"
                )

        self._debug_actor_stage_marker(
            f"{marker_prefix} install inner forward marker "
            f"target={target_name} module={target_cls}"
        )
        try:
            target.forward = wrapped_forward
            yield
        finally:
            target.forward = original_forward
            self._debug_actor_stage_marker(
                f"{marker_prefix} remove inner forward marker "
                f"target={target_name} module={target_cls}"
            )

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend,
        if needed, offload model parameters and optimizer states to CPU.
        """
        self.setup_model_and_optimizer()

        if self.enable_offload:
            self.offload_param_and_grad()
            self.offload_optimizer()

    def model_provider_func(self) -> nn.Module:
        model = get_model(self.cfg.actor.model)
        if model is None:
            model = super().model_provider_func()

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            model.load_state_dict(model_dict)

        return model

    def get_rollout_state_dict(self) -> dict:
        return self.get_model_state_dict(cpu_offload=False, full_state_dict=False)

    async def sync_model_to_rollout(self) -> None:
        try:
            debug_dir = os.environ.get("RLINF_DEBUG_SYNC_DIR", "/tmp")
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, "actor_sync_entered.log")
            with open(debug_path, "a", buffering=1) as f:
                f.write(
                    "[DEBUG-WEIGHT-SYNC-ENTRY] "
                    f"pid={os.getpid()} cwd={os.getcwd()} "
                    f"rank={self._rank} version={self.version} "
                    "entered sync_model_to_rollout\n"
                )
        except Exception as e:
            try:
                with open("actor_sync_entered.log", "a", buffering=1) as f:
                    f.write(
                        "[DEBUG-WEIGHT-SYNC-ENTRY] "
                        f"fallback_write_error={repr(e)} "
                        f"pid={os.getpid()} cwd={os.getcwd()} "
                        f"rank={self._rank} version={self.version} "
                        "entered sync_model_to_rollout\n"
                    )
            except Exception:
                pass
        self._debug_weight_sync_marker("before _debug_weight_sync enter")
        self._debug_weight_sync(
            f"[DEBUG-WEIGHT-SYNC] actor rank={self._rank} "
            f"version={self.version} enter sync_model_to_rollout"
        )
        self._debug_weight_sync_marker("after _debug_weight_sync enter")
        if self.enable_offload:
            if not self.is_optimizer_offloaded:
                self._debug_weight_sync(
                    f"[DEBUG-WEIGHT-SYNC] actor rank={self._rank} "
                    f"version={self.version} before offload_optimizer"
                )
                self._debug_weight_sync_marker("before offload_optimizer")
                self.offload_optimizer()
                self._debug_weight_sync_marker("after offload_optimizer")
                self._debug_weight_sync(
                    f"[DEBUG-WEIGHT-SYNC] actor rank={self._rank} "
                    f"version={self.version} after offload_optimizer"
                )

            if self.is_weight_offloaded:
                self._debug_weight_sync(
                    f"[DEBUG-WEIGHT-SYNC] actor rank={self._rank} "
                    f"version={self.version} before load_param_and_grad"
                )
                self._debug_weight_sync_marker("before load_param_and_grad")
                self.load_param_and_grad(self.device, False)
                self._debug_weight_sync_marker("after load_param_and_grad")
                self._debug_weight_sync(
                    f"[DEBUG-WEIGHT-SYNC] actor rank={self._rank} "
                    f"version={self.version} after load_param_and_grad"
                )

        self._debug_weight_sync(
            f"[DEBUG-WEIGHT-SYNC] actor rank={self._rank} "
            f"version={self.version} before get_rollout_state_dict"
        )
        self._debug_weight_sync_marker("before get_rollout_state_dict")
        state_dict = self.get_rollout_state_dict()
        self._debug_weight_sync_marker(f"after get_rollout_state_dict keys={len(state_dict)}")
        self._debug_weight_sync(
            f"[DEBUG-WEIGHT-SYNC] actor rank={self._rank} "
            f"version={self.version} after get_rollout_state_dict "
            f"keys={len(state_dict)}"
        )

        async def send_func(data):
            if not self._is_weight_sender:
                self._debug_weight_sync_marker("send_func skipped non-weight-sender")
                return
            self._debug_weight_sync_marker(
                f"send_func before broadcast data_type={type(data).__name__}"
            )
            await self.broadcast(
                data,
                groups=[
                    (self._group_name, 0),
                    (self._rollout_group_name, self._rollout_all_ranks),
                ],
                src=(self._group_name, 0),
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()
            self._debug_weight_sync_marker("send_func after broadcast")

        async def recv_func():
            if self._is_weight_sender:
                self._debug_weight_sync_marker(
                    "recv_func before recv metadata from rollout rank 0"
                )
                data = await self.recv(
                    src_group_name=self._rollout_group_name,
                    src_rank=0,
                    async_op=True,
                    options=self._sync_weight_comm_options,
                ).async_wait()
                self._debug_weight_sync_marker(
                    f"recv_func after recv metadata data_type={type(data).__name__}"
                )
            else:
                data = None
                self._debug_weight_sync_marker(
                    "recv_func waiting for metadata relay from actor rank 0"
                )

            self._debug_weight_sync_marker(
                "recv_func before actor metadata broadcast"
            )
            data = await self.broadcast(
                data,
                groups=[(self._group_name, list(range(self._actor_world_size)))],
                src=(self._group_name, 0),
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()
            self._debug_weight_sync_marker(
                f"recv_func after actor metadata broadcast data_type={type(data).__name__}"
            )
            return data

        if not self.weight_syncer.sender_initialized():
            self._debug_weight_sync(
                f"[DEBUG-WEIGHT-SYNC] actor rank={self._rank} "
                f"version={self.version} before weight_syncer.init_sender"
            )
            self._debug_weight_sync_marker("before weight_syncer.init_sender")
            await self.weight_syncer.init_sender(
                state_dict=state_dict,
                send=send_func,
                recv=recv_func,
                param_names_need_sync=self.param_names_need_sync,
            )
            self._debug_weight_sync_marker("after weight_syncer.init_sender")
            self._debug_weight_sync(
                f"[DEBUG-WEIGHT-SYNC] actor rank={self._rank} "
                f"version={self.version} after weight_syncer.init_sender"
            )

        self._debug_weight_sync(
            f"[DEBUG-WEIGHT-SYNC] actor rank={self._rank} "
            f"version={self.version} before weight_syncer.sync"
        )
        self._debug_weight_sync_marker("before weight_syncer.sync")
        await self.weight_syncer.sync(state_dict, send_func, version=self.version)
        self._debug_weight_sync_marker("after weight_syncer.sync")
        self._debug_weight_sync(
            f"[DEBUG-WEIGHT-SYNC] actor rank={self._rank} "
            f"version={self.version} after weight_syncer.sync"
        )

        if self.enable_offload:
            assert not self.is_weight_offloaded, (
                "weight should be offloaded in sync_model_to_rollout"
            )
            self._debug_weight_sync_marker("before final offload_param_and_grad")
            self.offload_param_and_grad(True)
            self._debug_weight_sync_marker("after final offload_param_and_grad")

    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        """
        Receive rollout trajectories from rollout workers.

        Args:
            input_channel: The input channel to read from.
        """
        clear_memory(sync=False)

        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []
        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        self.rollout_batch = convert_trajectories_to_batch(recv_list)

        self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)

    def _process_received_rollout_batch(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
        target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
        """
        rollout_epoch = self.cfg.algorithm.rollout_epoch
        rollout_batch = process_nested_dict_for_adv(rollout_batch, rollout_epoch)

        if (
            not self.cfg.env.train.auto_reset
            and not self.cfg.env.train.ignore_terminations
        ):
            dones = rollout_batch[
                "dones"
            ]  # [n_chunk_step, rollout_epoch x bsz, num_action_chunks]
            loss_mask, loss_mask_sum = compute_loss_mask(dones)

            if self.cfg.algorithm.reward_type == "chunk_level":
                loss_mask = loss_mask.any(dim=-1, keepdim=True)
                loss_mask_sum = loss_mask_sum[..., -1:]

            rollout_batch["loss_mask"] = loss_mask
            rollout_batch["loss_mask_sum"] = loss_mask_sum

        # filter data by rewards
        if self.cfg.algorithm.get("filter_rewards", False):
            rewards = rollout_batch[
                "rewards"
            ]  # [n_chunk_step, batch, num_action_chunks]
            if rollout_batch.get("loss_mask", None) is not None:
                rewards = rewards * rollout_batch["loss_mask"]
            n_chunk_step, batch_size, num_action_chunks = rewards.shape

            group_size = self.cfg.algorithm.group_size
            assert batch_size % group_size == 0, (
                f"batch {batch_size} not divisible by group_size {group_size}"
            )
            n_prompts = batch_size // group_size

            # calculate rewards by prompt
            rewards = rewards.transpose(
                0, 1
            )  # [batch, n_chunk_step, num_action_chunks]
            rewards = rewards.reshape(rewards.shape[0], -1)  # [batch, n_step]
            reward_matrix = rewards.reshape(
                n_prompts, group_size, rewards.shape[-1]
            )  # [n_prompts, group_size, n_step]
            reward_matrix = reward_matrix.sum(dim=-1)  # [n_prompts, group_size]
            mean_reward_in_group = reward_matrix.mean(dim=1)  # [n_prompts]

            # mask
            reward_filter_mask = (
                mean_reward_in_group >= self.cfg.algorithm.rewards_lower_bound
            ) & (
                mean_reward_in_group <= self.cfg.algorithm.rewards_upper_bound
            )  # [n_prompts]

            # extend mask dimension
            reward_filter_mask = reward_filter_mask.repeat_interleave(
                group_size
            )  # [batch]
            reward_filter_mask = (
                reward_filter_mask.unsqueeze(0).expand(n_chunk_step, -1).unsqueeze(-1)
            )  # [n_chunk_step, batch, 1]

            # update loss_mask
            if rollout_batch.get("loss_mask", None) is not None:
                rollout_batch["loss_mask"] = (
                    reward_filter_mask & rollout_batch["loss_mask"]
                )
            else:
                rollout_batch["loss_mask"] = reward_filter_mask

        return rollout_batch

    def _compute_local_rollout_metrics(self, data_buffer: dict) -> dict[str, float]:
        """Compute rollout logging metrics without distributed collectives."""
        rollout_metrics = {}

        if "rewards" in data_buffer:
            rewards = data_buffer["rewards"].detach().float()
            rollout_metrics["rewards"] = rewards.mean().cpu().item()

        if "advantages" in data_buffer:
            advantages = data_buffer["advantages"].detach().float()
            rollout_metrics.update(
                {
                    "advantages_mean": advantages.mean().cpu().item(),
                    "advantages_max": advantages.max().cpu().item(),
                    "advantages_min": advantages.min().cpu().item(),
                }
            )

        returns = data_buffer.get("returns", None)
        if returns is not None:
            returns = returns.detach().float()
            rollout_metrics.update(
                {
                    "returns_mean": returns.mean().cpu().item(),
                    "returns_max": returns.max().cpu().item(),
                    "returns_min": returns.min().cpu().item(),
                }
            )

        return rollout_metrics

    def _safe_local_metric_mean(self, values) -> float:
        """Average already-safe local metrics without forcing CUDA synchronization."""
        safe_values = []
        for value in values:
            if isinstance(value, torch.Tensor):
                if value.device.type == "cpu":
                    safe_values.append(float(value.detach().float().mean().item()))
                else:
                    safe_values.append(float("nan"))
            else:
                try:
                    safe_values.append(float(value))
                except (TypeError, ValueError):
                    safe_values.append(float("nan"))

        return float(np.nanmean(safe_values)) if safe_values else float("nan")

    @staticmethod
    def _parse_debug_flag(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _actor_debug_flag(
        self, cfg_name: str, env_name: str, default: bool = False
    ) -> bool:
        env_value = os.environ.get(env_name)
        if env_value is not None:
            return self._parse_debug_flag(env_value)
        return self._parse_debug_flag(self.cfg.actor.get(cfg_name, default))

    def _metric_mean_for_all_reduce(self, values) -> float:
        """Average metrics with synchronization allowed for opt-in diagnostics."""
        metric_values = []
        for value in values:
            if isinstance(value, torch.Tensor):
                metric_values.append(float(value.detach().float().mean().cpu().item()))
            else:
                try:
                    metric_values.append(float(value))
                except (TypeError, ValueError):
                    metric_values.append(float("nan"))

        return float(np.nanmean(metric_values)) if metric_values else float("nan")

    def _debug_cuda_sync_marker(self, label: str) -> None:
        self._debug_actor_stage_marker(f"before cuda synchronize {label}")
        try:
            if hasattr(self.torch_platform, "synchronize"):
                self.torch_platform.synchronize()
            elif Worker.torch_device_type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            elif Worker.torch_device_type == "xpu" and hasattr(torch, "xpu"):
                torch.xpu.synchronize()
            elif Worker.torch_device_type == "mps" and hasattr(torch, "mps"):
                torch.mps.synchronize()
        except Exception as exc:
            self._debug_actor_stage_marker(
                f"cuda synchronize {label} raised {type(exc).__name__}: {exc}"
            )
            raise
        self._debug_actor_stage_marker(f"after cuda synchronize {label}")

    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        """
        Compute the advantages and returns.
        """
        self._debug_actor_stage_marker("enter compute_advantages_and_returns")
        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": self.rollout_batch["rewards"],
            "dones": self.rollout_batch["dones"],
            "values": self.rollout_batch.get("prev_values", None),
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": self.rollout_batch.get("loss_mask", None),
            "loss_mask_sum": self.rollout_batch.get("loss_mask_sum", None),
        }

        rewards = kwargs["rewards"]
        dones = kwargs["dones"]
        values = kwargs["values"]
        self._debug_actor_stage_marker(
            "before calculate_adv_and_returns "
            f"rewards_shape={tuple(rewards.shape)} rewards_device={rewards.device} "
            f"dones_shape={tuple(dones.shape)} dones_device={dones.device} "
            f"values_shape={tuple(values.shape) if values is not None else None} "
            f"values_device={values.device if values is not None else None}"
        )
        advantages_and_returns = calculate_adv_and_returns(**kwargs)
        self._debug_actor_stage_marker(
            "after calculate_adv_and_returns "
            f"keys={list(advantages_and_returns.keys())}"
        )

        self.rollout_batch.update(advantages_and_returns)
        if kwargs["loss_mask"] is not None:
            self.rollout_batch.update({"loss_mask": kwargs["loss_mask"]})
        if kwargs["loss_mask_sum"] is not None:
            self.rollout_batch.update({"loss_mask_sum": kwargs["loss_mask_sum"]})

        self._debug_actor_stage_marker("before compute_local_rollout_metrics")
        rollout_metrics = self._compute_local_rollout_metrics(self.rollout_batch)
        self._debug_actor_stage_marker(
            f"after compute_local_rollout_metrics keys={list(rollout_metrics.keys())}"
        )
        self._debug_actor_stage_marker("exit compute_advantages_and_returns")
        return rollout_metrics

    def _build_sft_data_loader(self):
        if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.OPENPI]:
            # NOTE: This must be set before importing openpi.training.data_loader
            if self.cfg.actor.get("sft_data_path", None):
                os.environ["HF_LEROBOT_HOME"] = self.cfg.actor.sft_data_path

            import openpi.training.data_loader as _data

            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

            if "config_name" not in self.cfg.actor:
                raise ValueError(
                    "config_name is required when enable_sft_co_train=True"
                )
            training_config_name = self.cfg.actor.config_name
            data_loader_config = get_openpi_config(
                training_config_name,
                model_path=self.cfg.actor.model.model_path,
                data_kwargs=getattr(self.cfg.actor, "openpi_data", None),
            )
            self.data_loader = _data.create_data_loader(
                data_loader_config, framework="pytorch", shuffle=True
            )
            self.sft_iterator = iter(self.data_loader)
            self.train_epoch = 0
            self.sft_loss_weight = self.cfg.actor.get("sft_loss_weight", 0.1)
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def _train_sft_epoch(
        self,
        metrics_data: dict[str, torch.Tensor],
        loss: torch.Tensor,
        log_fragile_scalar_metrics: bool = False,
    ):
        """
        Train one epoch of SFT.
        """
        ppo_loss = loss.detach()
        metrics_data["ppo_loss"] = (
            ppo_loss if log_fragile_scalar_metrics else float("nan")
        )

        # Get next data batch
        try:
            observation, actions = next(self.sft_iterator)
        except StopIteration:
            self.train_epoch += 1
            self.data_loader.set_epoch(self.train_epoch)
            self.sft_iterator = iter(self.data_loader)
            observation, actions = next(self.sft_iterator)

        register_pytree_dataclasses(observation)
        observation = tree_map(
            lambda x: x.to(self.device) if x is not None else x,
            observation,
        )
        actions = actions.to(torch.float32)
        actions = actions.to(self.device)

        sft_losses = self.model(
            data={"observation": observation, "actions": actions},
            forward_type=ForwardType.SFT,
        )
        # Ensure losses is a tensor and handle different return types
        if isinstance(sft_losses, list | tuple):
            sft_losses = torch.stack(sft_losses)
        elif not isinstance(sft_losses, torch.Tensor):
            sft_losses = torch.tensor(
                sft_losses, device=self.device, dtype=torch.float32
            )

        sft_loss = sft_losses.mean()
        metrics_data["sft_loss"] = (
            sft_loss.detach() if log_fragile_scalar_metrics else float("nan")
        )
        total_loss = loss + self.sft_loss_weight * sft_loss
        loss = total_loss

        loss_ratio = sft_loss.detach().abs() / ppo_loss.abs().clamp_min(1e-8)
        metrics_data["loss_ratio"] = (
            loss_ratio if log_fragile_scalar_metrics else float("nan")
        )
        if log_fragile_scalar_metrics and float(loss_ratio.float().cpu().item()) > 1e5:
            self.logger.warning(
                "SFT/PPO loss imbalance detected: "
                f"ratio={float(loss_ratio.float().cpu().item()):.3e}, "
                f"sft_loss={float(sft_loss.detach().float().cpu().item()):.6f}, "
                f"ppo_loss={float(ppo_loss.float().cpu().item()):.6f}, "
                f"sft_loss_weight={self.sft_loss_weight:.6f}"
            )

    @Worker.timer("run_training")
    def run_training(self) -> None:
        """
        Run the training process using the received rollout batch.
        """
        self._debug_actor_stage_marker("enter run_training")
        log_fragile_scalar_metrics = self._actor_debug_flag(
            "log_fragile_scalar_metrics",
            "RLINF_LOG_FRAGILE_SCALAR_METRICS",
            default=False,
        )
        all_reduce_training_metrics = self._actor_debug_flag(
            "all_reduce_training_metrics",
            "RLINF_ALL_REDUCE_TRAINING_METRICS",
            default=False,
        )
        self._debug_actor_stage_marker(
            "run_training metric_options "
            f"log_fragile_scalar_metrics={log_fragile_scalar_metrics} "
            f"all_reduce_training_metrics={all_reduce_training_metrics}"
        )
        if self.is_weight_offloaded:
            self._debug_actor_stage_marker("run_training before load_param_and_grad")
            self.load_param_and_grad(self.device)
            self._debug_actor_stage_marker("run_training after load_param_and_grad")
        if self.is_optimizer_offloaded:
            self._debug_actor_stage_marker("run_training before load_optimizer")
            self.load_optimizer(self.device)
            self._debug_actor_stage_marker("run_training after load_optimizer")

        self._debug_actor_stage_marker("run_training before model.train")
        self.model.train()
        self._debug_actor_stage_marker("run_training after model.train")
        rollout_size = (
            self.rollout_batch["prev_logprobs"].shape[0]
            * self.rollout_batch["prev_logprobs"].shape[1]
        )
        self._debug_actor_stage_marker(f"run_training rollout_size={rollout_size}")
        g = torch.Generator()
        g.manual_seed(self.cfg.actor.seed + self._rank)
        self._debug_actor_stage_marker("run_training before torch.randperm")
        shuffle_id = torch.randperm(rollout_size, generator=g)
        self._debug_actor_stage_marker("run_training after torch.randperm")

        with torch.no_grad():
            self._debug_actor_stage_marker(
                "run_training before process_nested_dict_for_train"
            )
            self.rollout_batch = process_nested_dict_for_train(
                self.rollout_batch, shuffle_id
            )
            self._debug_actor_stage_marker(
                "run_training after process_nested_dict_for_train"
            )

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        ), "global_batch_size is not divisible by micro_batch_size * world_size"

        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        rollout_size = self.rollout_batch["prev_logprobs"].size(0)
        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        self._debug_actor_stage_marker(
            "run_training before split rollout batches "
            f"rollout_size={rollout_size} batch_size_per_rank={batch_size_per_rank}"
        )
        assert rollout_size % batch_size_per_rank == 0, (
            f"{rollout_size} is not divisible by {batch_size_per_rank}"
        )
        metrics = {}
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for update_epoch_idx in range(update_epoch):
            self._debug_actor_stage_marker(
                f"run_training before update_epoch={update_epoch_idx}"
            )
            rollout_dataloader_iter = split_dict_to_chunk(
                self.rollout_batch,
                rollout_size // batch_size_per_rank,
            )
            for global_batch_idx, train_global_batch in enumerate(
                rollout_dataloader_iter
            ):
                self._debug_actor_stage_marker(
                    "run_training before train_global_batch "
                    f"update_epoch={update_epoch_idx} global_batch_idx={global_batch_idx}"
                )
                # split batch into micro_batches
                train_global_batch_size = train_global_batch["prev_logprobs"].shape[0]
                assert (
                    train_global_batch_size
                    == self.cfg.actor.global_batch_size
                    // torch.distributed.get_world_size()
                )
                assert train_global_batch_size % self.cfg.actor.micro_batch_size == 0, (
                    f"{train_global_batch_size=}, {self.cfg.actor.micro_batch_size}"
                )

                train_micro_batch = split_dict_to_chunk(
                    train_global_batch,
                    train_global_batch_size // self.cfg.actor.micro_batch_size,
                )

                self._debug_actor_stage_marker(
                    "run_training before optimizer.zero_grad "
                    f"update_epoch={update_epoch_idx} global_batch_idx={global_batch_idx}"
                )
                self.optimizer.zero_grad()
                self._debug_actor_stage_marker(
                    "run_training after optimizer.zero_grad "
                    f"update_epoch={update_epoch_idx} global_batch_idx={global_batch_idx}"
                )
                for idx, batch in enumerate(train_micro_batch):
                    self._debug_actor_stage_marker(
                        "run_training before put_tensor_device "
                        f"update_epoch={update_epoch_idx} global_batch_idx={global_batch_idx} "
                        f"micro_batch_idx={idx}"
                    )
                    batch = put_tensor_device(
                        batch,
                        f"{Worker.torch_device_type}:{int(os.environ['LOCAL_RANK'])}",
                    )
                    self._debug_actor_stage_marker(
                        "run_training after put_tensor_device "
                        f"update_epoch={update_epoch_idx} global_batch_idx={global_batch_idx} "
                        f"micro_batch_idx={idx}"
                    )
                    self._debug_actor_stage_marker(
                        "run_training before before_micro_batch "
                        f"update_epoch={update_epoch_idx} global_batch_idx={global_batch_idx} "
                        f"micro_batch_idx={idx}"
                    )
                    backward_ctx = self.before_micro_batch(
                        self.model,
                        is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                    )
                    self._debug_actor_stage_marker(
                        "run_training after before_micro_batch "
                        f"update_epoch={update_epoch_idx} global_batch_idx={global_batch_idx} "
                        f"micro_batch_idx={idx}"
                    )
                    advantages = batch["advantages"]
                    prev_logprobs = batch["prev_logprobs"]
                    returns = batch.get("returns", None)
                    prev_values = batch.get("prev_values", None)
                    loss_mask = batch.get("loss_mask", None)
                    loss_mask_sum = batch.get("loss_mask_sum", None)

                    forward_inputs = batch.get("forward_inputs", None)

                    kwargs = {}
                    if SupportedModel(self.cfg.actor.model.model_type) in [
                        SupportedModel.OPENVLA,
                        SupportedModel.OPENVLA_OFT,
                    ]:
                        kwargs["temperature"] = (
                            self.cfg.algorithm.sampling_params.temperature_train
                        )
                        kwargs["top_k"] = self.cfg.algorithm.sampling_params.top_k
                    elif (
                        SupportedModel(self.cfg.actor.model.model_type)
                        == SupportedModel.GR00T
                    ):
                        kwargs["prev_logprobs"] = prev_logprobs

                    compute_values = (
                        True if self.cfg.algorithm.adv_type == "gae" else False
                    )

                    self._debug_actor_stage_marker(
                        "run_training before model forward "
                        f"update_epoch={update_epoch_idx} global_batch_idx={global_batch_idx} "
                        f"micro_batch_idx={idx}"
                    )
                    forward_marker_prefix = (
                        "run_training model forward "
                        f"update_epoch={update_epoch_idx} global_batch_idx={global_batch_idx} "
                        f"micro_batch_idx={idx}"
                    )
                    with self._debug_inner_model_forward_markers(
                        forward_marker_prefix
                    ):
                        with self.amp_context:
                            output_dict = self.model(
                                forward_inputs=forward_inputs,
                                compute_logprobs=True,
                                compute_entropy=self.cfg.algorithm.entropy_bonus > 0,
                                compute_values=compute_values,
                                use_cache=False,
                                **kwargs,
                            )
                    self._debug_actor_stage_marker(
                        "run_training after model forward "
                        f"update_epoch={update_epoch_idx} global_batch_idx={global_batch_idx} "
                        f"micro_batch_idx={idx}"
                    )
                    self._debug_cuda_sync_marker(
                        "after model forward "
                        f"update_epoch={update_epoch_idx} global_batch_idx={global_batch_idx} "
                        f"micro_batch_idx={idx}"
                    )

                    if (
                        SupportedModel(self.cfg.actor.model.model_type)
                        == SupportedModel.GR00T
                    ):
                        prev_logprobs = output_dict["prev_logprobs"]

                    kwargs = {
                        "loss_type": self.cfg.algorithm.loss_type,
                        "logprob_type": self.cfg.algorithm.logprob_type,
                        "reward_type": self.cfg.algorithm.reward_type,
                        "single_action_dim": self.cfg.actor.model.get("action_dim", 7),
                        "logprobs": output_dict["logprobs"],
                        "values": output_dict.get("values", None),
                        "old_logprobs": prev_logprobs,
                        "advantages": advantages,
                        "returns": returns,
                        "prev_values": prev_values,
                        "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
                        "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
                        "value_clip": self.cfg.algorithm.get("value_clip", None),
                        "huber_delta": self.cfg.algorithm.get("huber_delta", None),
                        "loss_mask": loss_mask,
                        "loss_mask_sum": loss_mask_sum,
                        "max_episode_steps": self.cfg.env.train.max_episode_steps,
                        "task_type": self.cfg.runner.task_type,
                        "critic_warmup": self.optimizer_steps
                        < self.critic_warmup_steps,
                    }
                    self._debug_actor_stage_marker(
                        "run_training before policy_loss "
                        f"update_epoch={update_epoch_idx} global_batch_idx={global_batch_idx} "
                        f"micro_batch_idx={idx}"
                    )
                    loss, metrics_data = policy_loss(**kwargs)
                    self._debug_actor_stage_marker(
                        "run_training after policy_loss "
                        f"update_epoch={update_epoch_idx} global_batch_idx={global_batch_idx} "
                        f"micro_batch_idx={idx}"
                    )
                    self._debug_cuda_sync_marker(
                        "after policy_loss "
                        f"update_epoch={update_epoch_idx} global_batch_idx={global_batch_idx} "
                        f"micro_batch_idx={idx}"
                    )

                    entropy_loss = torch.tensor(
                        0.0, device=Worker.torch_platform.current_device()
                    )
                    if (
                        self.cfg.algorithm.entropy_bonus > 0
                        and not kwargs["critic_warmup"]
                    ):
                        entropy = output_dict["entropy"]
                        entropy = reshape_entropy(
                            entropy,
                            entropy_type=self.cfg.algorithm.entropy_type,
                            action_dim=self.cfg.actor.model.get("action_dim", 7),
                            batch_size=output_dict["logprobs"].shape[0],
                        )
                        entropy_loss = masked_mean(entropy, mask=loss_mask)
                        loss -= self.cfg.algorithm.entropy_bonus * entropy_loss
                    entropy_metric_action = (
                        "record" if log_fragile_scalar_metrics else "skip"
                    )
                    self._debug_actor_stage_marker(
                        f"run_training {entropy_metric_action} entropy_loss scalar metric "
                        f"update_epoch={update_epoch_idx} global_batch_idx={global_batch_idx} "
                        f"micro_batch_idx={idx}"
                    )
                    metrics_data["actor/entropy_loss"] = (
                        entropy_loss.detach()
                        if log_fragile_scalar_metrics
                        else float("nan")
                    )

                    if self.enable_sft_co_train:
                        self._debug_actor_stage_marker(
                            "run_training before _train_sft_epoch "
                            f"update_epoch={update_epoch_idx} global_batch_idx={global_batch_idx} "
                            f"micro_batch_idx={idx}"
                        )
                        self._train_sft_epoch(
                            metrics_data,
                            loss,
                            log_fragile_scalar_metrics=log_fragile_scalar_metrics,
                        )
                        self._debug_actor_stage_marker(
                            "run_training after _train_sft_epoch "
                            f"update_epoch={update_epoch_idx} global_batch_idx={global_batch_idx} "
                            f"micro_batch_idx={idx}"
                        )

                    total_loss_metric = loss.detach()
                    loss /= self.gradient_accumulation
                    with backward_ctx:
                        self._debug_actor_stage_marker(
                            "run_training before backward "
                            f"update_epoch={update_epoch_idx} global_batch_idx={global_batch_idx} "
                            f"micro_batch_idx={idx}"
                        )
                        self.grad_scaler.scale(loss).backward()
                        self._debug_actor_stage_marker(
                            "run_training after backward "
                            f"update_epoch={update_epoch_idx} global_batch_idx={global_batch_idx} "
                            f"micro_batch_idx={idx}"
                        )
                    self._debug_cuda_sync_marker(
                        "after backward "
                        f"update_epoch={update_epoch_idx} global_batch_idx={global_batch_idx} "
                        f"micro_batch_idx={idx}"
                    )

                    total_metric_action = (
                        "record" if log_fragile_scalar_metrics else "skip"
                    )
                    self._debug_actor_stage_marker(
                        f"run_training {total_metric_action} total_loss scalar metric "
                        f"update_epoch={update_epoch_idx} global_batch_idx={global_batch_idx} "
                        f"micro_batch_idx={idx}"
                    )
                    metrics_data["actor/total_loss"] = (
                        total_loss_metric
                        if log_fragile_scalar_metrics
                        else float("nan")
                    )
                    append_to_dict(metrics, metrics_data)
                    # avoid gpu memory leak
                    train_micro_batch[idx] = None
                    del batch, output_dict, forward_inputs, loss, metrics_data

                self._debug_actor_stage_marker(
                    "run_training before empty_cache "
                    f"update_epoch={update_epoch_idx} global_batch_idx={global_batch_idx}"
                )
                self.torch_platform.empty_cache()
                self._debug_actor_stage_marker(
                    "run_training after empty_cache "
                    f"update_epoch={update_epoch_idx} global_batch_idx={global_batch_idx}"
                )

                self._debug_actor_stage_marker(
                    "run_training before optimizer_step "
                    f"update_epoch={update_epoch_idx} global_batch_idx={global_batch_idx}"
                )
                grad_norm, lr_list = self.optimizer_step()
                self._debug_actor_stage_marker(
                    "run_training after optimizer_step "
                    f"update_epoch={update_epoch_idx} global_batch_idx={global_batch_idx}"
                )
                self._debug_cuda_sync_marker(
                    "after optimizer_step "
                    f"update_epoch={update_epoch_idx} global_batch_idx={global_batch_idx}"
                )
                data = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                }
                if len(lr_list) > 1:
                    data["critic/lr"] = lr_list[1]
                append_to_dict(metrics, data)
        # put LR scheduler step here
        self._debug_actor_stage_marker("run_training before lr_scheduler.step")
        self.lr_scheduler.step()
        self._debug_actor_stage_marker("run_training after lr_scheduler.step")
        self._debug_actor_stage_marker("run_training before final optimizer.zero_grad")
        self.optimizer.zero_grad()
        self._debug_actor_stage_marker("run_training after final optimizer.zero_grad")
        self._debug_actor_stage_marker("run_training before clear_memory")
        clear_memory()
        self._debug_actor_stage_marker("run_training after clear_memory")
        self._debug_actor_stage_marker("run_training before metric aggregation")
        if all_reduce_training_metrics:
            mean_metric_dict = {
                key: self._metric_mean_for_all_reduce(value)
                for key, value in metrics.items()
            }
            self._debug_actor_stage_marker(
                "run_training before all_reduce_dict metric aggregation"
            )
            mean_metric_dict = all_reduce_dict(
                mean_metric_dict, op=torch.distributed.ReduceOp.AVG
            )
            self._debug_actor_stage_marker(
                "run_training after all_reduce_dict metric aggregation"
            )
        else:
            metric_mean = (
                self._metric_mean_for_all_reduce
                if log_fragile_scalar_metrics
                else self._safe_local_metric_mean
            )
            mean_metric_dict = {
                key: metric_mean(value) for key, value in metrics.items()
            }
            self._debug_actor_stage_marker(
                "run_training after local metric aggregation"
            )

        self._debug_actor_stage_marker("exit run_training")
        return mean_metric_dict

    def set_global_step(self, global_step: int) -> None:
        """
        Set the global step for the model, if needed.
        """
        self.version = global_step
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)
