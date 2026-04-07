from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from typing import List, Optional
import torch
import torch.distributed as dist

def print0(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)



class SelfForcingTrainingPipeline:
    def __init__(self,
                 denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: WanDiffusionWrapper,
                 num_frame_per_block=3,
                 independent_first_frame: bool = False,
                 same_step_across_blocks: bool = False,
                 last_step_only: bool = False,
                 num_max_frames: int = 21,
                 context_noise: int = 0,
                 **kwargs):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]  # remove the zero timestep for inference

        # Wan specific hyperparameters
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.i2v = False

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.kv_cache_size = num_max_frames * self.frame_seq_length

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device
            )
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()

    def inference_with_trajectory(
            self,
            noise: torch.Tensor,
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            **conditional_dict
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noise.shape
        print0(f"\n[inference_with_trajectory] noise: shape={noise.shape}, dtype={noise.dtype}, device={noise.device}")
        print0(f"[inference_with_trajectory] initial_latent={'None' if initial_latent is None else f'shape={initial_latent.shape}'}")
        print0(f"[inference_with_trajectory] independent_first_frame={self.independent_first_frame}, num_frame_per_block={self.num_frame_per_block}")
        print0(f"[inference_with_trajectory] denoising_step_list={[s.item() if hasattr(s, 'item') else s for s in self.denoising_step_list]}")
        print0(f"[inference_with_trajectory] same_step_across_blocks={self.same_step_across_blocks}, last_step_only={self.last_step_only}")
        print0(f"[inference_with_trajectory] context_noise={self.context_noise}, kv_cache_size={self.kv_cache_size}")

        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        print0(f"[inference_with_trajectory] num_blocks={num_blocks}, num_input_frames={num_input_frames}, num_output_frames={num_output_frames}")

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )
        print0(f"[inference_with_trajectory] output buffer allocated: shape={output.shape}, dtype={output.dtype}")

        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        print0(f"[inference_with_trajectory] KV cache initialized: {self.num_transformer_blocks} blocks x [B={batch_size}, kv_cache_size={self.kv_cache_size}, 12 heads, 128 dim]")
        print0(f"[inference_with_trajectory] Cross-attn cache initialized: {self.num_transformer_blocks} blocks x [B={batch_size}, 512, 12 heads, 128 dim]")
        # if self.kv_cache1 is None:
        #     self._initialize_kv_cache(
        #         batch_size=batch_size,
        #         dtype=noise.dtype,
        #         device=noise.device,
        #     )
        #     self._initialize_crossattn_cache(
        #         batch_size=batch_size,
        #         dtype=noise.dtype,
        #         device=noise.device
        #     )
        # else:
        #     # reset cross attn cache
        #     for block_index in range(self.num_transformer_blocks):
        #         self.crossattn_cache[block_index]["is_init"] = False
        #     # reset kv cache
        #     for block_index in range(len(self.kv_cache1)):
        #         self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
        #             [0], dtype=torch.long, device=noise.device)
        #         self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
        #             [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
            output[:, :1] = initial_latent
            print0(f"[inference_with_trajectory] Caching initial_latent (i2v conditioning): shape={initial_latent.shape}, timestep=0")
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )
            current_start_frame += 1
            print0(f"[inference_with_trajectory] Initial latent cached, current_start_frame now={current_start_frame}")

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        start_gradient_frame_index = num_output_frames - 21
        print0(f"[inference_with_trajectory] Temporal loop: {len(all_num_frames)} blocks, all_num_frames={all_num_frames}")
        print0(f"[inference_with_trajectory] num_denoising_steps={num_denoising_steps}, exit_flags (random step to stop at per block)={exit_flags}")
        print0(f"[inference_with_trajectory] start_gradient_frame_index={start_gradient_frame_index} (only last 21 frames get gradients)")

        # for block_index in range(num_blocks):
        for block_index, current_num_frames in enumerate(all_num_frames):
            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
            print0(f"\n[inference_with_trajectory] === Block {block_index}/{len(all_num_frames)-1} | current_start_frame={current_start_frame}, current_num_frames={current_num_frames} ===")
            print0(f"[inference_with_trajectory] noisy_input sliced: shape={noisy_input.shape}, dtype={noisy_input.dtype}")

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                if self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])  # Only backprop at the randomly selected timestep (consistent across all ranks)
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                print0(f"[inference_with_trajectory]   Denoising step {index}: current_timestep={current_timestep.item() if hasattr(current_timestep, 'item') else current_timestep}, exit_flag={exit_flag}")
                print0(f"[inference_with_trajectory]   timestep tensor: shape={timestep.shape}, value={current_timestep}")
                print0(f"[inference_with_trajectory]   noisy_input: shape={noisy_input.shape}, dtype={noisy_input.dtype}")

                if not exit_flag:
                    print0(f"[inference_with_trajectory]   → no_grad forward (not exit step, renoising to next level)")
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                        print0(f"[inference_with_trajectory]   denoised_pred: shape={denoised_pred.shape}, dtype={denoised_pred.dtype}")
                        print0(f"[inference_with_trajectory]   re-noised to next_timestep={next_timestep.item() if hasattr(next_timestep, 'item') else next_timestep}: noisy_input shape={noisy_input.shape}")
                else:
                    # for getting real output
                    # with torch.set_grad_enabled(current_start_frame >= start_gradient_frame_index):
                    if current_start_frame < start_gradient_frame_index:
                        print0(f"[inference_with_trajectory]   → EXIT STEP (no_grad, frame {current_start_frame} < gradient threshold {start_gradient_frame_index})")
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length
                            )
                    else:
                        print0(f"[inference_with_trajectory]   → EXIT STEP (WITH GRAD, frame {current_start_frame} >= gradient threshold {start_gradient_frame_index})")
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                    print0(f"[inference_with_trajectory]   EXIT denoised_pred: shape={denoised_pred.shape}, dtype={denoised_pred.dtype}, requires_grad={denoised_pred.requires_grad}")
                    break

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred
            print0(f"[inference_with_trajectory] Block {block_index} written to output[frames {current_start_frame}:{current_start_frame+current_num_frames}]")

            # Step 3.3: rerun with timestep zero to update the cache
            context_timestep = torch.ones_like(timestep) * self.context_noise
            # add context noise
            denoised_pred_for_cache = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep * torch.ones(
                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])
            print0(f"[inference_with_trajectory] Re-running with context_noise={self.context_noise} to update KV cache: shape={denoised_pred_for_cache.shape}")
            denoised_pred = denoised_pred_for_cache
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames
            print0(f"[inference_with_trajectory] current_start_frame updated to {current_start_frame}")

        # Step 3.5: Return the denoised timestep
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()

        print0(f"\n[inference_with_trajectory] DONE. output: shape={output.shape}, dtype={output.dtype}, requires_grad={output.requires_grad}")
        print0(f"[inference_with_trajectory] denoised_timestep_from={denoised_timestep_from}, denoised_timestep_to={denoised_timestep_to}")
        print0(f"[inference_with_trajectory] exit_flags={exit_flags} (denoising step index at which each block exited)")

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        return output, denoised_timestep_from, denoised_timestep_to

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
