# Self-Forcing Training Pipeline

This document describes the complete training pipeline for the Self-Forcing / DMD (Distribution Matching Distillation) framework as observed from runtime execution.

---

## 1. Overview

The framework trains a **causal video generation model** (the generator) to produce videos in a single forward pass, using score distillation from a large pretrained teacher diffusion model. The key idea is to unroll the generator autoregressively block-by-block using a KV cache, then apply DMD loss against a frozen teacher score.

### Three models

| Model | Architecture | Parameters | Trainable | Role |
|---|---|---|---|---|
| **Generator** | CausalWanModel (1.3B) | 1,418,996,800 | Yes | Student: generates videos causally, block-by-block |
| **Real score** | WanModel 14B (non-causal) | 14,288,491,584 | **No (frozen)** | Teacher: provides real data score for DMD gradient |
| **Fake score** | WanModel 1.3B (non-causal) | 1,418,996,800 | Yes | Critic: trained to match generator distribution |

Plus frozen **text encoder** (UMT5-XXL) and **VAE** (Wan2.1 VAE, 16 latent channels).

---

## 2. Key Configuration (from `self_forcing_dmd.yaml`)

```
Video shape:           [B=1, F=21, C=16, H=60, W=104]  (latent space)
Pixel resolution:      ~480x832 (after 8x VAE upscale)
Latent channels:       16
Frames per video:      21
Frames per block:      3  (generator processes 3 frames at a time)
Num blocks:            7  (21 frames / 3 frames per block)

Denoising steps:       [1000, 750, 500, 250] (raw config, before warp)
After warp_denoising_step: [1000.0, 937.5, 833.3, 625.0]
Num train timesteps:   1000
Timestep shift:        5.0 (shifts sampling toward high-noise region)
Min/max score step:    [20, 980] (2% and 98% of 1000)

Generator LR:          2e-6   (AdamW, beta1=0.0, beta2=0.999)
Critic LR:             4e-7   (AdamW, beta1=0.0, beta2=0.999)
EMA decay:             0.99   (starts at step 200)
Generator update ratio: every 5 steps  (dfake_gen_update_ratio=5)
Gradient checkpointing: True (both generator and fake score)
Precision:             bfloat16 mixed
Distribution:          FSDP hybrid_full (4 GPUs observed)
Dataset:               TextDataset — 248,221 text prompts (vidprom_filtered_extended)
```

---

## 3. Training Loop Structure

```
For each step:
  ├─ [Every 5th step] GENERATOR UPDATE
  │   ├─ fake_score.requires_grad_(False)
  │   ├─ generator.requires_grad_(True)
  │   ├─ generator_optimizer.zero_grad()
  │   ├─ fwdbwd_one_step(batch, train_generator=True)
  │   ├─ generator_optimizer.step()
  │   └─ EMA update (if step >= 200)
  │
  └─ [Every step] CRITIC UPDATE
      ├─ generator.requires_grad_(False)
      ├─ fake_score.requires_grad_(True)
      ├─ critic_optimizer.zero_grad()
      ├─ fwdbwd_one_step(batch, train_generator=False)
      └─ critic_optimizer.step()
```

Both generator and critic use **separate batches** (separate `next(dataloader)` calls).

---

## 4. One Training Step in Detail (`fwdbwd_one_step`)

### 4.1 Data loading

Each batch contains only `{'prompts': [...], 'idx': [...]}` — **no video frames**. The training is dataset-free in the sense that videos are generated on-the-fly from noise.

```
text_prompts: List[str]   (batch_size=1 per GPU)
clean_latent = None       (t2v mode, no conditioning image)
image_latent = None
```

### 4.2 Text encoding

```python
conditional_dict = text_encoder(text_prompts)
# → {'prompt_embeds': [B, 512, 4096], bfloat16, on GPU}

unconditional_dict = text_encoder([negative_prompt] * B)
# → {'prompt_embeds': [B, 512, 4096], bfloat16}
# Cached after first use — never recomputed between steps
```

The unconditional dict uses a fixed Chinese-language negative prompt and is computed once, then reused for every subsequent step.

---

## 5. Generator Loss Path

### 5.1 Autoregressive video generation (`_consistency_backward_simulation`)

This is the core "self-forcing" mechanism. The generator unrolls the full 21-frame video causally using a KV cache, block by block.

#### Setup

```
noise: [B=1, F=21, C=16, H=60, W=104]   bfloat16  (pure Gaussian noise)
num_blocks = 7
all_num_frames = [3, 3, 3, 3, 3, 3, 3]  (each block = 3 frames)
```

**KV cache shape:** 30 transformer blocks × `[B=1, kv_cache_size=32760, 12 heads, 128 dim]`
- `kv_cache_size = num_max_frames * frame_seq_length = 21 * 1560 = 32760`
- `frame_seq_length = 1560` (sequence length per frame in Wan's attention)

**Cross-attention cache shape:** 30 blocks × `[B=1, 512, 12 heads, 128 dim]`
- Caches text conditioning keys/values

Both caches are **re-initialized to zeros** at the start of each call.

#### Per-block denoising (Stochastic Truncated Denoising)

For each of the 7 blocks:

```
Block processes: noise[:, start_frame:start_frame+3, ...]   shape [B, 3, 16, 60, 104]
```

**Exit flags** are sampled once for all blocks on rank 0 and broadcast:
```
exit_flags = [i_0, i_1, i_2, i_3, i_4, i_5, i_6]   each ∈ {0, 1, 2, 3}
```
With `same_step_across_blocks=True`, ALL blocks use the same exit index `exit_flags[0]`.

For each denoising step `(index, timestep)` in `[1000.0, 937.5, 833.3, 625.0]`:

```
if index < exit_flag:
    # No gradient — intermediate step
    with torch.no_grad():
        flow_pred, denoised_pred = generator(noisy_input, t=timestep, kv_cache=..., current_start=block*1560)
    # Re-noise to next timestep level (stochastic resetting):
    noisy_input = scheduler.add_noise(denoised_pred, randn_like(denoised_pred), next_timestep)

elif index == exit_flag:
    # This is the exit step — compute the final denoised prediction
    if current_start_frame >= start_gradient_frame_index:
        # WITH gradient (last 21 frames; start_gradient_frame_index = num_output_frames - 21 = 0 here)
        flow_pred, denoised_pred = generator(noisy_input, t=timestep, kv_cache=..., current_start=...)
    else:
        # No gradient (early frames when video > 21 frames)
        with torch.no_grad():
            flow_pred, denoised_pred = generator(...)
    break
```

**Observed example (from `debug.log`):**
```
exit_flags = [1, 2, 0, 1, 2, 3, 1]   (sampled randomly, but only exit_flags[0] is used!)
same_step_across_blocks = True  → ALL 7 blocks exit at exit_flags[0] = 1
```
- All blocks: run step t=1000 (no grad), then exit at t=937.5 (WITH grad)
- The remaining entries `[2, 0, 1, 2, 3, 1]` are ignored when `same_step_across_blocks=True`

The exit timestep is the same for all blocks in a given training iteration. Across iterations it varies stochastically (sampling `exit_flags[0]` uniformly from `{0, 1, 2, 3}`).

**Observed denoised timestep values:**
```
denoised_timestep_from = 750   (the timestep step FROM which the block was denoised, i.e. warp[exit_flag])
denoised_timestep_to   = 500   (the next step level after exit; warp[exit_flag+1])
```
These are computed from `denoising_step_list` indices after warp: exit_flag=1 → warp[1]=937.5, but `from/to` track the *scheduler-registered* adjacent steps:
- `denoised_timestep_from` = warped step at `exit_flag` = 937.5 → nearest scheduler = 937.5 → reported as 750 (this is the original config index step, not the warped value — see Section 8)
- These values are **passed into `compute_distribution_matching_loss`** but with `ts_schedule=False` they are not used to constrain the score evaluation timestep (see Section 5.2).

#### KV cache update after each block

After recording the block's output, the model is run **one more time** with `timestep=0` and `context_noise=0` to populate the KV cache with the "clean" token representation:

```python
# Re-add context noise (noise level 0 here = no noise)
denoised_for_cache = scheduler.add_noise(denoised_pred, randn, context_noise * ones)
# context_noise=0 → add_noise with t=0 → essentially returns denoised_pred unchanged

with torch.no_grad():
    generator(denoised_for_cache, t=0, kv_cache=..., current_start=block*1560)
    # This forward does NOT return output — only side-effect: writes KV into cache
```

This is the "self-forcing" trick: future blocks see the clean (t=0) representation of past frames in their KV cache, which makes generation causal and consistent.

**`current_start` progression:**
```
Block 0: current_start = 0       (token offset = 0 * 1560)
Block 1: current_start = 4680    (token offset = 3 * 1560)
Block 2: current_start = 9360    (token offset = 6 * 1560)
Block 3: current_start = 14040
Block 4: current_start = 18720
Block 5: current_start = 23400
Block 6: current_start = 28080
```

#### Output

```
output: [B=1, F=21, C=16, H=60, W=104]   bfloat16   requires_grad=True  (generator path)
                                                      requires_grad=False  (critic path)
denoised_timestep_from: 750   (original config step index at exit_flag position)
denoised_timestep_to:   500   (original config step index at exit_flag+1 position)
```

With 21-frame video, `start_gradient_frame_index = num_output_frames - 21 = 0`, so ALL 7 blocks have gradients enabled in the generator path. In the critic path, the entire call is wrapped in `torch.no_grad()`, so all blocks have `requires_grad=False`.

### 5.2 DMD Generator Loss (`compute_distribution_matching_loss`)

Takes the generated video `pred_image: [B, 21, 16, 60, 104]` and computes:

#### Step A: Sample a noise level

```python
timestep = randint(0, 1000, [B, 1]).repeat(1, 21)    # uniform across all frames
# Apply timestep shift (shift=5.0):
timestep = 5.0 * (t/1000) / (1 + 4.0 * (t/1000)) * 1000
# Clamp to [20, 980]  (min_score_timestep=20, set as percentage of num_train_timesteps=1000)
timestep = timestep.clamp(20, 980)
# Result: shape [B=1, F=21], same value across all frames, in range [20, 980]
```

The shift formula concentrates timesteps toward higher noise levels (above ~200) — the shifted distribution favors the high-noise regime where guidance is most informative.

**`ts_schedule=False` (config default):** `denoised_timestep_from` and `denoised_timestep_to` returned by `inference_with_trajectory` are passed in but **not used** to restrict this range. With `ts_schedule=True`, the score evaluation timestep would be restricted to `[denoised_timestep_to, denoised_timestep_from]` — i.e., only evaluate the score at noise levels close to where the generator actually operated. With `ts_schedule=False`, the full `[min_score_timestep=0, num_train_timestep=1000]` range is used (before clamp), making the critic/teacher see a broader distribution of noise levels.

#### Step B: Add noise to generated video

```python
noise = randn_like(pred_image)   # [B, 21, 16, 60, 104]
noisy_latent = scheduler.add_noise(pred_image.flatten(0,1), noise.flatten(0,1), timestep.flatten(0,1))
# → [B*21, 16, 60, 104] → unflatten → [B, 21, 16, 60, 104]
```

All three score evaluations below use this same `noisy_latent`.

#### Step C: Compute KL gradient (`_compute_kl_grad`)

Everything inside is `torch.no_grad()`.

**Fake score (critic/denoiser) forward:**
```python
_, pred_fake = fake_score(noisy_latent, conditional_dict, timestep)
# fake_score = WanModel 1.3B (non-causal), uniform timestep → takes timestep[:,0] = [B]
# pred_fake: [B, 21, 16, 60, 104]  bfloat16
```

Since `fake_guidance_scale=0.0` (from `guidance_scale=3.0` going to `fake_guidance_scale=0.0` default), no unconditional fake pass.

**Real score (teacher) forward — two passes for CFG:**
```python
_, pred_real_cond   = real_score(noisy_latent, conditional_dict, timestep)
_, pred_real_uncond = real_score(noisy_latent, unconditional_dict, timestep)
# real_score = WanModel 14B (non-causal, FROZEN)
# Each returns [B, 21, 16, 60, 104]  bfloat16

pred_real = pred_real_cond + (pred_real_cond - pred_real_uncond) * real_guidance_scale
# real_guidance_scale=3.0
```

**Gradient computation (DMD eq. 7):**
```python
grad = pred_fake - pred_real                     # [B, 21, 16, 60, 104]
# Gradient normalization (DMD eq. 8):
p_real = pred_image - pred_real                  # [B, 21, 16, 60, 104]
normalizer = |p_real|.mean(dim=[1,2,3,4], keepdim=True)   # [B, 1, 1, 1, 1]
grad = grad / normalizer
grad = nan_to_num(grad)
```

The normalizer is per-sample (mean over all spatial+channel+frame dims), making the gradient scale-invariant.

**Observed values:**
```
dmdtrain_gradient_norm ≈ 0.324  (mean |grad|)
timestep mean ≈ 593 (after shift+clamp, center is ~mid-noise range)
```

#### Step D: DMD loss

```python
dmd_loss = 0.5 * MSE(pred_image.double(),
                     (pred_image.double() - grad.double()).detach(),
                     reduction='mean')
# = 0.5 * |grad|^2 / num_elements   (since target = pred_image - grad)
# This is the "pseudo-loss" whose gradient w.r.t. pred_image equals grad
# Loss dtype: float64 (double precision for numerical stability)
```

**Observed:** `generator_loss ≈ 0.094` (float64)

#### Step E: Backward + optimizer step

```python
generator_loss.backward()
grad_norm = generator.clip_grad_norm_(max_norm=10.0)
generator_optimizer.step()   # AdamW, lr=2e-6
# Observed grad_norm ≈ 0.41
```

---

## 6. Critic Loss Path

### 6.1 Generate fake video (no gradient)

The critic uses the **same autoregressive generation pipeline** as the generator, but wrapped entirely in `torch.no_grad()`. The resulting video is detached — `requires_grad=False`.

```
# Same SelfForcingTrainingPipeline.inference_with_trajectory call
generated_image: [B, 21, 16, 60, 104]   bfloat16   requires_grad=False
```

### 6.2 Sample critic timestep

```python
critic_timestep = randint(0, 1000, [B, 1]).repeat(1, 21)
# Same timestep shift:  5.0 * (t/1000) / (1 + 4 * (t/1000)) * 1000
# Clamp to [20, 980]
# Result: [B=1, F=21] — same value across all frames
```

**Observed:** `critic_timestep mean ≈ 910` (step 0) / `≈ 980.0` (step 1) — independently sampled from generator's timestep each step

### 6.3 Add noise to generated video

```python
critic_noise = randn_like(generated_image)
noisy_generated = scheduler.add_noise(generated_image.flatten(0,1),
                                       critic_noise.flatten(0,1),
                                       critic_timestep.flatten(0,1))
# → unflatten → [B, 21, 16, 60, 104]
```

### 6.4 Fake score (critic/denoiser) forward

```python
_, pred_fake_image = fake_score(noisy_generated, conditional_dict, critic_timestep)
# fake_score = WanModel 1.3B (non-causal), uniform timestep
# pred_fake_image: [B, 21, 16, 60, 104]  bfloat16
```

This is a **gradient-enabled** forward — fake_score is being trained.

### 6.5 Denoising loss for critic (flow matching)

Since `denoising_loss_type=flow`:

```python
# Convert x0 prediction to flow prediction:
# flow_pred = (xt - x0_pred) / sigma_t
flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
    scheduler, x0_pred=pred_fake_image.flatten(0,1),
    xt=noisy_generated.flatten(0,1), timestep=critic_timestep.flatten(0,1)
)

denoising_loss = flow_matching_loss(
    x=generated_image.flatten(0,1),
    x_pred=pred_fake_image.flatten(0,1),
    flow_pred=flow_pred,
    timestep=critic_timestep.flatten(0,1)
)
```

The critic is trained to **correctly denoise the generated (fake) videos**. This makes the fake score approximate the score function of the generator distribution.

**Observed:** `critic_loss ≈ 0.091` (bfloat16)

### 6.6 Backward + optimizer step

```python
critic_loss.backward()
grad_norm = fake_score.clip_grad_norm_(max_norm=10.0)
critic_optimizer.step()   # AdamW, lr=4e-7 (5x smaller than generator lr)
# Observed grad_norm ≈ 1.11
```

---

## 7. Model Internals: `WanDiffusionWrapper.forward`

Both causal and non-causal models share the same `WanDiffusionWrapper.forward`, but differ in:

| | Generator (causal) | Score models (non-causal) |
|---|---|---|
| `uniform_timestep` | `False` | `True` |
| Input timestep | `[B, F]` — per-frame | `[B, F]` → takes `[:,0]` = `[B]` |
| `kv_cache` | Always provided | `None` |
| Model class | `CausalWanModel` | `WanModel` |

**Input tensor permutation:** The model internally expects `[B, C, F, H, W]` but the wrapper receives `[B, F, C, H, W]`. The wrapper permutes before the call and permutes back after:
```python
flow_pred = model(noisy_input.permute(0,2,1,3,4), ...).permute(0,2,1,3,4)
```

**Flow-to-x0 conversion:**
```python
# Flow matching convention: flow_pred = noise - x0
# x_t = (1-sigma_t)*x0 + sigma_t*noise
# ⇒ x0 = x_t - sigma_t * flow_pred

pred_x0 = xt - sigma_t * flow_pred
```
Where `sigma_t` is looked up from the scheduler's sigmas table by matching the timestep.

**Output:** `(flow_pred, pred_x0)` — callers typically use `pred_x0`.

---

## 8. Scheduler and Timestep Details

### FlowMatchScheduler

- `timesteps` range: `[4.98, 1000.0]` (1000 steps)
- Timestep warp (applied to `denoising_step_list`):
  ```
  Raw config:    [1000,   750,   500,   250]
  After warp:    [1000.0, 937.5, 833.3, 625.0]
  ```
  The warp maps config indices (1-indexed from the end of the scheduler's timestep table) to actual timestep values. This is because the scheduler's timesteps are not uniformly spaced.

- When the model is called with timestep `937.5` (config), the actual tensor value seen is `936.0` — this is due to nearest-neighbor lookup in the scheduler's `timesteps` table.

### `add_noise` (flow matching):
```python
# x_t = (1 - sigma_t) * x0 + sigma_t * noise
# where sigma_t is looked up for the given timestep integer
```

### Timestep shift formula:
```python
# shift=5.0:
t_shifted = shift * (t/1000) / (1 + (shift-1) * (t/1000)) * 1000
# Effect: t=500 → t_shifted ≈ 714  (shifts mass toward high noise)
# This ensures the model trains more at high noise levels where guidance is informative
```

---

## 9. Gradient Flow Summary

```
Generator update (every 5th step):
  ┌─ noise → [CausalWanModel, kv_cache, 7 blocks × stochastic exit] → pred_video ─┐
  │   (exit step only has gradient; all prior denoising steps: no_grad)             │
  └─ pred_video → add_noise → [WanModel_1.3B (fake_score, no_grad for G update)]   │
                            → [WanModel_14B (real_score, always frozen)]            │
  └─→ KL grad → DMD pseudo-loss (float64 MSE) → backward → clip_grad(10) → AdamW   │

Critic update (every step):
  ┌─ noise → [CausalWanModel, kv_cache, 7 blocks] → generated_video [no_grad] ─────┐
  └─ generated_video → add_noise → [WanModel_1.3B (fake_score, WITH grad)] → flow_pred
  └─→ flow_matching_loss(generated_video, flow_pred) → backward → clip_grad(10) → AdamW
```

**Key gradient detachments:**
1. Inside `inference_with_trajectory`: intermediate denoising steps (non-exit) are `no_grad`
2. Inside `compute_distribution_matching_loss`: the entire `_compute_kl_grad` is `no_grad`; gradient only flows through the `MSE(pred_image, (pred_image - grad).detach())` pseudo-loss
3. KV cache updates (timestep=0 re-runs after each block) are always `no_grad`
4. Real score (14B teacher) is always frozen (`requires_grad=False`)
5. Critic's generator call is wrapped in `torch.no_grad()`

---

## 10. EMA

```python
# Created at step 0 (if ema_weight > 0), but cleared until step 200
# After step 200:
generator_ema = EMA_FSDP(generator, decay=0.99)
# Updated after each generator optimizer step:
generator_ema.update(generator)   # shadow_param = 0.99 * shadow + 0.01 * param
```

The EMA is used for inference/visualization but not for loss computation during training.

---

## 11. Observed Loss Values (Step 0)

```
generator_loss:           0.094362  (float64)
  dmdtrain_gradient_norm: 0.324219
  timestep mean:          593.5
  generator_grad_norm:    0.410971

critic_loss:              0.091309  (bfloat16)
  critic_timestep mean:   910.3
  critic_grad_norm:       1.112440
```
