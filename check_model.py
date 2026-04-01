
import torch

ckpt = torch.load("src/external/Self_Forcing/checkpoints/ode_init.pt", map_location="cpu")
sd = ckpt["generator"]

total_params = sum(v.numel() for v in sd.values())
print(f"Total parameters: {len(sd)} tensors, {total_params:,} ({total_params/1e9:.2f}B)\n")

# Show block 0 fully to understand block structure
print("=== Block 0 structure ===")
for k, v in sorted(sd.items()):
    if k.startswith("model.blocks.0."):
        name = k.replace("model.blocks.0.", "")
        print(f"  {name}: {list(v.shape)}")

# Count blocks
block_ids = set()
for k in sd:
    if k.startswith("model.blocks."):
        block_ids.add(int(k.split(".")[2]))
print(f"\n=== Summary ===")
print(f"Number of blocks: {len(block_ids)} (0-{max(block_ids)})")
print(f"Hidden dim: 1536")
print(f"Patch embedding: in_channels=16, kernel=[1,2,2]")
print(f"Text embedding: 4096 -> 1536 (2-layer MLP)")
print(f"Time embedding: 256 -> 1536 (2-layer MLP)")
print(f"Time projection: 1536 -> 9216 (=1536*6, for 6 modulation params)")
print(f"Head: 1536 -> 64 (=16*1*2*2, unpatchify)")
