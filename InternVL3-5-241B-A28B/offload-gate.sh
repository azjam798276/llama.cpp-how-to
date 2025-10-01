#!/bin/bash

# This script launches the llama-server with a complex, hybrid offloading strategy
# designed for a heterogeneous multi-GPU setup (NVIDIA CUDA + AMD Vulkan).

./llama-server \
  # --- Core Model & System Settings ---
  # Specifies the path to the large language model file.
  --model /InternVL3-5-241B-A28B/InternVL3_5-241B-A28B.Q4_K_M.gguf \
  # Use all available CPU threads for processing.
  --threads -1 \
  # Set the maximum context size for the model.
  --ctx-size 16384 \
  # A hint to offload as many layers as possible to the GPUs.
  --n-gpu-layers 99 \
  # --- GPU & Offloading Strategy ---
  # Defines the available GPU devices in order of preference.
  --device CUDA0,Vulkan0,Vulkan1 \
  # Sets the default tensor offload target to the first device (CUDA0).
  # The -ot rules will then redirect tensors that don't fit or are manually assigned elsewhere.
  --tensor-split 1,0,0 \
  # --- Granular Tensor Offloading Rules (-ot) ---
  # Rule 1: Offload ALL attention layers (`attn_.*`) to the primary CUDA device.
  -ot "blk.*.attn_.*=CUDA0" \
  # Rule 2: Offload the multimodal vision model to the first Vulkan device.
  -ot "vision_model.encoder.*=Vulkan0" \
  # Rule 3: Offload the 'gate' expert tensors for layers 0-54 to the first Vulkan device.
  -ot "blk.([0-9]|[1-4][0-9]|5[0-4])\.ffn_gate_exps.*=Vulkan0" \
  # Rule 4: Offload the 'gate' expert tensors for the remaining layers (55-93) to the second Vulkan device.
  -ot "blk.(5[5-9]|[6-8][0-9]|9[0-3])\.ffn_gate_exps.*=Vulkan1" \
  # Rule 5: Offload the heavier 'up' and 'down' expert tensors for ALL layers to the CPU (CUDA_Host memory).
  -ot "blk.*\.ffn_(up|down)_exps.*=CPU" \
  # --- Inference & Server Parameters ---
  # Set the sampling temperature for generation.
  --temp 0.7 \
  # Use top-p (nucleus) sampling.
  --top-p 0.9 \
  # Bind the server to all network interfaces.
  --host 0.0.0.0 \
  # Set the server port.
  --port 8080
