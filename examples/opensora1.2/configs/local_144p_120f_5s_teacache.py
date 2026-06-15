# FP16 + TeaCache configuration
# Based on local_144p_120f_5s.py, with TeaCache enabled.

resolution = "144p"
aspect_ratio = "1:1"
num_frames = 120
fps = 24
frame_interval = 1
save_fps = 24
ptq_config = "./configs/w8a8_120f_5s.yaml"
save_dir = "/home/rich/ViDiT-Q/.local/outputs/opensora_fp16_teacache_144p_120f_5s_10steps"
seed = 42
batch_size = 1
num_sample = 1
end_index = 1
multi_resolution = "STDiT2"
dtype = "fp16"
condition_frame_length = 5
align = 5
verbose = 1

model = dict(
    type="STDiT3-XL/2",
    from_pretrained="/home/rich/ViDiT-Q/.local/models/hpcai-tech/OpenSora-STDiT-v3",
    qk_norm=True,
    enable_flash_attn=False,
    enable_layernorm_kernel=False,
    force_huggingface=True,
)
vae = dict(
    type="OpenSoraVAE_V1_2",
    from_pretrained="/home/rich/ViDiT-Q/.local/models/hpcai-tech/OpenSora-VAE-v1.2",
    micro_frame_size=17,
    micro_batch_size=1,
    force_huggingface=True,
)
text_encoder = dict(
    type="t5",
    from_pretrained="/home/rich/ViDiT-Q/.local/models/DeepFloyd/t5-v1_1-xxl",
    model_max_length=300,
)
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=10,
    cfg_scale=7.0,
)

aes = 6.5
flow = None
precompute_text_embeds = True
prompt_path = "./prompts.txt"
model_path = "/home/rich/ViDiT-Q/.local/models"
hardware = False

# TeaCache options (calibrated 2026-06-12, R²=0.9902)
enable_teacache = True
teacache_thresh = 0.30
teacache_coeff = [1.75568353e+02, -1.44553580e+02, 2.97780810e+01, -1.48366012e+00]
# Speed mode: teacache_thresh = 0.50
# Quality mode: teacache_thresh = 0.20 (may have low hit rate with 10-step)
