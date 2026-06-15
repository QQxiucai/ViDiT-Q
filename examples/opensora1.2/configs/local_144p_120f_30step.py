# 120f 30-step FP16 + TeaCache — main evaluation config
resolution = "144p"
aspect_ratio = "1:1"
num_frames = 120
fps = 24
frame_interval = 1
save_fps = 24
save_dir = "/home/rich/ViDiT-Q/.local/outputs/opensora_fp16_teacache_144p_120f_30steps"
seed = 42
batch_size = 1
num_sample = 1
end_index = 1
multi_resolution = "STDiT2"
dtype = "fp16"
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
    num_sampling_steps=30,
    cfg_scale=7.0,
)

aes = 6.5
flow = None
precompute_text_embeds = True
prompt_path = "./prompts.txt"
model_path = "/home/rich/ViDiT-Q/.local/models"

# TeaCache (recalibrated 2026-06-13 for 120f 30-step, R²=0.985)
enable_teacache = True
teacache_thresh = 0.08
teacache_coeff = [-8.55449464e+01, 2.68463151e+01, 9.46500156e-01, 9.09173339e-03]
