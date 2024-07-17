# Evaluate with GUI (for the resolution of 400*400 with best PSNR)
CUDA_VISIBLE_DEVICES=0 python render.py --source_path data/trex --model_path outputs/trex --deform_type node --node_num 512 --hyper_dim 8 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800
