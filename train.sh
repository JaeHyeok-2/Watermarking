# Train with terminal only (for the resolution of 400*400 with best PSNR)
CUDA_VISIBLE_DEVICES=0 python train_gui.py --source_path data/lego --model_path outputs/lego --deform_type node --node_num 512 --hyper_dim 8 --is_blender --eval  --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800
