python3 -u src/train.py \
  --diffusion_type "gaussian" \
  --do_train \
  --do_test \
  --do_valid_only \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/workspace/FlavorGraph/Flavor_Diffusion" \
  --training_split "/workspace/FlavorGraph/Flavor_Diffusion/output_subgraphs/25/subgraphs_train.json" \
  --validation_split "/workspace/FlavorGraph/Flavor_Diffusion/output_subgraphs/25/subgraphs_val.json" \
  --test_split "/workspace/FlavorGraph/Flavor_Diffusion/output_subgraphs/25/subgraphs_test.json" \
  --batch_size 256 \
  --num_epochs 10 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --inference_trick "ddim"