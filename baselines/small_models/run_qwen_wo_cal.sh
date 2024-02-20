python run.py --output_dir outputs_qwen_1_8b \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --num_train_epochs 10 \
  --logging_strategy steps \
  --logging_steps 10 \
  --save_strategy epoch \
  --save_total_limit 10
