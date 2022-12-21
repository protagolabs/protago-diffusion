export MODEL_NAME="stable-diffusion-v1-5"
export INSTANCE_DIR="hm"
export CLASS_DIR="person"
export OUTPUT_DIR="dreambooth_outputs_test"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --train_text_encoder \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="photo of jnpabdr person" \
  --class_prompt="photo of a person" \
  --resolution=256 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=1500 \
  --max_train_steps=600 \
  --mixed_precision=fp16

  # 1200 1e-6
  # michaelxu person

# export MODEL_NAME="stable-diffusion-v1-5"
# export INSTANCE_DIR="dreambooth_inputs"
# export OUTPUT_DIR="dreambooth_outputs"

# accelerate launch train_dreambooth.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --train_text_encoder \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="a photo of sks person" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=2 --gradient_checkpointing \
#   --learning_rate=2e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --num_class_images=200 \
#   --max_train_steps=1200 \
#   --mixed_precision=fp16