HOME_DIR=$(readlink -m ..)

PYTHONPATH=$HOME_DIR python $HOME_DIR/score_by_modality/model_prompter.py \
  --task $TASK \
  --image-base $IMAGE_DIR \
  --input_json $INPUT_JSON \
  --output_json $OUTPUT_JSON \
  --instruction_file $HOME_DIR/properties/instructions.json \
  --model $MODEL \
  --chat-url $CHAT_SERVICE_URL