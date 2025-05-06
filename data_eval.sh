vllm serve $MODEL_PATH --served-model-name $MODEL --host $HOST --port $PORT  > output/model-infer-service.log 2>&1 &

sleep 2000

python score_by_modality/model_prompter_overwrite.py \
  --task $TASK \
  --image-base $IMAGE_DIR \
  --input_json $INPUT_JSON \
  --instruction_file properties/instructions.json \
  --model $MODEL \
  --chat-url "http://$HOST:$PORT/v1/chat/completions"