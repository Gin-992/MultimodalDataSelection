HOME_DIR=$(readlink -m ..)

python $HOME_DIR/score_by_modality/score_raw_string.py \
  -i $INPUT_JSON \
  -o $OUTPUT_JSON \
  -m $TASK