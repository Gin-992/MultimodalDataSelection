git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -e .

pip install lmdeploy openai
pip install qwen_vl_utils

sudo apt-get update
sudo apt-get install -y libgl1

model_name = $(basename "${MODEL}")

CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server ${MODEL} --model-name ${model_name} --server-port 23333  > test-model.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 lmdeploy serve api_server "Qwen/Qwen2.5-32B-Instruct-AWQ" --model-name Qwen2.5-32B-Instruct-AWQ --server-port 23334 --api-keys sk-123456  > judge-model.log 2>&1 &

cat << 'EOF' >> .env
LMDEPLOY_API_BASE=http://0.0.0.0:23333/v1/chat/completions

OPENAI_API_KEY=sk-123456
OPENAI_API_BASE=http://0.0.0.0:23334/v1/chat/completions
LOCAL_LLM=Qwen2.5-32B-Instruct-AWQ
EOF

until curl -s http://0.0.0.0:23333/v1/models | grep -q ${model_name}; do
  echo "Waiting for model…"
  sleep 5
done
echo "\n\n\nMODEL LOADED\n\n\n"
echo "\n\n\nWATING FOR JUDGE\n\n\n"

until curl -s -H "Authorization: Bearer sk-123456" http://0.0.0.0:23334/v1/models | grep -q Qwen2.5-32B-Instruct-AWQ; do
  echo "Waiting for judge model…"
  sleep 5
done

echo "\n\n\nJUDGE LOADED\n\n\n"


mkdir -p "/mnt/hdfs/andrew.estornell/vlm/output/${MODEL}"
mkdir -p "./output/${MODEL}"

python run.py --data ${BENCHMARKS} --model lmdeploy --work-dir "./output/${MODEL}" --verbose --api-nproc 1

mv "./output/${MODEL}" "/mnt/hdfs/andrew.estornell/vlm/output/${MODEL}"

