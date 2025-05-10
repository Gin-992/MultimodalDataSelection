git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -e .

pip install lmdeploy openai

CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server "/mnt/hdfs/andrew.estornell/vlm/${MODEL}" --model-name ${MODEL} --server-port 23333  > test-model.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 lmdeploy serve api_server "/mnt/hdfs/andrew.estornell/vlm/Qwen2.5-32B-Instruct-AWQ" --model-name Qwen2.5-32B-Instruct-AWQ --server-port 23334 --api-keys sk-123456  > judge-model.log 2>&1 &

cat << 'EOF' >> .env
LMDEPLOY_API_BASE=http://0.0.0.0:23333/v1/chat/completions

OPENAI_API_KEY=sk-123456
OPENAI_API_BASE=http://0.0.0.0:23334/v1/chat/completions
LOCAL_LLM=Qwen2.5-32B-Instruct-AWQ
EOF

until curl -s http://0.0.0.0:23333/v1/models | grep -q $MODEL; do
  echo "Waiting for model…"
  sleep 5
done

until curl -s http://0.0.0.0:23334/v1/models | grep -q $MODEL; do
  echo "Waiting for model…"
  sleep 5
done

mkdir -p "/mnt/hdfs/andrew.estornell/vlm/output/${MODEL}"

python run.py --data ${BENCHMARKS} --model lmdeploy --work-dir "/mnt/hdfs/andrew.estornell/vlm/output/${MODEL}" --verbose --api-nproc 64

