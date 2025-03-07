if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

dataset_names="ar bn de en es fa fi fr hi id ja ko ru sw te th yo zh"

VENV="/home/ubuntu/bstadt-smol/flagemb/env"
source $VENV/bin/activate

eval_args="\
    --eval_name miracl \
    --dataset_dir ./miracl/data \
    --dataset_names $dataset_names \
    --splits dev \
    --corpus_embd_save_dir ./miracl/corpus_embd \
    --output_dir ./miracl/search_results \
    --search_top_k 1000 \
    --cache_path $HF_HUB_CACHE \
    --overwrite False \
    --k_values 10 100 \
    --eval_output_method markdown \
    --eval_output_path ./miracl/miracl_eval_results.md \
    --eval_metrics ndcg_at_10 recall_at_100 \
"

model_args="\
    --embedder_name_or_path voyage
    --devices cuda:1 \
    --trust_remote_code \
    --query_instruction_for_retrieval 'search_query: ' \
    --passage_instruction_for_retrieval 'search_document: ' \
    --embedder_batch_size 32 \
    --cache_dir $HF_HUB_CACHE 
"

cmd="/home/ubuntu/bstadt-smol/flagemb/env/bin/python -m FlagEmbedding.evaluation.miracl \
    $eval_args \
    $model_args \
"

echo $cmd
eval $cmd
