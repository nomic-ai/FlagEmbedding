if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

# pass in language via cli, default is all languages
#"ar bn de en es fa fi fr hi id ja ko ru sw te th yo zh"
#                     0 0  1  1  2  2  3   3  3 4   4  5  5 6   6  7  7
dataset_names=(${1:-"ar bn de en es fa fi fr hi id ja ko ru sw te th yo zh"})
device=${2:-"cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7"}


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
    --embedder_name_or_path nomic-ai/eurobert-210m-2e4-128sl-subset \
    --devices $device \
    --trust_remote_code \
    --query_instruction_for_retrieval 'search_query: ' \
    --passage_instruction_for_retrieval 'search_document: ' \
    --embedder_batch_size 512 \
    --embedder_query_max_length 128 \
    --embedder_passage_max_length 128 \
    --cache_dir $HF_HUB_CACHE 
"

cmd="uv run python -W ignore -m FlagEmbedding.evaluation.miracl \
    $eval_args \
    $model_args \
"

echo $cmd
eval $cmd
