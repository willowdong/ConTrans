export CUDA_VISIBLE_DEVICES=0

python get_hidden.py \
--model_name llama-7b \
--method mean \
--tok_position last \
--save_hidden \
--data_file "data/wiki_split_input.txt"

python get_hidden.py \
--model_name llama-13b \
--method mean \
--tok_position last \
--save_hidden \
--data_file "data/wiki_split_input.txt"

emos=("happiness" "sadness" "anger" "fear" "disgust" "surprise")
for item in "${emos[@]}"; do
    python get_hidden.py \
    --model_name llama-7b \
    --method mean \
    --tok_position last \
    --save_hidden \
    --data_file "data/emotions/refine_$item.json"
    echo "Finished emotion: $item."
done
