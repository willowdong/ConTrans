export CUDA_VISIBLE_DEVICES=0

model_s="llama-7b"
model_t="llama-13b"
python train_affine_svd.py \
--source_model $model_s \
--target_model $model_t \
--source_data vectors/hidden_states/"$model_s"_wiki_split_input_all.npy \
--target_data vectors/hidden_states/"$model_t"_wiki_split_input_all.npy