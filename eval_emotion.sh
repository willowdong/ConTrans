export CUDA_VISIBLE_DEVICES=0

python eval_emotion.py \
--model_name llama-13b \
--method mean \
--intervention identity \
--emotion anger \
--strength 1 

python eval_emotion.py \
--model_name llama-13b \
--method mean \
--intervention steer \
--emotion anger \
--patching_vectors_path vectors/patch_vectors/llama-7to13-anger.npy \
--strength 1 

python eval_emotion.py \
--model_name llama-13b \
--method mean \
--intervention steer \
--emotion anger \
--patching_vectors_path vectors/patch_vectors/llama-7to13-anger.npy \
--strength 2