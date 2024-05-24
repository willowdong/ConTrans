import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from baukit import TraceDict
import argparse
import logging
import numpy as np
import os
import json
from sklearn.decomposition import PCA
from tqdm import tqdm
from util.utils import model_name2path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_activations_bau(model, prompt, model_name='llama7b'):

    model.eval()
    if 'llama' in model_name or 'amber' in model_name:
        LAYERS = [f"model.layers.{i}" for i in range(model.config.num_hidden_layers)]
    elif 'gpt2' in model_name:
        LAYERS = [f"transformer.h.{i}" for i in range(model.config.n_layer)]
    elif 'gpt-neo' in model_name:
        LAYERS = [f"transformer.h.{i}" for i in range(model.config.num_layers)]
    elif 'pythia' in model_name:
        LAYERS = [f"gpt_neox.layers.{i}" for i in range(model.config.num_hidden_layers)]

    targets = LAYERS

    with torch.no_grad():
        prompt = prompt.to(model.device)
        with TraceDict(model, targets, retain_grad=True) as ret:
            output = model(prompt, labels=prompt, output_hidden_states=True)
            

        hidden_states = output.hidden_states
        hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
        hidden_states = hidden_states.detach().cpu()


        layer_wise_hidden_states = [ret[t].output[0][0,:].detach().cpu() for t in targets]     # [0] is hidden_states, [bsz(1),seq,hidden_size]
        layer_wise_hidden_states = torch.stack(layer_wise_hidden_states, dim = 0)

    return hidden_states, layer_wise_hidden_states


def read_data(data_path):
    datas = []
    with open(data_path,"r+")as f:
        data = json.load(f)
    logger.info(f"Total data: {len(data['pos'])+len(data['neg'])} | pos: {len(data['pos'])} | neg: {len(data['neg'])}")
    length = min(len(data['pos']), len(data['neg']))
    datas.extend(data['pos'][:length])
    datas.extend(data['neg'][:length])
    return datas

def read_wiki_split(data_path):
    datas = []
    with open(data_path,"r+")as f:
        lines = f.read().splitlines()
        for line in lines:
            datas.append(line)
    logger.info(f"Total data: {len(datas)}")
    return datas

def recenter(x, mean=None):
    if mean is None:
        mean = x.mean(axis=0, keepdims=True)
    return x - mean

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama-7b',
                        )
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--method', type=str, default='mean')
    parser.add_argument('--tok_position', type=str, default='last', choices=['last', 'mean'])
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--save_hidden', action='store_true')
    args = parser.parse_args()

    data_file = args.data_file
    if 'wiki' in data_file:
        datas = read_wiki_split(data_file)
    else:
        datas = read_data(data_file)            # first half is positive, second half is negative
        logger.info(f"Pos data sample:\n {datas[:3]}\nNeg data sample:\n {datas[len(datas)//2:len(datas)//2+3]}")

    model_path = model_name2path[args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16)

    base_dir = 'vectors'
    kind = args.data_file.split('/')[-1].split('.')[0]
    

    all_layers_hidden_states = []
    for data in tqdm(datas):
        prompt = tokenizer(data, return_tensors="pt")
        hidden_states, layer_wise_hidden_states = get_activations_bau(model, prompt['input_ids'], args.model_name)
        if args.tok_position == 'last':
            all_layers_hidden_states.append(layer_wise_hidden_states[:, -1, :])
        elif args.tok_position == 'mean':
            all_layers_hidden_states.append(layer_wise_hidden_states.mean(axis=1))    # shape: (num_layers, hidden_size)

    len_pos = len(all_layers_hidden_states)//2
    logger.info(f"Total data: {len(all_layers_hidden_states)} | pos: {len_pos}")
    logger.info(f"all_layers_mlp_hidden_states.shape: {all_layers_hidden_states[0].shape}")

    pos_mlp_hidden_states = torch.stack(all_layers_hidden_states[:len_pos], dim=0)  
    neg_mlp_hidden_states = torch.stack(all_layers_hidden_states[len_pos:], dim=0)
    
    diff_vectors = pos_mlp_hidden_states - neg_mlp_hidden_states                # shape: (num_privacy, num_layers, hidden_size)
    if args.method == 'mean':
        steer_vectors = diff_vectors.mean(axis=0)       # shape: (num_layers, hidden_size)
    elif args.method == 'pca':
        directions = []
        layers = diff_vectors.shape[1]
        for l in range(layers):
            h_train = diff_vectors[:,l,:]
            h_train_mean = h_train.mean(axis=0,keepdim=True) 
            h_train = recenter(h_train, h_train_mean)
            pca_model = PCA(n_components=1, whiten=False).fit(h_train)
            directions.append(pca_model.components_[0,:])
        steer_vectors = np.array(directions)
    else:
        raise NotImplementedError

    if not 'wiki' in data_file:
        patch_dir = os.path.join(base_dir, 'patch_vectors')
        if not os.path.exists(patch_dir):
            os.makedirs(patch_dir, exist_ok=True)
        kind = kind.replace('refine_', '')
        save_file = f"{patch_dir}/{args.model_name}_{kind}_{args.method}.npy"
        steer_vectors = steer_vectors.to(torch.float)
        np.save(save_file, steer_vectors)
        logger.info(f"{save_file}")

    if args.save_hidden:
        hidden_dir = os.path.join(base_dir, 'hidden_states')
        if not os.path.exists(hidden_dir):
            os.makedirs(hidden_dir, exist_ok=True)
        if 'wiki' in data_file:
            ## wiki split
            vectors = torch.stack(all_layers_hidden_states, dim=0)
            save_file = f"{hidden_dir}/{args.model_name}_{kind}_all.npy"
            vectors = vectors.detach().to(torch.float).cpu().numpy()
            np.save(save_file, vectors)
        else:
            vectors = pos_mlp_hidden_states.to(torch.float).cpu().numpy()
            save_file = f"{hidden_dir}/{args.model_name}_{kind}_pos.npy"
            np.save(save_file, vectors)

            vectors = neg_mlp_hidden_states.to(torch.float).cpu().numpy()
            save_file = f"{hidden_dir}/{args.model_name}_{kind}_neg.npy"
            np.save(save_file, vectors)

 
if __name__ == "__main__":
    main()