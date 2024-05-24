import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaForCausalLM, LlamaConfig
import argparse
import logging
import os
import json
from tqdm import tqdm
from util.utils import model_name2path, get_interveted_output
from collections import defaultdict


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_data(data_path):
    datas = []
    with open(data_path,"r+")as f:
        data = json.load(f)
    logger.info(f"Total data: {len(data['pos'])+len(data['neg'])} | pos: {len(data['pos'])} | neg: {len(data['neg'])}")
    length = min(len(data['pos']), len(data['neg']))
    datas.extend(data['pos'][:length])
    datas.extend(data['neg'][:length])
    return datas

def recenter(x, mean=None):
    if mean is None:
        mean = x.mean(axis=0, keepdims=True)
    return x - mean

def get_rank1_logit(output):
    logits = output.logits[0,-1,:]
    probs = torch.softmax(logits, dim=-1)
    top1_token_id = probs.argmax().item()
    return top1_token_id, logits[top1_token_id]

def get_rank1_emo_indice(output, emo_token_ids):
    emo_logits = output.logits[0, -1, emo_token_ids]
    probs = torch.softmax(emo_logits, dim=-1)
    emo_max = probs.argmax().item()
    return emo_max, emo_logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama-7b',)
    parser.add_argument('--source_model', type=str, default=None)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--method', type=str, default='mean')
    parser.add_argument('--tok_position', type=str, default='last', choices=['last', 'mean'])
    parser.add_argument('--emotion', type=str, default='all', choices=['all','happiness', 'sadness', 'anger', 'fear', 'disgust', 'surprise'])
    parser.add_argument('--intervention', type=str, default='steer', choices=['identity', 'steer'])
    parser.add_argument('--patching_vectors_path', type=str, default=None)
    parser.add_argument('--strength', type=float, default=1.0)
    parser.add_argument('--generate', action='store_true')
    args = parser.parse_args()

    model_path = model_name2path[args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_path, device_map='auto')

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto',torch_dtype=torch.bfloat16)

    device = model.device

    if args.emotion == 'all':
        target_emos = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
    else:
        target_emos = [args.emotion]
    
    method = args.method
    results_all = defaultdict(dict)
    for target_emo in target_emos: 

        data_file = f'data/emotions/eval_{target_emo.strip()}.json'
        datas = read_data(data_file)            # first half is positive, second half is negative

        if args.patching_vectors_path is None and args.intervention == 'steer':
            raise ValueError("Please provide patching_vectors_path for steer intervention")
        elif args.patching_vectors_path == 'identity':
            patching_vectors_path = None
        else:
            patching_vectors_path = args.patching_vectors_path
                    
        if 'llama' in args.model_name:
            emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
            emo_token_ids = [tokenizer.encode(emo)[1] for emo in emotions]
        elif 'pythia' in args.model_name:
            target_emo = ' '+target_emo
            emotions = [" happiness", " sadness", " anger", " fear", " disgust", " surprise"]
            emo_token_ids = [tokenizer.encode(emo)[0] for emo in emotions]
        else:
            raise ValueError(f"model_name: {args.model_name} not supported")
        target_emo_id = emotions.index(target_emo)

        result_dict = defaultdict(dict)
            

        if '65b' in args.model_name:
            windows = [60]
        elif '13b' in args.model_name:
            windows = [30]
        elif '7b' in args.model_name:
            windows = [24]
        

        results = defaultdict(int)

        rights_rank1 = []
        rights_emo = []
        for data in tqdm(datas[:len(datas)//2]):
            prompt = tokenizer(data, return_tensors="pt")
            inputs = prompt['input_ids'].to(device)

            output = get_interveted_output(model,inputs, args.model_name, layer_ids = windows,
                                             alpha=args.strength,intervene_method=args.intervention,
                                            steer_vector_path=patching_vectors_path,)

            top1_token_id, top1_logits = get_rank1_logit(output)   
            tar = emo_token_ids[target_emo_id]
            if top1_token_id == tar:
                rights_rank1.append(1)
                rights_emo.append(1)
            else:
                rights_rank1.append(0)
                emo_max_indice, emo_logits = get_rank1_emo_indice(output, emo_token_ids)
                results[emotions[emo_max_indice]] += 1
                rights_emo.append(emo_max_indice == target_emo_id)

        print(f"{target_emo} pos Token Acc: {sum(rights_rank1)/len(rights_rank1)}")
        print(f"{target_emo} pos Logit Acc: {sum(rights_emo)/len(rights_emo)}")


        result_dict['pos']['token'] = sum(rights_rank1)/len(rights_rank1)
        result_dict['pos']['logit'] = sum(rights_emo)/len(rights_emo)


        rights_rank1 = []
        rights_emo = []
        for data in tqdm(datas[len(datas)//2:]):
            prompt = tokenizer(data, return_tensors="pt")
            inputs = prompt['input_ids'].to(device)

            output = get_interveted_output(model,inputs, args.model_name, layer_ids = windows,
                                             alpha=args.strength,intervene_method=args.intervention,
                                            steer_vector_path=patching_vectors_path,)

            top1_token_id, top1_logits = get_rank1_logit(output)   
            tar = emo_token_ids[target_emo_id]
            if top1_token_id == tar:
                rights_rank1.append(1)
                rights_emo.append(1)
            else:
                rights_rank1.append(0)
                # get emo logits
                emo_max_indice, emo_logits = get_rank1_emo_indice(output, emo_token_ids)
                results[emotions[emo_max_indice]] += 1
                rights_emo.append(emo_max_indice == target_emo_id)

        print(f"{target_emo} neg acc of Token Acc: {sum(rights_rank1)/len(rights_rank1)}")
        print(f"{target_emo} neg acc of Logit Acc: {sum(rights_emo)/len(rights_emo)}")


        result_dict['neg']['token'] = sum(rights_rank1)/len(rights_rank1)
        result_dict['neg']['logit'] = sum(rights_emo)/len(rights_emo)
        
        results_all[target_emo.strip()] = result_dict
        

        output_dir = 'eval_results'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if args.intervention == 'steer':
            vector = patching_vectors_path.split('/')[-1].split('.')[0] 
            out_filename = f"eval_results/{args.model_name}_{args.emotion}_eval_{args.intervention}_{vector}_s{args.strength}.json"
        elif args.intervention == 'identity':
            out_filename = f"eval_results/{args.model_name}_{args.emotion}_eval_{args.intervention}.json"

        with open(out_filename, "w+") as f:
            json.dump(results_all, f, indent=4)
            
        
if __name__ == "__main__":
    main()
