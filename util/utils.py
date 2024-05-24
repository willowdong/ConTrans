from baukit import TraceDict
from functools import partial
import torch
import numpy as np

layer_name_temp = {
    'llama-7b':"model.layers.{}",
    'llama-13b':"model.layers.{}",
    'llama-65b':"model.layers.{}",
    'llama2-7b':"model.layers.{}",
    'llama2-7b-chat':"model.layers.{}",
    'llama2-13b':"model.layers.{}",
    'llama2-13b-chat':"model.layers.{}",
    'llama2-70b-chat':"model.layers.{}",
    'llama2-70b':"model.layers.{}",
    'pythia-14m':"gpt_neox.layers.{}",
    'pythia-70m':"gpt_neox.layers.{}",
    'pythia-410m':"gpt_neox.layers.{}",
    'pythia-1.4b':"gpt_neox.layers.{}",
    'pythia-6.9b':"gpt_neox.layers.{}",
    'gpt2-small':"transformer.h.{}",
    'gpt2-xl':"transformer.h.{}",
    'gpt-neo':"transformer.h.{}",
}

model_name2path = {
    'llama-7b': '/models/llama7b-hf',
    'llama-13b': '/llama/llama-13b-hf',
    'llama-65b':'/models/llama-65b-hf/',
    'llama2-7b':'/llama-2/llama-2-7b-hf',
    'llama2-13b':'/llama-2/llama-2-13b-hf',
    'llama2-7b-chat':'/llama-2/llama-2-7b-chat-hf',
    'llama2-13b-chat':'/llama-2/llama-2-13b-chat-hf',
    'llama2-70b-chat':'/models/llama-2-70b-chat-hf',
    'llama2-70b': '/models/llama-2-70b',
    'pythia-14m':'/models/pythia-14m',
    'pythia-70m':'/models/pythia-70m',
    'pythia-410m':'/models/pythia-410m',
    'pythia-1.4b':'/models/pythia-1.4b',
    'pythia-6.9b':'/models/pythia-6.9b',
    'gpt2-small':'/models/gpt2',
    'llama2-7b-chat':'/llama-2/llama-2-7b-chat-hf',
    'gpt2-xl':'/models/gpt2-xl',
    'gpt-neo': '/models/gpt-neo-2.7B',
    'amber-7b':"/models/amber/",
    'amber-7b-ckpt300':"/models/Amber-ckpt300/",
    'amber-7b-chat':"/models/amber-chat/",
}

file_suffix = {
    'llama7b': 'llama7b',
    'gpt2-small':'small',
    'gpt2-xl':'xl',
    'gpt-neo':'neo',
}

def intervene_fn_zero(original_state, layer, model_name, layer_ids,):
    layers_to_intervene = [layer_name_temp[model_name].format(i) for i in layer_ids]
    if layer not in layers_to_intervene:
        return original_state
    if layer in layers_to_intervene:
        x = original_state
        if isinstance(original_state, tuple):
            original_state = x[0]
        original_state = 0 
    if isinstance(x, tuple):
        return (original_state, x[1])

    return original_state

def intervene_fn_neg(original_state, layer, model_name, layer_ids,):
    layers_to_intervene = [layer_name_temp[model_name].format(i) for i in layer_ids]
    if layer not in layers_to_intervene:
        return original_state
    if layer in layers_to_intervene:
        x = original_state
        if isinstance(original_state, tuple):
            original_state = x[0]
        original_state = -original_state 
    if isinstance(x, tuple):
        return (original_state, x[1])

    return original_state

def intervene_fn_add(original_state, layer, model_name, layer_ids, 
                     steer_vector_path=None, steer_vector=None, alpha=1):
    layers_to_intervene = [layer_name_temp[model_name].format(i) for i in layer_ids]
    if layer not in layers_to_intervene:
        return original_state

    assert steer_vector_path is not None or steer_vector is not None
    if steer_vector is not None:                # prefer steer_vector
        steer_vector_tensor = steer_vector  
    else:                                       # load steer_vector from path otherwise
        steer_vector_tensor = np.load(steer_vector_path)
    layers = steer_vector_tensor.shape[0]
    assert layers > 0
    steer_vector_dict = {}
    for i in range(layers):
        steer_vector_dict[layer_name_temp[model_name].format(i)] = steer_vector_tensor[i]

    if layer in layers_to_intervene:
        x = original_state
        if isinstance(original_state, tuple):
            original_state = x[0]
        window_steer_vector = torch.tensor(steer_vector_dict[layer], device=original_state.device)
        original_state += alpha*window_steer_vector
        del window_steer_vector
    if isinstance(x, tuple):
        return (original_state, x[1])

    return original_state

def intervene_fn_identity(original_state, layer):
    return original_state

def get_interveted_output(model, inputs, model_name,
                            layer_ids,
                            intervene_method, steer_vector_path, 
                            alpha=1, generate=False,
                            steer_vector=None,):
    '''
    layer_ids: list of layer ids to intervene
    alpha: scalar, strength of intervention
    '''
    layers_to_intervene = [layer_name_temp[model_name].format(i) for i in layer_ids]
    if intervene_method == 'suppress':
        intervene_fn = partial(intervene_fn_zero, layer_ids=layer_ids)
    elif intervene_method == 'neg':
        intervene_fn = partial(intervene_fn_neg, layer_ids=layer_ids)
    elif intervene_method == 'steer':
        intervene_fn = partial(intervene_fn_add, model_name=model_name, layer_ids=layer_ids,
                               steer_vector_path=steer_vector_path,steer_vector=steer_vector,alpha=alpha)
    elif intervene_method == 'identity':
        intervene_fn = intervene_fn_identity
        layers_to_intervene = []
    else:
        raise NotImplementedError

    with TraceDict(model, layers_to_intervene, edit_output=intervene_fn) as ret: 
        if not generate:
            if isinstance(inputs, dict):
                outputs = model(**inputs, labels=inputs['input_ids'])
            elif isinstance(inputs, torch.Tensor):
                outputs = model(inputs)
        else:
                outputs = model.generate(  
                        **inputs,
                        max_new_tokens=250,
                        # temperature=0.75,
                        # top_p=0.95, 
                        # top_k=50,
                        do_sample=True,
                        num_return_sequences=1,
                        pad_token_id=2,
                    )
    return outputs


if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import numpy as np

    model_path = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    steer_vector_path = '...'
    steer_vector = np.ones((12,768))
    layer_id = [0,2]

    inputs = tokenizer('hello world', return_tensors="pt")

    output_intervened = get_interveted_output(model, inputs=inputs['input_ids'], model_name='gpt2-small',
                                              layer_ids =layer_id,
                                              intervene_method='steer', 
                                              steer_vector=steer_vector, steer_vector_path=steer_vector_path)