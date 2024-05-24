import torch
import numpy as np
import os


def apply_affine():
    '''
    LLAMA2
    7B: 4096
    13B: 5120
    70B: 8192
    '''

    source_shape = 4096
    target_shape = 5120
    target_layer_num = 40

    source_layers = [24]
    target_layers = [30]
    
    concept = 'anger'

    save_path = 'vectors/affine_weights/llama-7b_llama-13b_wiki_split'
    weight_path = 'vectors/affine_weights/llama-7b_llama-13b_svd_wiki_split'

    affine_svd = True
    base_dir = 'vectors/patch_vectors'
    device = 'cuda'

    input_diff_state = np.load(f'vectors/patch_vectors/llama-7b_{concept}_mean.npy')
    print(input_diff_state.shape)
    input_diff_state = torch.from_numpy(input_diff_state).float().to(device)
    
    example = torch.zeros((target_layer_num, target_shape))
    for source_layer, target_layer in zip(source_layers, target_layers):
        param_path = os.path.join(save_path, f'affine_{source_layer}_to_{target_layer}.pth')
        
        if affine_svd:
            W = torch.load(f'{weight_path}/affine_{source_layer}_to_{target_layer}.pth').to(device)
            output_diff_state = torch.mm(input_diff_state, W.T)
        else:
            pass

        example[target_layer, :] = output_diff_state[source_layer, :]
    
    example = example.cpu().detach().numpy()
    
    out_file = f'{base_dir}/llama-7to13-{concept}.npy'
    np.save(out_file, example)
    print(f'Save to {out_file}')

apply_affine()