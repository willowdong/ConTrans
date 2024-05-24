import numpy as np

'''process wiki split'''
file_name = 'wiki_split.txt'
texts = []
with open(file_name, 'r') as f:
    for line in f.readlines():
        texts.append(line.strip())

template_str = "Pretend you're an honest person making statements about the world."
user_tag = "[INST] "
assistant_tag = "[/INST] "


max_cut_off_len = 5
min_cut_off_len = 2
random_seed = 42
np.random.seed(random_seed)
output_file = 'wiki_split_input_chat.txt'
with open(output_file, 'w') as f:
    for text in texts:
        cut_off = np.random.randint(min_cut_off_len, max_cut_off_len)
        splits = text.split(' ')
        if len(splits) <= min_cut_off_len:
            continue
        text = ' '.join(splits[:-cut_off])
        f.write(text+'\n')
