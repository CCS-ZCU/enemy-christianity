#precompute embeddings
import pandas as pd
import torch
from transformers import AutoModel
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm 

grouped = pd.read_pickle('../data/large-data/grouped_df.pkl')
torch.set_num_threads(16)

tokenizer = AutoTokenizer.from_pretrained("pranaydeeps/Ancient-Greek-BERT")
model = AutoModel.from_pretrained("pranaydeeps/Ancient-Greek-BERT")

def chunk_by_tokens(text, tokenizer, chunk_size=512):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]
    # Pad last chunk if necessary
    for i in range(len(chunks)):
        if len(chunks[i]) < chunk_size:
            chunks[i] += [tokenizer.pad_token_id] * (chunk_size - len(chunks[i]))
    return chunks

all_embeddings = []
for sentence in tqdm(grouped['lamma_sentence'], desc="Embedding sentences"):
    token_chunks = chunk_by_tokens(sentence, tokenizer)
    chunk_tensors = torch.tensor(token_chunks).to(model.device)
    # Create attention masks: 1 for real tokens, 0 for padding
    attention_masks = (chunk_tensors != tokenizer.pad_token_id).long()
    with torch.no_grad():
        outputs = model(chunk_tensors, attention_mask=attention_masks)
        chunk_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    agg_embedding = np.mean(chunk_embeddings, axis=0)
    all_embeddings.append(agg_embedding)

embeddings = np.array(all_embeddings)
np.save("../data/large-data/bert-embeddings.npy", embeddings)