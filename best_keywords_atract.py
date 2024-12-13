
from typing import Any
import polars as pl
import numpy as np

from vi_tokenizer import vietnamese_tokenizer

#|%%--%%| <ty1qEFFD5d|W9tMCQACtz>

keywords_df = pl.read_parquet("data/keywords.parquet")
keyword_idf = pl.read_parquet("data/keywords_idf.parquet")

#|%%--%%| <W9tMCQACtz|tk7QV7kXty>

def calculate_tf(tokens: list[str]):
    total_words = len(tokens)
    word_count = {}
    for token in tokens:
        word_count[token] = word_count.get(token, 0) + 1

    for counter in word_count.items():
        word_count[counter[0]] = word_count[counter[0]] / total_words

    # values = list(word_count.items())

    # return word_count.values() 
    return word_count


def list_to_dict(keys: list[Any], values: list[Any]) -> dict[Any, Any]:
    new_dict = {}
    for key, value in zip(keys, values):
        new_dict[key] = value

    return new_dict

def calc_tf_idf(term_frq: dict[str, int], idf_df: pl.DataFrame):
    term_frq_items = term_frq.items()
    tokens: list[str] = [token[0] for token in term_frq_items]
    idf_df_filterd = idf_df.filter(idf_df["keywords"].is_in(tokens))
    tokens = idf_df_filterd["keywords"].to_list()
    idf = idf_df_filterd["idf"].to_list()
    tokens_idf = list_to_dict(tokens, idf)

    ans = {}
    for term in term_frq_items:
        idf = tokens_idf.get(term[0], 1)

        ans[term[0]] = float(term[1]) * float(idf)


    return sorted(list(ans.items()), key=lambda x: x[1], reverse=True)

def get_best_keyword(keywords: list[tuple[str, float]], top_k: int = 20) -> list[str]:
    return list(map(lambda x: x[0], keywords[:top_k]))

def get_best_threshold(keywords: list[tuple[str, float]], threshold: float = 0.04) -> list[str]:
    top_k = 0
    for idx, (_, tf_idfd) in enumerate(keywords):
        if tf_idfd <= threshold:
            top_k = idx
            break

    return list(map(lambda x: x[0], keywords[:top_k]))

#|%%--%%| <tk7QV7kXty|ngYoUVqorE>

# train_data = pl.read_csv("~/Downloads/articles_testing.tsv", separator="\t")
train_data = pl.read_csv("data/train_data.tsv", separator="\t")
ouput_df = pl.DataFrame({
    "content": train_data['content'],
    "tags": keywords_df['keywords'].map_elements(lambda x: get_best_keyword(calc_tf_idf(calculate_tf(x), keyword_idf)), return_dtype=pl.Array)
})

#|%%--%%| <ngYoUVqorE|SF2KCTFwz5>

ouput_df_formated = ouput_df.with_columns(
    tags = ouput_df['tags'].map_elements(lambda x: ",".join(x), return_dtype=pl.String)
)
ouput_df_formated.write_csv("data/train_ans.tsv", separator="\t", include_header=True)

#|%%--%%| <SF2KCTFwz5|x3oo7GiKI5>

train_data_tags = train_data.with_columns(
    tags = train_data["tags"].map_elements(lambda x: x.split(","), return_dtype=pl.Array)
)
# train_data
#|%%--%%| <x3oo7GiKI5|xdgVCyxZIj>

def eval_tag(real_tag: list[str], pred_tag: list[str]):
    count = 0
    for tag in real_tag:
        count += tag in pred_tag

    return count / len(real_tag)

eval_list = []
for real, pred in zip(train_data_tags['tags'], ouput_df['tags']):
    eval_list.append(eval_tag(real, pred))

#|%%--%%| <xdgVCyxZIj|cUITqPzKIF>

eval_list_np = np.array(eval_list)

#|%%--%%| <cUITqPzKIF|fEgPnr0lz7>

eval_list_np.mean()

#|%%--%%| <fEgPnr0lz7|IuxbLMFzKi>

import tokenizer

input_text = train_data['content'][0]
for token in tokenizer.tokenize(input_text):
    kind, txt, val = token
    if kind == tokenizer.TOK.PERSON:
        print(txt.title())
        # Do something with word tokens
        pass
    else:
        # Do something else
        pass

# print(tokenizer.tokenize(train_data['content'][0]))

#|%%--%%| <IuxbLMFzKi|y7WeBWqqt8>


from transformers import AutoTokenizer

# Load a pre-trained Vietnamese tokenizer
tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

# Tokenize a Vietnamese text
# text = "Xin chào, thế giới!"
tokens = tokenizer.tokenize(input_text)
print(tokens)
