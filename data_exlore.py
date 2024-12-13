from typing import Any
import polars as pl
import numpy as np

# |%%--%%| <KRpOw3w1Hj|JFcM35XTKt>

train_data_df = pl.scan_csv("data/train_data.tsv", separator="\t").collect()

#|%%--%%| <JFcM35XTKt|d0oj1F8Pmp>

print(train_data_df['tags'][0])

# |%%--%%| <d0oj1F8Pmp|efCNZPV75l>

keywords_df = pl.read_parquet("data/keywords.parquet")

print(keywords_df)

# |%%--%%| <efCNZPV75l|d97Tr4L6vw>

for keyword in train_data_df["content"].head(1).to_list():
    print(keyword)

# |%%--%%| <d97Tr4L6vw|8lOAXRHv00>

from vi_tokenizer import vietnamese_tokenizer


# |%%--%%| <8lOAXRHv00|Ce8NxPTzER>

# Sample documents
documents = train_data_df["content"]

# Initialize CountVectorizer with binary=True to count presence (1 if word is present)
# vectorizer = CountVectorizer(binary=True, lowercase=False, tokenizer=tokenizer)
vectorizer = CountVectorizer(
    lowercase=False, tokenizer=vietnamese_tokenizer, token_pattern=None
)

# Fit the documents to the vectorizer and transform them into a document-term matrix
X = vectorizer.fit_transform(documents)

# Get the document frequency for each word
doc_freq = X.sum(axis=0)

# |%%--%%| <Ce8NxPTzER|jGt3qDtdlL>

# type(doc_freq)
# for keyword in vectorizer.get_feature_names_out():
#     print(keyword)
# print(vectorizer.get_feature_names_out())

# |%%--%%| <jGt3qDtdlL|ZMmQiqw2l4>

keywords_freq = pl.DataFrame(
    {
        "keywords": vectorizer.get_feature_names_out(),
        "frq": doc_freq.tolist()[0],
    }
)

# |%%--%%| <ZMmQiqw2l4|nLv5KOrGVY>

keywords_freq

# |%%--%%| <nLv5KOrGVY|OxlxYDfRAs>

keywords_freq.write_parquet("data/keywords_freq.parquet")

# |%%--%%| <OxlxYDfRAs|7dqCGytjYv>
r"""°°°
## Inverse document freqency
°°°"""
# |%%--%%| <7dqCGytjYv|eHWnUq1nXz>

num_of_document = len(train_data_df)

token_idf_df = pl.DataFrame(
    {
        "keywords": keywords_freq["keywords"],
        "idf": np.log(num_of_document / keywords_freq["frq"].to_numpy()),
    }
)

token_idf_df.write_parquet("data/keywords_idf.parquet")

# |%%--%%| <eHWnUq1nXz|zivrim5zNr>

token_idf_df.sort(by="idf", descending=True)

# |%%--%%| <zivrim5zNr|GFFf0vvA9b>

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


#|%%--%%| <GFFf0vvA9b|HKZKkUVvmu>

print(calc_tf_idf(calculate_tf(keywords_df['keywords'][0]), token_idf_df))

# |%%--%%| <HKZKkUVvmu|OiDPljdbxv>

# calculate_tf(keywords_df['keywords'][0])
keywords_tf_idf = (
    keywords_df["keywords"]
    .head(5)
    .map_elements(
        lambda keywords: calc_tf_idf(calculate_tf(keywords), token_idf_df), return_dtype=pl.Object
    )
)

keywords_tf_idf

#|%%--%%| <OiDPljdbxv|zWA54g5It5>

keywords_tf_idf = pl.read_parquet("data/keywords_idf.parquet")

print(keywords_tf_idf)

#|%%--%%| <zWA54g5It5|GMZCNRcRMj>


