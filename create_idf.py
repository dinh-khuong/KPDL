from sklearn.feature_extraction.text import CountVectorizer
import polars as pl
import numpy as np
from vi_tokenizer import vietnamese_tokenizer

def split_keywords(data_dir: str) -> pl.DataFrame:
    train_data = pl.read_csv("{}/train_data.tsv".format(data_dir), separator="\t")

    keywords = train_data["content"].map_elements(
        lambda content: [keyword for keyword in vietnamese_tokenizer(content)],
        return_dtype=pl.List(pl.String),
    )

    keywords_df = keywords.to_frame("keywords")
    keywords_df.write_parquet("{}/keywords.parquet".format(data_dir))

    return keywords_df

def create_idf_table(data_dir: str) -> pl.DataFrame:
    keyword_df = pl.read_parquet("{}/keywords.parquet".format(data_dir))
    documents = keyword_df["keywords"]

    keywords_freq = None
    vectorizer = CountVectorizer(
        lowercase=False, tokenizer=lambda text: text, token_pattern=None
    )

    X = vectorizer.fit_transform(documents)

    doc_freq = X.sum(axis=0)

    keywords_freq = pl.DataFrame(
        {
            "keywords": vectorizer.get_feature_names_out(),
            "frq": doc_freq.tolist()[0],
        }
    )

    keywords_freq.write_parquet("{}/keywords_freq.parquet".format(data_dir))

    num_of_document = len(keyword_df)

    token_idf_df = pl.DataFrame(
        {
            "keywords": keywords_freq["keywords"],
            "idf": np.log(num_of_document / keywords_freq["frq"].to_numpy()),
        }
    )

    token_idf_df.write_parquet("{}/keywords_idf.parquet".format(data_dir))

    return token_idf_df

# create_idf_table("data")
