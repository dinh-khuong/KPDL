
import polars as pl
from vi_tokenizer import vietnamese_tokenizer

#|%%--%%| <sHFjmMYFy9|MXTaYOiSGj>

test_df = pl.read_csv("~/Downloads/articles_testing.tsv", separator="\t")
train_df = pl.read_csv("data/train_data.tsv", separator="\t")

#|%%--%%| <MXTaYOiSGj|eRjb8vy4lU>

train_test_df = train_df.select('content').extend(test_df)
train_test_df.write_parquet("data/combine.parquet")

#|%%--%%| <eRjb8vy4lU|3SaLFoRXhx>

eval_keyword_df = pl.DataFrame(
    {
        'keywords': test_df['content'].map_elements(lambda text: vietnamese_tokenizer(text), return_dtype=pl.List(pl.String))
    }
)

#|%%--%%| <3SaLFoRXhx|7PIwfUobmY>

eval_keyword_df.write_parquet("data/eval_keywords.parquet")
