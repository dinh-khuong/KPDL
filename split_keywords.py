import polars as pl
from vi_tokenizer import vietnamese_tokenizer

def main():
    train_data = pl.read_parquet("data/combine.parquet")

    keywords = train_data["content"].map_elements(
        lambda content: [keyword for keyword in vietnamese_tokenizer(content)],
        return_dtype=pl.List(pl.String),
    )

    keywords_df = keywords.to_frame("keywords")
    keywords_df.write_parquet("data/keywords.parquet")

if __name__ == "__main__":
    main()
