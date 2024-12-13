
import polars as pl
from keybert import KeyBERT

#|%%--%%| <wdv31Xd4qG|lyucD3sYXY>

doc = """
         Supervised learning is the machine learning task of learning a function that
         maps an input to an output based on example input-output pairs. It infers a
         function from labeled training data consisting of a set of training examples.
         In supervised learning, each example is a pair consisting of an input object
         (typically a vector) and a desired output value (also called the supervisory signal).
         A supervised learning algorithm analyzes the training data and produces an inferred function,
         which can be used for mapping new examples. An optimal scenario will allow for the
         algorithm to correctly determine the class labels for unseen instances. This requires
         the learning algorithm to generalize from the training data to unseen situations in a
         'reasonable' way (see inductive bias).
      """
kw_model = KeyBERT(model="paraphrase-multilingual-MiniLM-L12-v2")
keywords = kw_model.extract_keywords(doc)

#|%%--%%| <lyucD3sYXY|vBPSGRRk7R>

data = pl.read_csv("data/train_data.tsv", separator='\t')

#|%%--%%| <vBPSGRRk7R|N9ksaow0rB>

kw_model.extract_keywords(data['content'][0], top_n=20, keyphrase_ngram_range=(1,3))
# kw_model.extract_keywords(doc, top_n=20, keyphrase_ngram_range=(1,2))
