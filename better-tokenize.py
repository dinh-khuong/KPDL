import spacy
from spacy.tokens.token import Token
import polars as pl

nlp = spacy.load("vi_core_news_lg")  # Load a language model
test_df = pl.read_csv("~/Downloads/articles_testing.tsv", separator="\t")

#|%%--%%| <vZ22wySIc7|d015LfySkO>

text = """Vào tối 22/10, sau đêm bán kết của cuộc thi Hoa hậu Hòa bình Quốc tế Miss Grand International 2024, ban tổ chức đã công bố 22 bộ trang phục Dân tộc đẹp nhất bước vào vòng 2."""
# test_df['content'][0]
# print(text)
doc = nlp(text)

on_number = False
on_noun = False
on_verb = False
token_idx = 0

# token_res = [{'text': ''}]
# t_index = 0

print(type(doc[0]))
# while token_idx < len(doc):
#     token = doc[token_idx]
#     # print(doc[token_idx])
#     if token.tag_.startwith('N'):
#         token_res[t_index].text += token
#
#
#
# for token in doc:
#     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
#             token.shape_, token.is_alpha, token.is_stop)

    # if token.tag_.startswith('N') and not on_number:
    #     on_noun = True
    #     token_res[t_index] += token.text
    # elif token.tag_.startswith('M'):
    #     on_noun = False
    #     on_number = True
    #     token_res[t_index] += token.text
    #
    # if not (token.tag_.startswith('N') or token.tag_.startswith('M')):
    #     on_number = False
    #     token_res.append('')
    #     t_index += 1
    #




    # print(token.tag_)
    # print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
    #         token.shape_, token.is_alpha, token.is_stop)

