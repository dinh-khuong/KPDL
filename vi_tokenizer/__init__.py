import re
from spacy.lang.vi import Vietnamese

nlp = Vietnamese()

date_pattern = r"""
(
[nN]gày\s*[-/.]?\s*\d{1,2}\s*[-/.]?\s*[tT]háng\s*[-/.]?\s*\d{1,2}\s*[-/.]?\s*[nN]ăm\s*[-/.]?\s*\d{1,}|
[nN]gày\s\d{1,2}[-/.]\d{1,2}[-/.]\d{1,}|
[nN]gày\s\d{1,2}[-/.]\d{1,2}|
\d{1,2}[-/.]\d{1,2}[-/.]\d{1,}|
[nN]gày\s*\d{1,2}\s*[tT]háng\s\d{1,2}|
[nN]gày\s*\d{1,2}|
[tT]háng\s*\d{1,2}|
[nN]ăm\s*\d{1,}
)
""".replace("\n", "")

date_regex = re.compile(date_pattern)

def get_date_vietnamese(text: str) -> list[str]:
    return date_regex.findall(text)

def vietnamese_tokenizer(text: str) -> list[str]:
    doc = nlp(text)
    date = get_date_vietnamese(text)
    keywords = [keyword.text.replace("_", " ") for keyword in doc if not keyword.is_stop and not keyword.is_punct]
    keywords.extend(date)

    return keywords


# from spacy.tokens import Doc, Span
# from spacy.language import Language
# from spacy.util import filter_spans
# import dateparser

# ordinal_to_number = {
#     "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
#     "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
#     "eleventh": "11", "twelfth": "12", "thirteenth": "13", "fourteenth": "14",
#     "fifteenth": "15", "sixteenth": "16", "seventeenth": "17", "eighteenth": "18",
#     "nineteenth": "19", "twentieth": "20", "twenty-first": "21", "twenty-second": "22",
#     "twenty-third": "23", "twenty-fourth": "24", "twenty-fifth": "25", "twenty-sixth": "26",
#     "twenty-seventh": "27", "twenty-eighth": "28", "twenty-ninth": "29", "thirtieth": "30", 
#     "thirty-first": "31"
# }

# @Language.component("find_dates")
# def find_dates(doc: Doc):
#     # Set up a date extension on the span
#     Span.set_extension("date", default=None, force=True)

#     # A regex pattern to capture a variety of date formats
#     matches = list(date_regex.finditer(doc.text, re.VERBOSE))
#     new_ents = []
#     for match in matches:
#         start_char, end_char = match.span()
#         # Convert character offsets to token offsets
#         start_token = None
#         end_token = None
#         for token in doc:
#             if token.idx == start_char:
#                 start_token = token.i
#             if token.idx + len(token.text) == end_char:
#                 end_token = token.i
#         if start_token is not None and end_token is not None:
#             hit_text = doc.text[start_char:end_char]
#             parsed_date = dateparser.parse(hit_text)
#             if parsed_date:  # Ensure the matched string is a valid date
#                 ent = Span(doc, start_token, end_token + 1, label="DATE")
#                 ent._.date = parsed_date
#                 new_ents.append(ent)
#             else:
#                 # Replace each ordinal in hit_text with its numeric representation
#                 for ordinal, number in ordinal_to_number.items():
#                     hit_text = hit_text.replace(ordinal, number)

#                 # Remove the word "of" from hit_text
#                 new_date = hit_text.replace(" of ", " ")

#                 parsed_date = dateparser.parse(new_date)
#                 ent = Span(doc, start_token, end_token + 1, label="DATE")
#                 ent._.date = parsed_date
#                 new_ents.append(ent)
#     # Combine the new entities with existing entities, ensuring no overlap
#     doc.ents = list(doc.ents) + new_ents
    
#     return doc
