import html

import spacy
from spacy.matcher import Matcher
from spacy.tokens import Token


def _deid_tokenizer():
    prefixes = spacy.util.compile_prefix_regex(nlp.Defaults.prefixes)

    my_infix = ['\.\.\.+',
                '(?<=[0-9])-(?=[0-9])',
                '(?<=[0-9])(?=[A-Za-z])',
                '[!&:;#,()/_\\-\\^~%{}=\'<>@]']
    infixes = spacy.util.compile_infix_regex(list(nlp.Defaults.infixes) + my_infix)

    suffixes = spacy.util.compile_suffix_regex(nlp.Defaults.suffixes)

    return spacy.tokenizer.Tokenizer(nlp.vocab, nlp.Defaults.tokenizer_exceptions,
                                     prefix_search=prefixes.search,
                                     infix_finditer=infixes.finditer, suffix_search=suffixes.search,
                                     token_match=None)


def _new_sentence_after_three_newlines_component(doc):
    def has_newlines(text):
        return text.count('\n') > 2

    for i in range(len(doc[:-2])):
        if has_newlines(doc[i].text) and not has_newlines(doc[i + 1].text):
            doc[i + 1].sent_start = True
    return doc


def _new_sentence_for_bulleted_lists_component(doc):
    def has_newlines(text):
        return text.count('\n') > 0

    def is_bullet(text):
        return text.startswith('-') or text.startswith('*') or text.startswith('.') or text == 'o' or text[0].isdigit()

    for i in range(len(doc[:-2])):
        if has_newlines(doc[i].text) and not has_newlines(doc[i + 1].text) and is_bullet(doc[i + 1].text):
            doc[i + 1].sent_start = True
    return doc


def _new_sentence_after_three_dashes_component(doc):
    for i in range(3, len(doc[:-3])):
        if all(token.text == '-' for token in doc[i - 3:i]) and doc[i].text != '-':
            doc[i].sent_start = True

    return doc


# https://spacy.io/usage/linguistic-features#section-rule-based-matching
class _HTMLMerger(object):
    def __init__(self, nlp):
        Token.set_extension('unescaped_html', default=None)
        self.matcher = Matcher(nlp.vocab)
        self.matcher.add('BAD_HTML', None,
                         [{'ORTH': '<'}, {'LOWER': 'br'}, {'ORTH': '>'}],
                         [{'ORTH': '<'}, {'LOWER': 'br'}, {'ORTH': '/'}, {'ORTH': '>'}],
                         [{'ORTH': '&'}, {'SHAPE': 'xx'}, {'ORTH': ';'}],  # &lt;
                         [{'ORTH': '&'}, {'SHAPE': 'xxx'}, {'ORTH': ';'}],  # &amp;
                         [{'ORTH': '&'}, {'ORTH': '#'}, {'SHAPE': 'dd'}, {'ORTH': ';'}],  # &#12;
                         [{'ORTH': '&'}, {'ORTH': '#'}, {'SHAPE': 'ddd'}, {'ORTH': ';'}],
                         [{'ORTH': '&'}, {'ORTH': '#'}, {'SHAPE': 'dddd'}, {'ORTH': ';'}])

    def __call__(self, doc):
        matches = self.matcher(doc)
        spans = []
        for match_id, start, end in matches:
            spans.append(doc[start:end])
        for span in spans:
            span.merge()
            for token in span:
                if '<br' in token.text:
                    token._.unescaped_html = '\n'
                else:
                    token._.unescaped_html = html.unescape(token.text)
        return doc


nlp = spacy.load('en', disable=['ner'])
nlp.tokenizer = _deid_tokenizer()
nlp.add_pipe(_new_sentence_after_three_newlines_component, before='parser')
nlp.add_pipe(_new_sentence_for_bulleted_lists_component, before='parser')
nlp.add_pipe(_new_sentence_after_three_dashes_component, before='parser')
nlp.add_pipe(_HTMLMerger(nlp), last=True)


def tokenize(text):
    """ Converts a text to a spaCy document using a customized tokenizer.

    :param text: the input text
    :return: a spaCy Doc object
    """
    return nlp(text)
