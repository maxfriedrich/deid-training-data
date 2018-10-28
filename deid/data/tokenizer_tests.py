from . import tokenize


def assert_number_of_tokens(doc, count):
    assert len(doc) == count, f'token sequence {[str(t) for t in doc]} has length {len(doc)}, expected {count}'


def assert_number_of_sentences(doc, count):
    sents = list(str(sent) for sent in doc.sents)
    assert len(sents) == count, f'doc {sents} has {len(sents)} sentences, expected {count}'


def test_tokenize_one_sentence():
    doc = tokenize('A sentence that is simple to tokenize.')
    assert_number_of_tokens(doc, 8)
    assert doc[-1].text == '.'


def test_tokenize_multiple_sentences():
    doc = tokenize('One sentence. And another sentence.')
    assert_number_of_sentences(doc, 2)


def test_tokenize_phone_number():
    doc = tokenize('555-2394-72-01')
    assert_number_of_tokens(doc, 7)


def test_tokenize_custom_infixes():
    doc = tokenize('a/b')
    assert_number_of_tokens(doc, 3)

    doc = tokenize('a_b_c')
    assert_number_of_tokens(doc, 5)

    doc = tokenize('81-year-old')
    tokens = [str(t) for t in doc]
    assert tokens == ['81', '-', 'year', '-', 'old']

    doc = tokenize('a^b')
    tokens = [str(t) for t in doc]
    assert tokens == ['a', '^', 'b']

    doc = tokenize('25yo')
    tokens = [str(t) for t in doc]
    assert tokens == ['25', 'yo']


def test_tokenize_sentences():
    doc = tokenize('Here is some text that is followed by many newlines\n \n \n \n \nAnd here is some other text.')
    assert_number_of_sentences(doc, 2)

    doc = tokenize("""- First list item
    - and the second list item, which does not necessarily look like a sentence start.""")
    assert_number_of_sentences(doc, 2)

    doc = tokenize("""1. test
        2: ok""")
    assert_number_of_sentences(doc, 2)

    doc = tokenize("""----list with unusual format
    ----starting with some dashes, no space between dashes and first word
    ---sometimes it's a different number of dashes""")
    assert_number_of_sentences(doc, 6)


def test_tokenize_html():
    doc = tokenize('NASA &amp; SpaceX')
    tokens = [t for t in doc]
    assert [str(t) for t in tokens] == ['NASA', '&amp;', 'SpaceX']
    assert tokens[0]._.unescaped_html is None
    assert tokens[1]._.unescaped_html == '&'

    doc = tokenize('NASA &#62; SpaceX')
    tokens = [t for t in doc]
    assert [str(t) for t in tokens] == ['NASA', '&#62;', 'SpaceX']
    assert tokens[0]._.unescaped_html is None
    assert tokens[1]._.unescaped_html == '>'

    doc = tokenize('Nasa <br> SpaceX')
    tokens = [t for t in doc]
    assert tokens[1]._.unescaped_html == '\n'
