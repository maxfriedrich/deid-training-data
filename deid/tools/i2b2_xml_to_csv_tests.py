import os

from deid.env import Test
from .i2b2_xml_to_csv import xml_to_annotated_tokens_and_text


def find_token(tokens, text):
    return next((index, token) for index, token in enumerate(tokens) if token.text.startswith(text))


def test_xml_to_annotated_tokens_and_text():
    tokens, text = xml_to_annotated_tokens_and_text(os.path.join(Test().data_dir, 'train_xml', '999-99.xml'),
                                                    check_alignment=True)

    _, date_token = find_token(tokens, '2018')
    assert date_token.type == 'B-DATE'
    assert date_token.start == 16

    max_index, max_token = find_token(tokens, 'Max')
    assert max_token.type == 'B-PATIENT'
    assert max_token.start == 28
    assert max_token.end == 31

    assert tokens[max_index + 1].type == 'I-PATIENT'
    assert tokens[max_index + 2].type == 'O'

    lines = text.strip().split('\n')
    assert lines[0] == 'Record date: 2018-06-15'
    assert lines[2] == 'Max Friedrich is a 25-year-old Computer science student living in Hamburg, Germany.'


def test_tags_right_after_each_other():
    tokens, text = xml_to_annotated_tokens_and_text(os.path.join(Test().data_dir, 'train_xml', '999-98.xml'),
                                                    check_alignment=True)
    one_index, one_token = find_token(tokens, 'one')
    assert one_token.type == 'B-DATE'

    two_token = tokens[one_index + 1]
    assert two_token.type == 'B-AGE'

    three_token = tokens[one_index + 2]
    assert three_token.type == 'I-AGE'

    four_token = tokens[one_index + 3]
    assert four_token.type == 'B-DATE'

    medical_record_token = tokens[one_index + 4]
    assert medical_record_token.type == 'B-MEDICALRECORD'

    hospital_token = tokens[one_index + 5]
    print(hospital_token)
    assert hospital_token.type == 'B-HOSPITAL'


def test_uses_start_tag_even_with_wrong_alignment():
    tokens, text = xml_to_annotated_tokens_and_text(os.path.join(Test().data_dir, 'train_xml', '999-97.xml'),
                                                    check_alignment=True)
    print(tokens)
    zero_index, zero_token = find_token(tokens, 'zero')
    assert zero_token.type == 'O'

    one_token = tokens[zero_index + 1]
    assert one_token.type == 'B-DATE'

    two_token = tokens[zero_index + 2]
    assert two_token.type == 'B-AGE'  # not I-AGE

    three_token = tokens[zero_index + 3]
    assert three_token.type == 'O'

    four_token = tokens[zero_index + 4]
    assert four_token.type == 'B-DATE'


def test_escape_html():
    tokens, text = xml_to_annotated_tokens_and_text(os.path.join(Test().data_dir, 'train_xml', '999-96.xml'),
                                                    check_alignment=True)
    print(tokens)
    zero_index, zero_token = find_token(tokens, 'zero')
    assert zero_token.type == 'O'

    lt_token = tokens[zero_index + 1]
    assert lt_token.text == '<'
    assert lt_token.type == 'O'
    assert lt_token.start == 8

    one_token = tokens[zero_index + 2]
    assert one_token.type == 'B-DATE'
    assert one_token.start == 13
