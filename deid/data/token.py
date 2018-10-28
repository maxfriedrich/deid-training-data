from typing import NamedTuple
from ..tools.i2b2.classes import PHITrackEvaluation


class Token(NamedTuple):
    text: str
    type: str
    start: int
    end: int

    @classmethod
    def with_text(cls, text, label='O'):
        """ Mostly useful for unit tests """
        return Token(text, label, 0, 0)


# noinspection SpellCheckingInspection
TOKEN_TYPE = {
    'PATIENT': 'NAME',
    'DOCTOR': 'NAME',
    'USERNAME': 'NAME',
    'PROFESSION': 'PROFESSION',
    'ROOM': 'LOCATION',
    'DEPARTMENT': 'LOCATION',
    'HOSPITAL': 'LOCATION',
    'ORGANIZATION': 'LOCATION',
    'STREET': 'LOCATION',
    'CITY': 'LOCATION',
    'STATE': 'LOCATION',
    'COUNTRY': 'LOCATION',
    'ZIP': 'LOCATION',
    'LOCATION-OTHER': 'LOCATION',
    'AGE': 'AGE',
    'DATE': 'DATE',
    'PHONE': 'CONTACT',
    'FAX': 'CONTACT',
    'EMAIL': 'CONTACT',
    'URL': 'CONTACT',
    'IPADDR': 'CONTACT',
    'SSN': 'ID',
    'MEDICALRECORD': 'ID',
    'HEALTHPLAN': 'ID',
    'ACCOUNT': 'ID',
    'LICENSE': 'ID',
    'VEHICLE': 'ID',
    'DEVICE': 'ID',
    'BIOID': 'ID',
    'IDNUM': 'ID',
    'OTHER': 'OTHER'
}

HIPAA_TOKEN_TYPE = {tag: type for tag, type in TOKEN_TYPE.items() if any([n_re.match(type) and t_re.match(tag)
                                                                         for n_re, t_re in
                                                                         PHITrackEvaluation.HIPAA_regexes])}

BINARY_LABEL = 'PATIENT'
