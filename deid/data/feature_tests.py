import numpy as np

from .feature import CaseFeature, apply_features
from .token import Token


def test_case():
    feature = CaseFeature()
    assert np.all(feature.apply(Token.with_text('1234')) == np.array([0, 1, 0, 0, 0, 0, 0]))  # all numeric
    assert np.all(feature.apply(Token.with_text('123a')) == np.array([0, 0, 1, 0, 0, 0, 0]))  # mainly numeric
    assert np.all(feature.apply(Token.with_text('ok4y')) == np.array([0, 0, 0, 1, 0, 0, 0]))  # all lower
    assert np.all(feature.apply(Token.with_text('OKAY')) == np.array([0, 0, 0, 0, 1, 0, 0]))  # all upper
    # ...


def test_apply_features():
    features = [CaseFeature()]
    case_features = apply_features(features, [Token.with_text('UPPER'), Token.with_text('CASE')])
    assert len(case_features) == 2
    assert np.all(case_features[0] == np.array([0, 0, 0, 0, 1, 0, 0]))

    features = [CaseFeature(), CaseFeature()]
    case_features = apply_features(features, [Token.with_text('UPPER'), Token.with_text('CASE')])
    print(case_features)
    assert np.all(case_features[0] == np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]))
