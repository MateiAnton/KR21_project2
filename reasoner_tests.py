from BayesNet import *
from BNReasoner import *

IJXYO = "testing/lecture_example2.BIFXML"


def approximately_equal(a, b, epsilon=0.0001):
    return abs(a - b) < epsilon


def test_maximum_a_posteriori():
    bn = BayesNet(path=IJXYO)
    reasoner = BNReasoner(bn)
    prob, values = reasoner.maximum_a_posteriori(["I", "J"], {"O": True})
    assert approximately_equal(prob, 0.242720)
    assert values["J"] == False


def test_most_probable_explanation():
    bn = BayesNet(path=IJXYO)
    reasoner = BNReasoner(bn)
    prob, values = reasoner.most_probable_explanation({"J": True, "O": False})
    assert approximately_equal(prob, 0.230422)
    assert values["J"] == True
    assert values["I"] == False
    assert values["X"] == False
    assert values["Y"] == False
    assert values["O"] == False


test_maximum_a_posteriori()
test_most_probable_explanation()
