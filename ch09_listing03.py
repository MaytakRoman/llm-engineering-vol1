"""Намеренно дефектная награда и отдельный скрытый валидатор."""
from fractions import Fraction

def flawed(answer):
    return float('42' in answer)

def checker(answer, expected):
    try:
        return Fraction(answer.strip()) == expected
    except (ValueError, ZeroDivisionError):
        return False

def run():
    candidates = ['42', '42 is mentioned but the answer is 0',
                  '84/2', '0042', '420', 'NaN', '1/0']
    expected = Fraction(42)
    for answer in candidates:
        print(repr(answer), flawed(answer), checker(answer, expected))
    assert flawed(candidates[1]) == 1
    assert not checker(candidates[1], expected)
    assert flawed('84/2') == 0
    assert checker('84/2', expected)
    assert checker('0042', expected)
    assert not checker('420', expected)
    assert not checker('NaN', expected)
    assert not checker('1/0', expected)

if __name__ == '__main__':
    run()
