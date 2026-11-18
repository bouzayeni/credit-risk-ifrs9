import re

from credit_risk.reporting import cards


def test_cards_render_from_committed_summaries_without_missing_values():
    s1, s3, s4, s5, s6, s7 = (cards.load(f"phase{i}_summary") for i in (1, 3, 4, 5, 6, 7))
    text = "\n".join([
        cards.scorecard_card(s1, s3, s7), cards.pd_card(s4, s6, s7), cards.lgd_ead_card(s5),
        cards.ecl_card(s6), cards.register(s1, s3, s4, s6, s7),
    ])  # fmt: skip
    assert not re.search(r"\bnan\b|\bNone\b|\binf\b", text)
    assert f"{100 * s6['coverage']:.2f}%" in text  # headline ECL coverage is carried through


def test_ok_treats_nan_and_none_as_missing():
    assert cards.ok(0.5) and not cards.ok(None) and not cards.ok(float("nan"))
