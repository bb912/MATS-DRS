from unittest import TestCase

import pypeg2 as p

import autograph.lib.hoa as h
from autograph.lib.automata import AutomatonSet


class TestHOA(TestCase):
    def assert_prop(self, text, aps, result, alias=None):
        if alias is None:
            alias = {}

        alias = ((name, p.parse(a, h.LabelExpr)) for name, a in alias.items())

        label = p.parse(text, h.LabelExpr)
        obj = {
            "alias": dict(alias)
        }

        self.assertEqual(result, label.evaluate(aps, obj, ), text)

    def test_basic(self):
        self.assert_prop("t", [], True)
        self.assert_prop("f", [], False)
        self.assert_prop("0", [True], True)
        self.assert_prop("1", [True, False], False)
        self.assert_prop("!0", [True], False)
        self.assert_prop("t & t", [], True)
        self.assert_prop("t&f", [], False)
        self.assert_prop("!f&f", [], False)
        self.assert_prop("(!f)&f", [], False)
        self.assert_prop("!0&0", [False], False)
        self.assert_prop("!(0&0)", [False], True)
        self.assert_prop("!(f&f)", [], True)
        self.assert_prop("t|t", [], True)
        self.assert_prop("f|t", [], True)
        self.assert_prop("f | f", [], False)

    def test_multi(self):
        s = "0&!1&2 | 3"
        self.assert_prop(s, [False] * 4, False)
        self.assert_prop(s, [True, False, True, False], True)
        self.assert_prop(s, [False, False, False, True], True)
        self.assert_prop(s, [True, True, True, False], False)

    def test_alias(self):
        s = "!@a&@b"
        alias = {
            "@a": "0",
            "@b": "1|@c",
            "@c": "2"
        }
        self.assert_prop(s, [False, True, True], True, alias)
        self.assert_prop(s, [True] * 3, False, alias)
        self.assert_prop(s, [False] * 3, False, alias)
        self.assert_prop(s, [False, False, True], True, alias)

    def test_automaton(self):
        aut = """
        HOA: v1
        name: "X!p1 | G(!p0 | Fp2)"
        States: 5
        Start: 0
        AP: 3 "p0" "p2" "p1"
        acc-name: Buchi
        Acceptance: 1 Inf(0)
        properties: trans-labels explicit-labels state-acc
        --BODY--
        State: 0
        [t] 1
        [!0 | 1] 2
        [0&!1] 3
        State: 1
        [!2] 4
        State: 2 {0}
        [!0 | 1] 2
        [0&!1] 3
        State: 3
        [1] 2
        [!1] 3
        State: 4 {0}
        [t] 4
        --END--
        """

        sset = AutomatonSet.from_hoa(h.parse(aut))
        self.assertEqual(sset.states, {0})
        self.assertEqual(sset.acceptance(), set())

        sset12 = sset.transition([False, True, True])
        self.assertEqual(sset.states, {0})  # Not modified
        self.assertEqual(sset12.states, {1, 2})
        self.assertEqual(sset12.acceptance(), {0})
        self.assertNotEqual(sset, sset12)

        sset13 = sset.transition([True, False, True])
        self.assertEqual(sset13.states, {1, 3})
        self.assertEqual(sset13.acceptance(), set())

        sset24 = sset12.transition([False, True, False])
        self.assertEqual(sset24.states, {2, 4})

        sset34 = sset24.transition([True, False, True])
        self.assertEqual(sset34, sset13.transition([True, False, False]))

        sset3 = sset12.transition([True, False, True])
        self.assertEqual(sset3.states, {3})
        self.assertEqual(sset3.acceptance(), set())

        sset2 = sset3.transition([True, True, True])
        self.assertEqual(sset2.states, {2})
        self.assertEqual(sset2.acceptance(), {0})

    def test_ltlf(self):
        sset = AutomatonSet.from_ltlf("(X!p1) | G((!p0) | (F(p2)))", ["p0", "p2", "p1"])

        def accepts(seq):
            aut = sset
            for selement in seq:
                aut = sset.transition(selement)

            return 0 in aut.acceptance()

        self.assertTrue(accepts([[True, True, True],
                                 [True, True, False]]))

        self.assertTrue(accepts([[False, False, False],
                                 [False, False, True],
                                 [False, False, True]]))

        self.assertFalse(accepts([[True, True, True],
                                  [True, False, True]]))
