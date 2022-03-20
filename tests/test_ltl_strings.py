from unittest import TestCase

from autograph.lib.util.LTLStrings import X, F, G


class TestLTLStrings(TestCase):
    def test_x(self):
        self.assertEqual(X[3]("Test"), "X(X(X(Test)))")
        self.assertEqual(X[0]("Test"), "Test")
        self.assertEqual(X[1]("pie"), "X(pie)")

    def test_ltl_type(self):
        self.assertEqual(F[0:2]("ap"), "(ap)|(X((ap)|(X(ap))))")
        self.assertEqual(G[2:6:2]("ap"), "X(X((ap)&(X(X((ap)&(X(X(ap))))))))")
