import pytest

from opinions import Space


class TestSpace():
    def test_interaction_order(self):

        activation = lambda x,y: x

        a, b = Space.interact("first", "last", activation)
        assert a == "first"
        assert b == "last"
