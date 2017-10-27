import pytest

import numpy as np
from opinions import Space, GeographicSpace, Activation


class TestSpace():
    def test_interaction_order(self):

        activation = lambda x,y: x

        a, b = Space.interact("first", "last", activation)
        assert a == "first"
        assert b == "last"


    def test_pairs_all_agents(self):
        # assert all agents are accounted for
        # assert all agents accounted for just once.

        agents = list(range(100))
        pairs = GeographicSpace.pairs(agents)

        pair_agents = [p for pair in pairs for p in pair]
        assert set(pair_agents) == set(range(100))
        assert len(pair_agents) == 100

    def test_pairs_are_unique_pairs(self):
        # assert each pair has 2 agents
        # assert each pair has 2 unique agents

        agents = list(range(100))
        pairs = GeographicSpace.pairs(agents)

        assert all([pair[0] is not pair[1] for pair in pairs])


    def test_pairs_throws_odd_agents(self):
        # assert pair only runs for even number of agents

        agents = list(range(99))
        with pytest.raises(Exception):
            pairs = GeographicSpace.pairs(agents)


class TestActivation():
    def test_callable(self):
        # the activation type should be callable, behave like a function.
        # why not just use a function?
        activation = Activation()

        activation(0,1)

    def test_towards_first(self):
        # the activation should result in a diff w.r.t the first value.
        # This activation should have activation(a,b) == - activation(b,a).
        # this probably wont hold for future activation functions.


        activation = Activation()

        front = activation(0,1)
        back = activation(1,0)
        assert np.sign(front) == 1
        assert np.sign(back) == -1
