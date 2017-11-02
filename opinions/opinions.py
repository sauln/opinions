import random
from itertools import cycle
import logging

import numpy as np
from bokeh.plotting import figure, output_file, show
from bokeh.palettes import Spectral11
from bokeh.io import output_notebook

from mesa import Agent, Model
from mesa.time import BaseScheduler

"""

Draft for opinion dynamics model.

"""

logger = logging.Logger("opin")

NUM_AGENTS = 50
NUM_STEPS = 105

def make_adjacency_matrix():
    # Make a random adjacency matrix
    random_matrix = np.random.rand(NUM_AGENTS,NAGENTS)
    symmetric_matrix = np.dot(random_matrix, random_matrix.T)
    np.fill_diagonal(symmetric_matrix, 0)
    normalized = pre.normalize(symmetric_matrix)

    fuzzy_adjacency_matrix = normalized
    return fuzzy_adjacency_matrix


class Space():
    """ Base class for Opinion and Geography space. The dynamics of each are identical except activation function. """

    @staticmethod
    def interact(value_a, value_b, activation):
        """ Update each agents' state according to the activation function """
        # Can we assert anything about the signature of activation?

        pre_value = value_a
        diff_a = activation(value_a, value_b)
        diff_b = activation(value_b, pre_value)

        return diff_a, diff_b


class OpinionSpace(Space):
    """  """
    pass


class GeographicSpace(Space):

    # Build a 1-skeleton based on the distance from each person.

    # Potential algorithms

    # greedy maximal-diametric reduction
    #   - find the longest edge in a complete graph, name e.
    #   - add the closest edge to a vertex in e to solution.
    #   - repeat.

    # simulated annealing esque algo
    #   - choose a random solution. compute result
    #   - choose a random subset and choose a new random solution.
    #       - if the total solution is cheaper, take it as new solution. otherwise discard.

    #

    @staticmethod
    def pairs(agents):
        """ Take list of agents, return list of pairs of agents """
        # TODO: Eventual use information about location in opinion space to build pairs.

        if not len(agents) % 2 == 0:
            raise Exception("Require even number of agents")

        half = int(len(agents) / 2)
        random.shuffle(agents)

        pairs = zip(agents[:half], agents[half:])
        return pairs


class Activation():
    """ This class will be overwritten by various activation functions """
        # TODO: information about the pairs will be used to choose which activation

    def body(self, first, second):
        rate = 0.1
        center = (first + second) / 2
        direction = (center - first) * rate
        return direction

    def __call__(self, first, second):
        return self.body(first, second)


class Scheduler(BaseScheduler):
    def update_pair_geography(self, pair):
        # Should geography and opinion behave exactly the same?
        # Whats the best way to extract this function? get_attr is too slow, right?

        first, second = pair
        diff_a, diff_b = GeographicSpace.interact(first.location,
                                                second.location,
                                                Activation())

        first.update_location(diff_a)
        second.update_location(diff_b)

        assert 0 <= first.location <= 1, f"First.location is {first.location}"
        assert 0 <= second.location <= 1, f"Second.location is {second.location}"

    def update_pair_opinions(self, pair):

        first, second = pair
        new_a, new_b = OpinionSpace.interact(first.opinion,
                                             second.opinion,
                                             Activation())

        first.update_opinion(new_a)
        second.update_opinion(new_b)

        # Assert this never happens or clip at edges?
        assert 0 <= first.opinion <= 1, f"First.opinion is {first.opinion}"
        assert 0 <= second.opinion <= 1, f"Second.opinion is {second.opinion}"

    def step(self):
        pairs = GeographicSpace.pairs(self.agents[:])
        for pair in pairs:
            self.update_pair_opinions(pair)
            self.update_pair_geography(pair)

        self.steps += 1
        self.time += 1


class OpinionatedAgent(Agent):
    """ Each agent should have an embedding in opinion space and in geographic space """

    def __init__(self, unique_id, model, momentum_gain=0.1):
        self.opinion_history = [] #np.zeros(NUM_STEPS)
        self.opinion = random.random()  # 1d space for now # TODO: make n-dimensional
        self.location = random.random() # 1d space for now
        self.momentum = 0
        self.momentum_gain = momentum_gain
        self.steps = 0
        super().__init__(unique_id, model)

    def update_opinion(self, diff):
        # Store current history
        self.opinion_history.append(self.opinion)

        # Update history
        self.momentum = diff + self.momentum_gain * self.momentum
        logger.debug(f"OpinAgent: {self.unique_id}, opinion: {self.opinion}, newdiff: {diff}, diff: {self.momentum}")
        self.opinion += self.momentum

    def update_location(self, diff):
        self.location += diff

    def __repr__(self):
        return f"<OpinAgent(id={self.unique_id}, op={self.opinion}, loc={self.location})>"


class OpinionModel(Model):
    def __init__(self):
        self.steps = 1
        self.schedule = Scheduler(self)

        for i in range(NUM_AGENTS):
            ag = OpinionatedAgent(i, self)
            self.schedule.add(ag)

    def step(self):
        self.schedule.step()
        self.steps += 1

    def __repr__ (self):
        """ """
        agents = [str(agent) for agent in self.schedule.agents]
        agents = "\n".join(agents)
        return agents


class Visualize():
    @staticmethod
    def line_plot(agents, notebook=1, filename="results/agents.html"):
        if notebook:
            output_notebook()
        else:
            output_file(filename)


        p = figure(plot_width=800, plot_height=250)

        for agent, color in zip(agents, cycle(Spectral11)):
            p.line(x=np.arange(len(agent.opinion_history)),
                   y=agent.opinion_history,
                   color=color)

        show(p)


if __name__ == "__main__":
    model = OpinionModel()

    logger.debug("Initial State:")
    logger.debug(model)

    for i in range(NUM_STEPS):
        model.step()

    logger.debug("Terminal State:")
    logger.debug(model)

    Visualize.line_plot(model.schedule.agents)
