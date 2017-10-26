import random

from mesa import Agent, Model
from mesa.time import BaseScheduler

"""

Create a model

Model

"""

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
    @staticmethod
    def pairs(agents):
        """ Take list of agents, return list of pairs of agents """
        # TODO: Eventual use information about location in opinion space to build pairs.

        assert len(agents) % 2 == 0
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
        new_a, new_b = GeographicSpace.interact(first.location,
                                                second.location,
                                                Activation())

        first.location += new_a
        second.location += new_b

        assert 0 <= first.location <= 1, f"First.location is {first.location}"
        assert 0 <= second.location <= 1, f"Second.location is {second.location}"

    def update_pair_opinions(self, pair):
        activation = lambda x, y: (x + y) / 2

        first, second = pair
        new_a, new_b = OpinionSpace.interact(first.opinion,
                                             second.opinion,
                                             Activation())

        first.opinion += new_a
        second.opinion += new_b

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

    def __init__(self, unique_id, model):
        self.opinion = random.random()  # 1d space for now # TODO: make n-dimensional
        self.location = random.random() # 1d space for now
        super().__init__(unique_id, model)

    def __repr__(self):
        return f"<Agent(id={self.unique_id}, op={self.opinion}, loc={self.location})>"


class OpinionModel(Model):
    def __init__(self):
        self.steps = 1
        self.schedule = Scheduler(self)

        for i in range(10):
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


if __name__ == "__main__":
    model = OpinionModel()

    print("Initial State:")
    print(model)

    for i in range(100):
        model.step()

    print("Terminal State:")
    print(model)
