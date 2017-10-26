import random

from mesa import Agent, Model
from mesa.time import BaseScheduler

"""

Create a model

Model

"""

class OpinionSpace():
    @staticmethod
    def interact(agent_a, agent_b, activation):
        """ Update each agents opinions according to the activation function """
        # Can we assert anything about the signature of activation?

        pre_change = agent_a.opinion
        agent_a.opinion = activation(agent_a.opinion, agent_b.opinion)
        agent_b.opinion = activation(agent_b.opinion, pre_change)

class MatchMaker():
    @staticmethod
    def pairs(agents):
        """ Take list of agents, return list or pairs of agents """
        # Eventual use information about location in opinion space
        # to build pairs.

        assert len(agents) % 2 == 0
        half = int(len(agents) / 2)
        random.shuffle(agents)

        pairs = zip(agents[:half], agents[half:])
        return pairs


class Scheduler(BaseScheduler):
    def step(self):
        pairs = MatchMaker.pairs(self.agents[:])
        for pair in pairs:
            activation = lambda x, y: (x + y) / 2
            OpinionSpace.interact(pair[0], pair[1], activation)

        for agent in self.agents[:]:
            agent.step()

        self.steps += 1
        self.time += 1


class OpinionatedAgent(Agent):
    """ Each agent should have an embedding in opinion space and in geographic space """

    def __init__(self, unique_id, model):
        self.opinion = random.random()
        super().__init__(unique_id, model)

    def __repr__(self):
        return f"<Agent(unique_id={self.unique_id}, opinion={self.opinion})>"


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

    def state(self):
        agents = [str(agent) for agent in self.schedule.agents]
        agents = "\n".join(agents)

        return agents


if __name__ == "__main__":
    model = OpinionModel()

    print("Initial State:")
    print(model.state())

    for i in range(10):
        model.step()

    print("Terminal State:")
    print(model.state())
