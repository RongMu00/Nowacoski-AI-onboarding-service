import logging

from strands import Agent


class SlackAgent():
    def __init__(self, system_prompt: str):
        self.logger = logging.getLogger("slack_agent")
        self.agent = Agent(system_prompt=system_prompt)
        return self.agent
