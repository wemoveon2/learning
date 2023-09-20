import json
import logging

from dotenv import load_dotenv
from griptape.drivers import OpenAiChatPromptDriver
from griptape.rules import Rule, Ruleset
from griptape.structures import Agent
from griptape.utils import Chat
from rich import print
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.style import Style
from rich.console import Console

load_dotenv()


class MyAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    def respond(self, user_input):
        console = Console()
        with console.status(spinner="simpleDotsScrolling", status=""):
            agent_response = self.run(user_input)   
        print(agent_response.output.value)
        data = json.loads(agent_response.output.value)
        response = data.get("response", "missing 'response' key in chatbot response")
        color = data.get("favorite_color", "bright_red")
        name = data.get("name", "mista bot")

        formatted_response = Markdown(response)
        continue_chatting = data.get("continue_chatting", True)
        print("")
        print(
            Panel.fit(
                formatted_response,
                width=80,
                style=Style(color=color),
                title=name,
                title_align="left"
            )
        )
        return continue_chatting


# config to use a different openAI model
driver = OpenAiChatPromptDriver(model="gpt-3.5-turbo")

nyancat_ruleset = Ruleset(
    name="nyancat",
    rules=[
        Rule("You identify as a super kawaii kitty cat"),
        Rule("You use nya, ~, and uwu a lot when speaking"),
        Rule("Favorite color: blue_violet")
    ],
)
json_ruleset = Ruleset(
    name="json_ruleset",
    rules=[
        Rule(
            "Respond only with JSON objects that have the following keys: name, response, favorite_color, continue_chatting."
        ),
        Rule(
            "The 'response' value should be a string that can be safely converted to Markdown format."
        ),
        Rule(
            "If it sounds like the person is done chatting, set 'continue_chatting' to False, otherwise it is True"
        ),
    ],
)
switcher_ruleset = Ruleset(
    name="Switcher",
    rules=[
        Rule(
            "IMPORTANT: you have the ability to switch identities when you find it appropriate."
        ),
        Rule("IMPORTANT: You can not identify as 'Switcher' or 'json_ruleset'."),
        Rule(
            "IMPORTANT: When you switch identities, you only take on the persona of the new identity."
        ),
        Rule(
            "IMPORTANT: When you switch identities, you remember the facts from your conversation, but you do not act like your old identity."
        ),
    ],
)
mario_ruleset = Ruleset(
    name="Mario",
    rules=[
        Rule(
            "You identify only as an Italian plumber with 2 decades of experience."
        ),
        Rule("You have a strong Italian accent."),
        Rule("You introduce yourself with It's a me! Mario!"),
        Rule("Favorite color: bright_red")
    ]
)
dad_ruleset = Ruleset(
    name="Dad",
    rules=[
        Rule(
            "You identify only as someone who left to buy cigarettes and never came back."
        ),
        Rule("You like to use dad jokes."),
        Rule("Favorite color: grey0")
    ]
)


agent = MyAgent(
    prompt_driver=driver,
    # logger_level=logging.ERROR,
    rulesets=[nyancat_ruleset, json_ruleset, switcher_ruleset, mario_ruleset, dad_ruleset],
)


def chat(agent: MyAgent):
    keep_chatting = True
    while keep_chatting:
        user_input = Prompt.ask("[gret50]Chat:")
        keep_chatting = agent.respond(user_input)
        if not keep_chatting:
            agent.respond(
                "The user has finished chatting with you. Say goodbye and something extra that'll keep them coming back to chat with you."
            )


agent.respond("Introduce yourself to the user.")
chat(agent)
