import asyncio
import logging

import langchain
import shortuuid
import uvloop
from langchain.agents import AgentType, initialize_agent
from langchain.schema import SystemMessage
from langchain.callbacks import StdOutCallbackHandler
from langchain.chat_models import PromptLayerChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from pyrogram import filters
from pyrogram.client import Client
from pyrogram.methods.utilities.idle import idle
from pyrogram.types import BotCommand, Message

from backend import (
    add_player,
    get_win_prob,
    list_players,
    list_players_pretty,
    update_ratings,
    get_fair_match,
    _get_fair_match,
)

langchain.debug = True

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


def get_agent():
    run_uuid = shortuuid.uuid()[:10]

    chat = PromptLayerChatOpenAI(
        temperature=0,
        model="gpt-4",
        callbacks=[StdOutCallbackHandler()],
        pl_tags=[run_uuid],
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    tools: list[Tool] = [  # type: ignore
        add_player,
        get_win_prob,
        list_players,
        update_ratings,
        get_fair_match,
    ]

    for tool in tools:
        tool.handle_tool_error = True

    agent = initialize_agent(
        tools,
        chat,
        agent=AgentType.OPENAI_FUNCTIONS,
        memory=memory,
        agent_kwargs={
            "system_message": SystemMessage(
                content="You are an AI assistant that can help users calculate their ranking for badminton (or other) games, using TrueSkill. If a user does not exist, you should add them first."
            )
        },
    )
    return agent


async def main():
    app = Client("my_account")

    intro = """
    Hi there! I calculate and update rankings for players in badminton (or other) games, and use that information to generate fair matchups.

Just tell me the result of a match, e.g. 'Adam and Bob won Cindy and Dave', and I'll do the rest.

You can also ask me to calculate the probability of a team winning another team, list the rankings (/list also works), add a player, or suggest matchups, all in natural language.
        
For more information, see [here](https://github.com/extrange/rating-bot).
    """

    @app.on_message(filters.command("start") | filters.command("help"))
    async def handle_start(client, message: Message):
        await message.reply(intro, disable_web_page_preview=True)

    @app.on_message(filters.command("list"))
    async def handle_list(client, message: Message):
        await message.reply(f"```{list_players_pretty()}```")

    @app.on_message(filters.command("match"))
    async def handle_match(client, message: Message):
        await message.reply(_get_fair_match())

    # Respond to either private messages or @mentions in groups
    @app.on_message((filters.private & filters.text) | filters.mentioned)
    async def handle_message(client, message: Message):
        if message.mentioned and len(message.text.split()) == 1:
            # User didn't send anything
            await message.reply(intro, disable_web_page_preview=True)
            return

        reply = await message.reply("Thinking...")
        agent = get_agent()
        result = await agent.arun(
            input=f"{list_players({})}\nUser's request: {message.text}"
        )
        await reply.edit_text(result)

    await app.start()
    await app.set_bot_commands(
        [
            BotCommand("list", "Display ranking leaderboard"),
            BotCommand("help", "Show help"),
            BotCommand('match', "Calculate fair matchups")
        ]
    )
    await idle()
    await app.stop()


if __name__ == "__main__":
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    asyncio.run(main())
