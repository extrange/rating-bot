import itertools
import math
import re
from typing import Sequence, TypedDict, cast

from langchain.tools import tool
from langchain.tools.base import ToolException
from prettytable import PrettyTable
from tinydb import Query, TinyDB
from tinydb.table import Document
from trueskill import Rating, TrueSkill

players = TinyDB('players.json')
env = TrueSkill(backend='mpmath')


class Player(TypedDict):
    name: str
    mu: float
    sigma: float


@tool
def add_player(name: str) -> int:
    """Add a player to the database.

    Args:
        name (str): Name of the player.

    Returns:
        int: id of the player.
    """
    if query := players.search(Query().name.matches(name, flags=re.IGNORECASE)):
        # attempt to add an existing player, just return the ID
        return query[0].doc_id
    rating = env.create_rating()

    return players.insert({'name': name.title(), 'mu': rating.mu, 'sigma': rating.sigma})


@tool
def list_players() -> str:
    """Print out a list of all the players in the database, with their names, ids and ratings."""

    return "All Players:\n\n" + "\n".join([f'id={p.doc_id}, name={p["name"]}, rating={p["mu"]:.0f}' for p in players.all()])


def list_players_pretty() -> str:
    """Prettier table-formatted version of list_players, for human consumption"""
    tbl = PrettyTable()
    tbl.field_names = ["Name", "Rating"]
    tbl.align["Name"] = 'l'
    tbl.align["Rating"] = 'r'
    all_players = players.all()
    all_players.sort(key=lambda p: p['mu'], reverse=True)
    for p in all_players:
        tbl.add_row([p['name'], f'{p["mu"]:.0f}'])
    return tbl.get_string()
    ...


def validate_players(team: list[int]) -> list[Document]:
    """Ensures that all players exist, and the list is not empty.

    Returns a list of documents corresponding to the provided player IDs."""
    if not team:
        raise ToolException('A team cannot be empty')

    team_players = cast(list[Document], players.get(doc_ids=team))
    if (unadded := set(team) - set([p.doc_id for p in team_players])):
        raise ToolException(
            f'Player IDs {list(unadded)} are not in the database, add them first.')

    return team_players


@tool
def update_ratings(winners: list[int], losers: list[int]) -> str:
    """Update player ratings given a match result.

    Args:
        winners (list[int]): Winning team. Given as a list of player IDs, e.g. [2,19].
        losers (list[int]): Losing team. Given as a list of player IDs.

    Returns:
        str: Updated ratings of each player.
    """
    winner_players = validate_players(winners)
    loser_players = validate_players(losers)

    rating_groups = [{p.doc_id: Rating(p['mu'], p['sigma'])for p in winner_players}, {
        p.doc_id: Rating(p['mu'], p['sigma']) for p in loser_players}]

    rated_rating_groups = env.rate(rating_groups, ranks=[0, 1])
    merged_rating_groups = rated_rating_groups[0] | rated_rating_groups[1]

    results = []

    for p_id in [*winners, *losers]:
        mu = merged_rating_groups[p_id].mu
        sigma = merged_rating_groups[p_id].sigma
        player = cast(Document, players.get(doc_id=p_id))
        results.append(
            f'{player["name"]}: {mu:.0f} ({mu - player["mu"]:+.0f})')
        players.update({'mu': mu, 'sigma': sigma}, doc_ids=[p_id])

    return "New Ratings:\n\n" + "\n".join(results)


@tool
def get_win_prob(team1: list[int], team2: list[int]) -> str:
    """Calculate the probability that team1 will win team2.

    Args:
        team1 (list[int]): list of player IDs
        team2 (list[int]): list of player IDs

    Returns:
        str: Probabiilty that team 1 will win team 2, formatted as a percentage.
    """
    # Fetch players from db
    team1_players = [Rating(p['mu'], p['sigma'])
                     for p in validate_players(team1)]
    team2_players = [Rating(p['mu'], p['sigma'])
                     for p in validate_players(team2)]
    return f'{_get_win_prob(team1_players, team2_players)*100: .1f}%'


def _get_win_prob(team1: Sequence[Rating], team2: Sequence[Rating]) -> float:

    delta_mu = sum(r.mu for r in team1) - sum(r.mu for r in team2)
    sum_sigma = sum(r.sigma ** 2 for r in itertools.chain(team1, team2))
    size = len(team1) + len(team2)
    denom = math.sqrt(size * (env.beta * env.beta) + sum_sigma)
    return float(env.cdf(delta_mu / denom))
