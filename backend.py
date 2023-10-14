import itertools
import math
import re
from pathlib import Path
from typing import Sequence, TypedDict, cast, Optional, Any, TypeVar

from langchain.tools import tool
from langchain.tools.base import ToolException
from prettytable import PrettyTable
from tinydb import Query, TinyDB
from tinydb.table import Document
from trueskill import Rating, TrueSkill

players = TinyDB(Path(__file__).parent / "players.json")
env = TrueSkill(backend="mpmath")


class Player(TypedDict):
    name: str
    mu: float
    sigma: float


def to_rating(player: Document) -> Rating:
    return Rating(mu=player["mu"], sigma=player["sigma"])


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

    return players.insert(
        {"name": name.title(), "mu": rating.mu, "sigma": rating.sigma}
    )


@tool
def list_players() -> str:
    """Print out a list of all the players in the database, with their names, ids and ratings."""

    return "All Players:\n\n" + "\n".join(
        [
            f'id={p.doc_id}, name={p["name"]}, rating={p["mu"]:.0f}'
            for p in players.all()
        ]
    )


def list_players_pretty() -> str:
    """Prettier table-formatted version of list_players, for human consumption"""
    tbl = PrettyTable()
    tbl.field_names = ["Name", "Rating"]
    tbl.align["Name"] = "l"
    tbl.align["Rating"] = "r"
    all_players = players.all()
    all_players.sort(key=lambda p: p["mu"], reverse=True)
    for p in all_players:
        tbl.add_row([p["name"], f'{p["mu"]:.0f}'])
    return tbl.get_string()
    ...


def validate_players(team: list[int]) -> list[Document]:
    """Ensures that all players exist, and the list is not empty.

    Returns a list of documents corresponding to the provided player IDs."""
    if not team:
        raise ToolException("A team cannot be empty")

    team_players = cast(list[Document], players.get(doc_ids=team))
    if unadded := set(team) - set([p.doc_id for p in team_players]):
        raise ToolException(
            f"Player IDs {list(unadded)} are not in the database, add them first."
        )

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

    if len(set(winners + losers)) != len(winners + losers):
        # Duplicates:
        raise ToolException("A player can't appear twice here.")

    rating_groups = [
        {p.doc_id: to_rating(p) for p in winner_players},
        {p.doc_id: to_rating(p) for p in loser_players},
    ]

    rated_rating_groups = env.rate(rating_groups, ranks=[0, 1])
    merged_rating_groups = rated_rating_groups[0] | rated_rating_groups[1]

    results = []

    for p_id in [*winners, *losers]:
        mu = merged_rating_groups[p_id].mu
        sigma = merged_rating_groups[p_id].sigma
        player = cast(Document, players.get(doc_id=p_id))
        results.append(f'{player["name"]}: {mu:.0f} ({mu - player["mu"]:+.0f})')
        players.update({"mu": mu, "sigma": sigma}, doc_ids=[p_id])

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
    team1_players = [to_rating(p) for p in validate_players(team1)]
    team2_players = [to_rating(p) for p in validate_players(team2)]
    return f"{_get_win_prob(team1_players, team2_players)*100: .1f}%"


def _get_win_prob(team1: Sequence[Rating], team2: Sequence[Rating]) -> float:
    delta_mu = sum(r.mu for r in team1) - sum(r.mu for r in team2)
    sum_sigma = sum(r.sigma**2 for r in itertools.chain(team1, team2))
    size = len(team1) + len(team2)
    denom = math.sqrt(size * (env.beta * env.beta) + sum_sigma)
    return float(env.cdf(delta_mu / denom))


@tool
def get_fair_match(team: Optional[list[int]] = None) -> str:
    """Calculate the fairest matchup, as defined by the highest probability of drawing.

    Args:
        team (Optional[Sequence[int]]): If provided, calculates the fairest opposing team.

    Returns:
        str: If `team` was provided, output the fairest opposing team matchup. Otherwise, the top 3 fairest matchups.
    """
    return _get_fair_match(team)


def _get_fair_match(team: Optional[list[int]] = None) -> str:
    all_players = players.all()
    if len(all_players) < 4:
        raise ToolException(
            "Insufficient players in database: at least 4 players is required"
        )
    all_players_ids = {p.doc_id for p in all_players}

    if team:
        team_players = validate_players(team)
        g_rating = [to_rating(p) for p in team_players]

        # Exclude these team players
        rest_ids = all_players_ids - set(team)
        rest = cast(list[Document], players.get(doc_ids=list(rest_ids)))

        # Generate doubles groups from these remaining players
        candidate_groups = get_groups(rest)

        fairness_team: list[tuple[list[Document], float]] = []

        # Calculate all Trueskill fairness
        for c in candidate_groups:
            candidate_ratings = [to_rating(p) for p in c]

            # Get rating
            quality = env.quality([g_rating, candidate_ratings])
            fairness_team.append((c, quality))

        # Sort ratings
        fairness_team.sort(key=lambda x: x[1], reverse=True)

        from pprint import pprint

        pprint(fairness_team)

        player_str = []

        for c in fairness_team[:3]:
            player_str.append(
                f"""{" and ".join([f"{p['name']} ({p['mu']:.0f})" for p in c[0]])} ({c[1]*100:.0f}%)"""
            )

        return (
            f"""Best opponents for {' and '.join([f"{p['name']} ({p['mu']:.0f})" for p in team_players])}, sorted by fairness:\n\n"""
            + "\n\n".join(player_str)
        )

    else:
        # Calculate best possible matchups for all players.

        # 1. Get all possible double matchups
        all_doubles_groups = get_groups(all_players)

        # [
        #   ([Doc1, Doc2], [Doc3, Doc4], 0.56), etc
        # ]
        fairness_all: list[tuple[list[Document], list[Document], float]] = []

        # 2. For each group, find all possible matchups
        for g in all_doubles_groups:
            # Exclude these players
            rest = set(all_players_ids) - set([p.doc_id for p in g])
            candidate_groups = get_groups(
                cast(list[Document], players.get(doc_ids=list(rest)))
            )

            g_rating = [to_rating(p) for p in g]

            # Calculate Trueskill fairness
            for c in candidate_groups:
                candidate_ratings = [to_rating(p) for p in c]

                # Get rating
                quality = env.quality([g_rating, candidate_ratings])
                fairness_all.append((g, c, quality))

        # Sort in fairness, descending
        fairness_all.sort(key=lambda x: x[2], reverse=True)

        player_str = []

        for c in fairness_all[:5]:
            # TODO Tidy this up
            player_str.append(
                f"""{' vs '.join([" and ".join([f"{p['name']} ({p['mu']:.0f})" for p in c[0]]), " and ".join([f"{p['name']} ({p['mu']:.0f})" for p in c[1]]) ])} ({c[2]*100:.0f}%)"""
            )

        return "Best 5 matches, sorted by fairness:\n\n" + "\n\n".join(player_str)


T = TypeVar("T")


def get_groups(items: list[T]) -> list[list[T]]:
    """Generate all possible 2-item pairings given a list of objects."""
    groups = []
    for idx, item in enumerate(items):
        remaining = items[idx + 1 :]
        for r in remaining:
            groups.append([item, r])
    return groups
