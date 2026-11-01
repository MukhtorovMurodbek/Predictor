"""Find a pro team from what a person types: 'spirit', 'Team Liquid', 'NAVI'."""


def find_team(query: str, names: dict[int, tuple[str, str]], activity: dict[int, int]) -> int:
    """Team id for `query`. `names` maps team id -> (name, tag); `activity` maps the teams we
    may answer with to how many recent matches they played (busier teams win ties).
    Raises ValueError with a helpful message if nothing or several teams match."""
    q = query.strip().lower()
    if not q:
        raise ValueError("empty team name")

    def score(team_id):
        name, tag = (s.lower() for s in names.get(team_id, ("", "")))
        if q in (name, tag, name.removeprefix("team ").strip()):
            return 3
        if any(w.startswith(q) for w in name.split()) or name.startswith(q):
            return 2
        return 1 if q in name else 0

    scored = [(score(t), n, t) for t, n in activity.items()]
    best = max((s for s, _, _ in scored), default=0)
    if best == 0:
        raise ValueError(f"no active team matches {query.strip()!r}")
    cands = sorted([(n, t) for s, n, t in scored if s == best], reverse=True)
    if len(cands) == 1 or cands[0][0] >= 2 * cands[1][0]:
        return cands[0][1]
    options = ", ".join(names.get(t, (f"team {t}", ""))[0] for _, t in cands[:6])
    raise ValueError(f"{query.strip()!r} could be: {options}. Be more specific.")
