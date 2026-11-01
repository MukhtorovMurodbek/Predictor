const MIN_MATCHES_PER_OPPONENT = 20;

function heroId(value) {
  const id = Number(value && typeof value === 'object' ? value.id : value);
  return Number.isInteger(id) && id > 0 ? id : null;
}

async function fetchJson(url, signal) {
  const response = await fetch(url, { signal, headers: { Accept: 'application/json' } });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `Request failed (HTTP ${response.status}).`);
  return body;
}

export async function loadHeroes({ signal } = {}) {
  const result = await fetchJson('/api/heroes', signal);
  if (!Array.isArray(result.heroes)) throw new Error('Invalid hero catalogue.');
  return result.heroes
    .filter(row => heroId(row.id) && typeof row.name === 'string')
    .map(row => ({ id: Number(row.id), name: row.name, portrait: row.portrait || '' }))
    .sort((a, b) => a.name.localeCompare(b.name));
}

/** Load a filtered aggregate of the locally collected public-match dataset. */
export async function fetchSample({ rankMin = null, rankMax = null, patch = null, signal } = {}) {
  const url = new URL('/api/sample', window.location.origin);
  if (rankMin != null) url.searchParams.set('rank_min', String(rankMin));
  if (rankMax != null) url.searchParams.set('rank_max', String(rankMax));
  if (patch) url.searchParams.set('patch', patch);
  const sample = await fetchJson(url, signal);
  if (!Array.isArray(sample.pairwise)) throw new Error('Invalid matchup data.');
  sample.pairwise = new Map(sample.pairwise.map(pair => [
    `${pair.hero_id}:${pair.enemy_id}`, { wins: pair.wins, matches: pair.matches }
  ]));
  sample.warnings ||= [];
  return sample;
}

/** Compare observed opponents only; an unseen matchup is never set to 50%.
 * Sorting favors adequate evidence and matchup coverage before score. */
export function rankCandidates({ heroes = [], enemies = [], allies = [], bans = [], sample } = {}) {
  const enemyIds = [...new Set(enemies.map(heroId).filter(Boolean))];
  if (!enemyIds.length || !(sample?.pairwise instanceof Map)) return [];
  const unavailable = new Set([...enemyIds, ...allies.map(heroId), ...bans.map(heroId)].filter(Boolean));
  return heroes.filter(hero => !unavailable.has(heroId(hero))).map(hero => {
    const id = heroId(hero);
    const perEnemy = enemyIds.map(enemyId => {
      const record = sample.pairwise.get(`${id}:${enemyId}`) || { wins: 0, matches: 0 };
      return { enemyId, wins: record.wins, matches: record.matches, winRate: record.matches ? record.wins / record.matches * 100 : null };
    });
    const observed = perEnemy.filter(pair => pair.matches > 0);
    const totalMatches = observed.reduce((sum, pair) => sum + pair.matches, 0);
    const score = observed.length ? observed.reduce((sum, pair) => sum + pair.winRate, 0) / observed.length : null;
    const coverage = observed.length;
    const reliable = coverage === enemyIds.length && perEnemy.every(pair => pair.matches >= MIN_MATCHES_PER_OPPONENT);
    return { hero, score, perEnemy, coverage, enemyCount: enemyIds.length, totalMatches, reliable, limitedConfidence: !reliable };
  }).filter(candidate => candidate.coverage > 0).sort((a, b) => {
    if (a.reliable !== b.reliable) return a.reliable ? -1 : 1;
    if (a.coverage !== b.coverage) return b.coverage - a.coverage;
    // Tiny datasets should not top the list solely due to one lucky win.
    if (!a.reliable && a.totalMatches !== b.totalMatches) return b.totalMatches - a.totalMatches;
    return (b.score ?? -Infinity) - (a.score ?? -Infinity) || b.totalMatches - a.totalMatches || a.hero.name.localeCompare(b.hero.name);
  });
}
