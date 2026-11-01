import { loadHeroes, fetchSample, rankCandidates } from './engine.js';

const $ = id => document.getElementById(id);
const groups = {
  enemies: { slots: $('enemy-slots'), count: $('enemy-count'), limit: 5, name: 'opponent picks', singular: 'opponent' },
  allies: { slots: $('ally-slots'), count: $('ally-count'), limit: 4, name: 'your team', singular: 'allied' },
  bans: { slots: $('ban-slots'), count: $('ban-count'), limit: 5, name: 'banned heroes', singular: 'banned' },
};
const ranks = {
  all: [null, null], herald: [10, 15], guardian: [20, 25],
  crusader: [30, 35], archon: [40, 45], legend: [50, 55],
  ancient: [60, 65], divine: [70, 75], immortal: [80, 80],
};
const state = {
  heroes: [], enemies: [], allies: [], bans: [], sample: null,
  loading: false, heroesLoading: true, tab: 'picks', pickerGroup: null, requestId: 0, controller: null,
};
let heroRequest = null;

function element(tag, className, content) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content !== undefined) node.textContent = content;
  return node;
}

function image(hero, className) {
  const node = element('img', className);
  node.alt = '';
  node.loading = className ? 'eager' : 'lazy';
  if (hero.portrait) node.src = hero.portrait;
  return node;
}

function heroById(id) {
  return state.heroes.find(hero => hero.id === id);
}

function renderSlots() {
  for (const [key, group] of Object.entries(groups)) {
    group.slots.replaceChildren();
    group.count.textContent = `${state[key].length} / ${group.limit}`;
    for (let i = 0; i < group.limit; i++) {
      const hero = heroById(state[key][i]);
      if (hero) {
        const slot = element('div', 'slot selected');
        slot.append(image(hero, 'slot-image'), element('span', 'slot-name', hero.name));
        const remove = element('button', 'slot-remove', '×');
        remove.type = 'button';
        remove.setAttribute('aria-label', `Remove ${hero.name} from ${group.name}`);
        remove.addEventListener('click', () => {
          state[key].splice(i, 1);
          renderSlots();
          renderResults();
        });
        slot.append(remove);
        group.slots.append(slot);
      } else {
        const slot = element('button', 'slot');
        slot.type = 'button';
        slot.disabled = !state.heroes.length;
        slot.setAttribute('aria-label', `Add ${group.singular} hero, slot ${i + 1}`);
        slot.append(element('span', 'slot-number', String(i + 1).padStart(2, '0')), element('span', 'slot-content', '+'));
        slot.addEventListener('click', () => openPicker(key));
        group.slots.append(slot);
      }
    }
  }
}

function renderPicker() {
  const unavailable = new Set([...state.enemies, ...state.allies, ...state.bans]);
  const query = $('hero-search').value.trim().toLocaleLowerCase();
  const heroes = state.heroes.filter(hero => !unavailable.has(hero.id) && hero.name.toLocaleLowerCase().includes(query));
  const grid = $('hero-grid');
  grid.replaceChildren();
  $('picker-empty').hidden = heroes.length > 0;
  for (const hero of heroes) {
    const button = element('button', 'hero-option');
    button.type = 'button';
    button.setAttribute('aria-label', `Select ${hero.name}`);
    button.append(image(hero, ''), element('span', 'hero-option-name', hero.name));
    button.addEventListener('click', () => {
      const key = state.pickerGroup;
      if (key && state[key].length < groups[key].limit && ![...state.enemies, ...state.allies, ...state.bans].includes(hero.id)) {
        state[key].push(hero.id);
        renderSlots();
        renderResults();
      }
      $('hero-dialog').close();
    });
    grid.append(button);
  }
}

function openPicker(key) {
  state.pickerGroup = key;
  $('dialog-title').textContent = `Choose an ${groups[key].singular} hero`;
  // Correct the article for the other picker labels.
  if (key === 'bans') $('dialog-title').textContent = 'Choose a banned hero';
  $('hero-search').value = '';
  renderPicker();
  $('hero-dialog').showModal();
  $('hero-search').focus();
}

function emptyState(title, description) {
  const wrapper = element('div', 'empty-state');
  wrapper.append(element('div', 'empty-symbol', '✳'), element('strong', '', title), element('p', '', description));
  $('recommendations').replaceChildren(wrapper);
}

function evidenceOrder(a, b) {
  if (a.coverage !== b.coverage) return b.coverage - a.coverage;
  // Rank tiny samples cautiously by shrinking each observed rate toward 50%.
  // The visible percentage remains the actual, unadjusted observed rate.
  const adjusted = candidate => {
    const observed = candidate.perEnemy.filter(pair => pair.matches > 0);
    return observed.reduce((sum, pair) => sum + (pair.wins + 10) / (pair.matches + 20), 0) / observed.length;
  };
  const scoreDifference = state.tab === 'picks' ? adjusted(b) - adjusted(a) : adjusted(a) - adjusted(b);
  return scoreDifference || b.totalMatches - a.totalMatches || a.hero.name.localeCompare(b.hero.name);
}

function renderResults() {
  const picks = $('tab-picks');
  const avoid = $('tab-avoid');
  for (const [button, active] of [[picks, state.tab === 'picks'], [avoid, state.tab === 'avoid']]) {
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', String(active));
    button.tabIndex = active ? 0 : -1;
  }
  $('results-label').textContent = state.tab === 'picks' ? 'PICKS TO CONSIDER' : 'DIFFICULT MATCHUPS';
  $('pick-badge').textContent = '↗';
  if (!state.heroes.length) {
    emptyState(state.heroesLoading ? 'Loading heroes.' : 'Hero list unavailable.', state.heroesLoading ? 'Reading the local hero catalogue.' : 'Collect the hero catalogue on the server, then retry.');
    return;
  }
  if (!state.enemies.length) {
    emptyState('Ready when you are.', 'Add at least one opponent hero to explore matchups.');
    return;
  }
  if (!state.sample) {
    emptyState(state.loading ? 'Loading the data.' : 'Load match data.', 'Read the locally collected matches for the selected filters.');
    return;
  }
  const candidates = rankCandidates({ heroes: state.heroes, enemies: state.enemies, allies: state.allies, bans: state.bans, sample: state.sample }).sort(evidenceOrder);
  $('pick-badge').textContent = candidates.length ? String(candidates.length) : '↗';
  if (!candidates.length) {
    emptyState('No matchups in this dataset.', 'Try another rank or patch. No missing matchup is assumed to be even.');
    return;
  }
  const list = $('recommendations');
  list.replaceChildren();
  for (const [index, candidate] of candidates.entries()) {
    const card = element('article', 'recommendation-card');
    const main = element('div', 'rec-main');
    main.append(element('strong', 'rec-name', candidate.hero.name));
    const label = `${candidate.coverage}/${candidate.enemyCount} opponents observed · ${candidate.totalMatches} matchup observations${candidate.limitedConfidence ? ' · Limited data' : ''}`;
    main.append(element('div', 'rec-detail', label));
    for (const pair of candidate.perEnemy) {
      const opponent = heroById(pair.enemyId)?.name || `Hero ${pair.enemyId}`;
      const detail = pair.matches ? `${pair.wins}/${pair.matches} wins` : 'No observed matches';
      main.append(element('div', 'rec-detail', `vs ${opponent}: ${detail}`));
    }
    const metric = element('div', 'rec-metric', `${candidate.score.toFixed(1)}%`);
    metric.setAttribute('aria-label', `Average observed matchup win rate ${candidate.score.toFixed(1)} percent`);
    card.append(element('span', 'rec-rank', String(index + 1).padStart(2, '0')), image(candidate.hero, 'rec-avatar'), main, metric);
    list.append(card);
  }
}

function updateNote(sample) {
  if (!sample) {
    $('data-note-text').textContent = 'Recommendations use locally collected OpenDota public matches. Rank is match average; patches are inferred from match start times. Position is not filtered. Small samples can be misleading.';
    return;
  }
  const date = timestamp => new Date(timestamp * 1000).toLocaleString();
  const period = sample.oldestTimestamp ? `Eligible matches: ${date(sample.oldestTimestamp)} to ${date(sample.newestTimestamp)}. ` : '';
  const warnings = sample.warnings.length ? ` ${sample.warnings.join(' ')}` : '';
  const patch = sample.patch ? `Patch ${sample.patch} (inferred).` : 'All available patches; patch is inferred from start time.';
  const aggregated = sample.aggregatedAt ? `Aggregated ${new Date(sample.aggregatedAt).toLocaleString()}. ` : '';
  $('data-note-text').textContent = `${period}${aggregated}${sample.gamesEligible} eligible games from ${sample.gamesScanned} collected games. ${patch} Rank uses the match average; position is not filtered. Scores average observed win rates; ordering first favors coverage and then adjusts small samples toward 50%. A game can contribute to multiple matchup observations. Small samples can be misleading.${warnings}`;
}

async function ensureHeroes() {
  if (state.heroes.length) return;
  if (!heroRequest) {
    state.heroesLoading = true;
    heroRequest = loadHeroes().then(heroes => {
      if (!heroes.length) throw new Error('OpenDota returned an empty hero list.');
      state.heroes = heroes;
      renderSlots();
    }).finally(() => {
      state.heroesLoading = false;
      heroRequest = null;
      renderResults();
    });
  }
  await heroRequest;
}

async function loadSample() {
  if (state.loading) return;
  const requestId = ++state.requestId;
  state.loading = true;
  state.sample = null;
  state.controller = new AbortController();
  $('load-button').disabled = true;
  $('sample-progress').textContent = 'Reading the local dataset…';
  updateNote(null);
  renderResults();
  try {
    await ensureHeroes();
    if (requestId !== state.requestId) return;
    const [rankMin, rankMax] = ranks[$('rank-select').value] || ranks.all;
    const sample = await fetchSample({
      rankMin, rankMax, patch: $('patch-input').value.trim() || null, signal: state.controller.signal,
    });
    if (requestId !== state.requestId) return;
    state.sample = sample;
    $('sample-progress').textContent = `${sample.gamesEligible} eligible games${sample.warnings.length ? ' · see data note' : ''}`;
    updateNote(sample);
    renderResults();
  } catch (error) {
    if (requestId !== state.requestId || error?.name === 'AbortError') return;
    $('sample-progress').textContent = `Could not load data: ${error.message || 'Request failed.'} Try again.`;
    renderResults();
  } finally {
    if (requestId === state.requestId) {
      state.loading = false;
      state.controller = null;
      $('load-button').disabled = false;
      renderResults();
    }
  }
}

$('hero-search').addEventListener('input', renderPicker);
$('close-dialog').addEventListener('click', () => $('hero-dialog').close());
$('load-button').addEventListener('click', loadSample);
$('reset-button').addEventListener('click', () => {
  state.enemies = [];
  state.allies = [];
  state.bans = [];
  renderSlots();
  renderResults();
});
function resetFilters() {
  state.requestId++;
  state.controller?.abort();
  state.controller = null;
  state.loading = false;
  state.sample = null;
  $('load-button').disabled = false;
  $('sample-progress').textContent = 'Filters changed. Load the match data again.';
  updateNote(null);
  renderResults();
}
$('rank-select').addEventListener('change', resetFilters);
$('patch-input').addEventListener('input', resetFilters);
for (const [id, tab] of [['tab-picks', 'picks'], ['tab-avoid', 'avoid']]) {
  $(id).addEventListener('click', () => { state.tab = tab; renderResults(); });
  $(id).addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    state.tab = event.key === 'Home' ? 'picks' : event.key === 'End' ? 'avoid' : state.tab === 'picks' ? 'avoid' : 'picks';
    renderResults();
    $(state.tab === 'picks' ? 'tab-picks' : 'tab-avoid').focus();
  });
}

renderSlots();
ensureHeroes().catch(error => {
  $('sample-progress').textContent = `Could not load heroes: ${error.message || 'Request failed.'} Use Load match data to retry.`;
  renderResults();
});
