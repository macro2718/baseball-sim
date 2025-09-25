// Application-wide configuration and constant lookups.

export const CONFIG = {
  maxOutsDisplay: 3,
  api: {
    endpoints: {
      gameState: '/api/game/state',
      gameStart: '/api/game/start',
      gameRestart: '/api/game/restart',
      gameStop: '/api/game/stop',
      gameSwing: '/api/game/swing',
      gameBunt: '/api/game/bunt',
      pinchHit: '/api/strategy/pinch_hit',
      pinchRun: '/api/strategy/pinch_run',
      defenseSubstitution: '/api/strategy/defense_substitution',
      changePitcher: '/api/strategy/change_pitcher',
      clearLog: '/api/log/clear',
      simulationRun: '/api/simulation/run',
      reloadTeams: '/api/teams/reload',
      teamSelect: '/api/teams/library/select',
      teamSave: '/api/teams/library/save',
      teamDelete: '/api/teams/library/delete',
      teamDetail: '/api/teams/library',
      // Player endpoints
      playersList: '/api/players/list',
      playersCatalog: '/api/players/catalog',
      playerDetail: '/api/players/detail',
      playerSave: '/api/players/save',
      playerDelete: '/api/players/delete',
      health: '/api/health',
    },
  },
  css: {
    classes: {
      positionToken: 'position-token',
      selected: 'selected',
      disabled: 'disabled',
      invalid: 'invalid',
    },
  },
};

export const FIELD_POSITIONS = [
  { key: 'CF', label: 'CF', className: 'pos-cf' },
  { key: 'LF', label: 'LF', className: 'pos-lf' },
  { key: 'RF', label: 'RF', className: 'pos-rf' },
  { key: '2B', label: '2B', className: 'pos-2b' },
  { key: 'SS', label: 'SS', className: 'pos-ss' },
  { key: '3B', label: '3B', className: 'pos-3b' },
  { key: '1B', label: '1B', className: 'pos-1b' },
  { key: 'P', label: 'P', className: 'pos-p' },
  { key: 'C', label: 'C', className: 'pos-c' },
  { key: 'DH', label: 'DH', className: 'pos-dh' },
];

export const FIELD_POSITION_KEYS = new Set(FIELD_POSITIONS.map((slot) => slot.key));

export const BATTING_COLUMNS = [
  { label: 'Player', key: 'name' },
  { label: 'PA', key: 'pa' },
  { label: 'AB', key: 'ab' },
  { label: '1B', key: 'single' },
  { label: '2B', key: 'double' },
  { label: '3B', key: 'triple' },
  { label: 'HR', key: 'hr' },
  { label: 'R', key: 'runs' },
  { label: 'RBI', key: 'rbi' },
  { label: 'BB', key: 'bb' },
  { label: 'SO', key: 'so' },
  { label: 'AVG', key: 'avg' },
];

export const PITCHING_COLUMNS = [
  { label: 'Player', key: 'name' },
  { label: 'IP', key: 'ip' },
  { label: 'H', key: 'h' },
  { label: 'R', key: 'r' },
  { label: 'ER', key: 'er' },
  { label: 'BB', key: 'bb' },
  { label: 'SO', key: 'so' },
  { label: 'ERA', key: 'era' },
  { label: 'WHIP', key: 'whip' },
];

export const ABILITY_BATTING_COLUMNS = [
  { label: 'Player', key: 'name' },
  { label: '区分', key: 'role_label' },
  { label: 'Pos', key: 'position' },
  { label: '打席', key: 'bats' },
  { label: 'K%', key: 'k_pct' },
  { label: 'BB%', key: 'bb_pct' },
  { label: 'Hard%', key: 'hard_pct' },
  { label: 'GB%', key: 'gb_pct' },
  { label: 'Speed', key: 'speed' },
  { label: 'Field', key: 'fielding' },
];

export const ABILITY_PITCHING_COLUMNS = [
  { label: 'Player', key: 'name' },
  { label: '区分', key: 'role_label' },
  { label: 'Type', key: 'pitcher_type' },
  { label: 'Throws', key: 'throws' },
  { label: 'K%', key: 'k_pct' },
  { label: 'BB%', key: 'bb_pct' },
  { label: 'Hard%', key: 'hard_pct' },
  { label: 'GB%', key: 'gb_pct' },
  { label: 'Stamina', key: 'stamina' },
];
