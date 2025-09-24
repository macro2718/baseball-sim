import { elements } from '../dom.js';
import { stateCache, setUIView } from '../state.js';
import {
  hideDefenseMenu,
  hideOffenseMenu,
  toggleDefenseMenu,
  toggleLogPanel,
  toggleOffenseMenu,
} from '../ui/menus.js';
import { closeModal, openModal, resolveModal } from '../ui/modals.js';
import {
  updateStatsPanel,
  updateAbilitiesPanel,
  render,
  updateScreenVisibility,
} from '../ui/renderers.js';
import { showStatus } from '../ui/status.js';
import { handleDefensePlayerClick, updateDefenseSelectionInfo } from '../ui/defensePanel.js';

const DEFAULT_TEAM_TEMPLATE = JSON.stringify(
  {
    name: 'New Team',
    pitchers: ['Pitcher 1'],
    lineup: [
      { name: 'Player 1', position: 'C' },
      { name: 'Player 2', position: '1B' },
      { name: 'Player 3', position: '2B' },
      { name: 'Player 4', position: '3B' },
      { name: 'Player 5', position: 'SS' },
      { name: 'Player 6', position: 'LF' },
      { name: 'Player 7', position: 'CF' },
      { name: 'Player 8', position: 'RF' },
      { name: 'Player 9', position: 'DH' },
    ],
    bench: ['Reserve 1'],
  },
  null,
  2
);

const BATTER_POSITIONS = ['C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF'];
const PITCHER_TYPES = ['SP', 'RP'];

const DEFAULT_PLAYER_TEMPLATES = {
  batter: {
    name: 'New Batter',
    eligible_positions: ['LF', 'CF', 'RF', 'DH'],
    bats: 'R',
    k_pct: 22.8,
    bb_pct: 8.5,
    hard_pct: 38.6,
    gb_pct: 44.6,
    speed: 4.3,
    fielding_skill: 100,
  },
  pitcher: {
    name: 'New Pitcher',
    pitcher_type: 'SP',
    throws: 'R',
    k_pct: 22.8,
    bb_pct: 8.5,
    hard_pct: 38.6,
    gb_pct: 44.6,
    stamina: 80,
  },
};

function clonePlayerTemplate(role = 'batter') {
  const template = DEFAULT_PLAYER_TEMPLATES[role] || DEFAULT_PLAYER_TEMPLATES.batter;
  return JSON.parse(JSON.stringify(template));
}

function setPlayerBuilderFeedback(message, level = 'info') {
  if (!elements.playerBuilderFeedback) return;
  const el = elements.playerBuilderFeedback;
  el.textContent = message || '';
  el.classList.remove('danger', 'success', 'info');
  if (message) {
    el.classList.add(level);
    el.dataset.level = level;
  } else {
    el.removeAttribute('data-level');
  }
}

function updateChipToggleState(button, selected) {
  if (!button) return;
  if (selected) {
    button.classList.add('active');
    button.setAttribute('aria-pressed', 'true');
  } else {
    button.classList.remove('active');
    button.setAttribute('aria-pressed', 'false');
  }
}

function setSelectedPositions(positions) {
  const selectedSet = new Set((positions || []).map((pos) => String(pos).toUpperCase()));
  elements.playerPositionButtons?.forEach((button) => {
    const key = String(button.dataset.positionOption || '').toUpperCase();
    updateChipToggleState(button, selectedSet.has(key));
  });
}

function getSelectedPositions() {
  return (elements.playerPositionButtons || [])
    .filter((button) => button.classList.contains('active'))
    .map((button) => String(button.dataset.positionOption || '').toUpperCase())
    .filter(Boolean);
}

function setSelectedPitcherType(type) {
  const normalized = typeof type === 'string' ? type.toUpperCase() : '';
  let matched = false;
  elements.playerPitcherTypeButtons?.forEach((button, index) => {
    const key = String(button.dataset.pitcherType || '').toUpperCase();
    const shouldSelect = key === normalized;
    updateChipToggleState(button, shouldSelect);
    if (shouldSelect) {
      matched = true;
    }
    if (!normalized && index === 0 && !matched) {
      updateChipToggleState(button, true);
      matched = true;
    }
  });
  if (!matched && elements.playerPitcherTypeButtons?.length) {
    const [first] = elements.playerPitcherTypeButtons;
    updateChipToggleState(first, true);
    return String(first.dataset.pitcherType || 'SP').toUpperCase();
  }
  const active = elements.playerPitcherTypeButtons?.find((button) => button.classList.contains('active'));
  return active ? String(active.dataset.pitcherType || '').toUpperCase() : null;
}

function getSelectedPitcherType() {
  const active = elements.playerPitcherTypeButtons?.find((button) => button.classList.contains('active'));
  return active ? String(active.dataset.pitcherType || '').toUpperCase() : null;
}

function updatePlayerRoleUI(role = 'batter') {
  const normalized = role === 'pitcher' ? 'pitcher' : 'batter';
  if (elements.playerEditorRole) {
    elements.playerEditorRole.value = normalized;
  }
  elements.playerRoleButtons?.forEach((button) => {
    const value = String(button.dataset.roleChoice || '').toLowerCase();
    const isActive = value === normalized;
    updateChipToggleState(button, isActive);
  });
  elements.playerRoleSections?.forEach((section) => {
    const roles = (section.dataset.roleSection || '')
      .split(',')
      .map((value) => value.trim().toLowerCase())
      .filter(Boolean);
    const shouldShow = !roles.length || roles.includes(normalized);
    section.classList.toggle('hidden', !shouldShow);
    if (shouldShow) {
      section.removeAttribute('aria-hidden');
    } else {
      section.setAttribute('aria-hidden', 'true');
    }
  });
  const enablePositions = normalized === 'batter';
  elements.playerPositionButtons?.forEach((button) => {
    button.disabled = !enablePositions;
    if (!enablePositions) {
      updateChipToggleState(button, false);
    }
  });
  const enablePitcher = normalized === 'pitcher';
  elements.playerPitcherTypeButtons?.forEach((button) => {
    button.disabled = !enablePitcher;
    if (!enablePitcher) {
      updateChipToggleState(button, false);
    }
  });
  return normalized;
}

function clearPlayerForm(role = 'batter') {
  if (elements.playerEditorName) elements.playerEditorName.value = '';
  if (elements.playerEditorKPct) elements.playerEditorKPct.value = '';
  if (elements.playerEditorBBPct) elements.playerEditorBBPct.value = '';
  if (elements.playerEditorHardPct) elements.playerEditorHardPct.value = '';
  if (elements.playerEditorGBPct) elements.playerEditorGBPct.value = '';
  setSelectedPositions([]);
  const normalized = role === 'pitcher' ? 'pitcher' : 'batter';
  if (normalized === 'batter') {
    if (elements.playerEditorBats) elements.playerEditorBats.value = 'R';
    if (elements.playerEditorSpeed) elements.playerEditorSpeed.value = '';
    if (elements.playerEditorFielding) elements.playerEditorFielding.value = '';
  } else {
    if (elements.playerEditorThrows) elements.playerEditorThrows.value = 'R';
    if (elements.playerEditorStamina) elements.playerEditorStamina.value = '';
    setSelectedPitcherType('SP');
  }
}

function setInputValue(input, value) {
  if (!input) return;
  if (value === null || value === undefined || value === '') {
    input.value = '';
    return;
  }
  const numeric = Number(value);
  if (Number.isFinite(numeric)) {
    input.value = String(numeric);
  } else {
    input.value = String(value);
  }
}

function applyPlayerFormData(player, role = 'batter') {
  const normalized = updatePlayerRoleUI(role);
  if (!player) {
    clearPlayerForm(normalized);
    return;
  }
  if (elements.playerEditorName) elements.playerEditorName.value = player.name || '';
  setInputValue(elements.playerEditorKPct, player.k_pct);
  setInputValue(elements.playerEditorBBPct, player.bb_pct);
  setInputValue(elements.playerEditorHardPct, player.hard_pct);
  setInputValue(elements.playerEditorGBPct, player.gb_pct);
  if (normalized === 'batter') {
    if (elements.playerEditorBats) {
      elements.playerEditorBats.value = player.bats || 'R';
    }
    setInputValue(elements.playerEditorSpeed, player.speed);
    setInputValue(elements.playerEditorFielding, player.fielding_skill);
    const eligible = Array.isArray(player.eligible_positions)
      ? player.eligible_positions
      : Array.isArray(player.eligible)
      ? player.eligible
      : [];
    const filtered = eligible
      .map((pos) => String(pos).toUpperCase())
      .filter((pos) => pos && pos !== 'DH');
    setSelectedPositions(filtered);
  } else {
    if (elements.playerEditorThrows) {
      elements.playerEditorThrows.value = player.throws || 'R';
    }
    setInputValue(elements.playerEditorStamina, player.stamina);
    const type = typeof player.pitcher_type === 'string' ? player.pitcher_type.toUpperCase() : 'SP';
    setSelectedPitcherType(PITCHER_TYPES.includes(type) ? type : 'SP');
  }
}

function readNumberField(input, label, { allowEmpty = false } = {}) {
  if (!input) return { value: null };
  const raw = String(input.value ?? '').trim();
  if (!raw) {
    if (allowEmpty) {
      return { value: null };
    }
    return { error: `${label}を入力してください。` };
  }
  const value = Number.parseFloat(raw);
  if (Number.isNaN(value)) {
    return { error: `${label}には数値を入力してください。` };
  }
  return { value };
}

function getPlayerFormData(role = 'batter') {
  const normalized = role === 'pitcher' ? 'pitcher' : 'batter';
  const name = elements.playerEditorName?.value?.trim() || '';
  if (!name) {
    setPlayerBuilderFeedback('名前を入力してください。', 'danger');
    return null;
  }

  const kPct = readNumberField(elements.playerEditorKPct, 'K%');
  if (kPct.error) {
    setPlayerBuilderFeedback(kPct.error, 'danger');
    return null;
  }
  const bbPct = readNumberField(elements.playerEditorBBPct, 'BB%');
  if (bbPct.error) {
    setPlayerBuilderFeedback(bbPct.error, 'danger');
    return null;
  }
  const hardPct = readNumberField(elements.playerEditorHardPct, 'Hard%');
  if (hardPct.error) {
    setPlayerBuilderFeedback(hardPct.error, 'danger');
    return null;
  }
  const gbPct = readNumberField(elements.playerEditorGBPct, 'GB%');
  if (gbPct.error) {
    setPlayerBuilderFeedback(gbPct.error, 'danger');
    return null;
  }

  const baseData = {
    name,
    k_pct: kPct.value,
    bb_pct: bbPct.value,
    hard_pct: hardPct.value,
    gb_pct: gbPct.value,
  };

  if (normalized === 'batter') {
    const speed = readNumberField(elements.playerEditorSpeed, 'Speed');
    if (speed.error) {
      setPlayerBuilderFeedback(speed.error, 'danger');
      return null;
    }
    const fielding = readNumberField(elements.playerEditorFielding, 'Fielding');
    if (fielding.error) {
      setPlayerBuilderFeedback(fielding.error, 'danger');
      return null;
    }
    const selectedPositions = getSelectedPositions();
    const uniquePositions = Array.from(
      new Set(
        selectedPositions.filter((pos) => BATTER_POSITIONS.includes(pos)).map((pos) => pos.toUpperCase()),
      ),
    );
    const positionsWithDh = Array.from(new Set([...uniquePositions, 'DH']));
    return {
      ...baseData,
      eligible_positions: positionsWithDh,
      bats: elements.playerEditorBats?.value || 'R',
      speed: speed.value,
      fielding_skill: fielding.value,
    };
  }

  const stamina = readNumberField(elements.playerEditorStamina, 'Stamina');
  if (stamina.error) {
    setPlayerBuilderFeedback(stamina.error, 'danger');
    return null;
  }
  const type = getSelectedPitcherType();
  if (!type) {
    setPlayerBuilderFeedback('投手タイプを選択してください。', 'danger');
    return null;
  }
  return {
    ...baseData,
    pitcher_type: type,
    throws: elements.playerEditorThrows?.value || 'R',
    stamina: stamina.value,
  };
}

function loadPlayerTemplate(role = 'batter') {
  const normalized = updatePlayerRoleUI(role);
  const template = clonePlayerTemplate(normalized);
  applyPlayerFormData(template, normalized);
  if (elements.playerEditorSelect) {
    elements.playerEditorSelect.value = '__new__';
  }
  setPlayerBuilderFeedback('テンプレートを読み込みました。', 'info');
}

function setTeamBuilderFeedback(message, level = 'info') {
  if (!elements.teamBuilderFeedback) return;
  const feedback = elements.teamBuilderFeedback;
  feedback.textContent = message || '';
  feedback.classList.remove('danger', 'success', 'info');
  if (message) {
    feedback.classList.add(level);
    feedback.dataset.level = level;
  } else {
    feedback.removeAttribute('data-level');
  }
}

function loadTeamTemplate() {
  if (!elements.teamEditorJson) return;
  elements.teamEditorJson.value = DEFAULT_TEAM_TEMPLATE;
  if (elements.teamEditorSelect) {
    elements.teamEditorSelect.value = '__new__';
  }
  stateCache.teamBuilder.currentTeamId = null;
  stateCache.teamBuilder.editorDirty = false;
  stateCache.teamBuilder.lastSavedId = '__new__';
  setTeamBuilderFeedback('テンプレートを読み込みました。', 'info');
}

function refreshView() {
  if (stateCache.data) {
    render(stateCache.data);
  } else {
    updateScreenVisibility();
  }
}

export function initEventListeners(actions) {
  async function populatePlayerSelect(role, desiredValue) {
    const select = elements.playerEditorSelect;
    if (!select) return [];
    const players = await actions.fetchPlayersList(role);
    const prevValue = desiredValue ?? select.value;
    select.innerHTML = '';
    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = '選手を選択';
    select.appendChild(placeholder);
    const newOpt = document.createElement('option');
    newOpt.value = '__new__';
    newOpt.textContent = '新規選手を作成';
    select.appendChild(newOpt);
    players.forEach((p) => {
      const opt = document.createElement('option');
      opt.value = p.id; // use stable id
      opt.textContent = p.name;
      select.appendChild(opt);
    });
    const validValues = new Set(['', '__new__', ...players.map((p) => p.id)]);
    const targetValue = prevValue && validValues.has(prevValue) ? prevValue : '';
    select.value = targetValue;
    return players;
  }

  updatePlayerRoleUI(elements.playerEditorRole?.value || 'batter');

  elements.startButton.addEventListener('click', () => actions.handleStart(false));
  elements.reloadTeams.addEventListener('click', actions.handleReloadTeams);
  elements.restartButton.addEventListener('click', () => actions.handleStart(true));
  elements.returnTitle.addEventListener('click', actions.handleReturnToTitle);
  elements.clearLog.addEventListener('click', actions.handleClearLog);
  elements.swingButton.addEventListener('click', actions.handleSwing);
  elements.buntButton.addEventListener('click', actions.handleBunt);

  if (elements.pinchButton) {
    elements.pinchButton.addEventListener('click', actions.handlePinchHit);
  }
  if (elements.openOffenseButton) {
    elements.openOffenseButton.addEventListener('click', toggleOffenseMenu);
  }
  if (elements.offensePinchMenuButton) {
    elements.offensePinchMenuButton.addEventListener('click', () => openModal('offense'));
  }
  if (elements.openDefenseButton) {
    elements.openDefenseButton.addEventListener('click', toggleDefenseMenu);
  }
  if (elements.defenseSubMenuButton) {
    elements.defenseSubMenuButton.addEventListener('click', () => {
      updateDefenseSelectionInfo();
      openModal('defense');
    });
  }
  if (elements.pitcherMenuButton) {
    elements.pitcherMenuButton.addEventListener('click', () => openModal('pitcher'));
  }
  if (elements.openStatsButton) {
    elements.openStatsButton.addEventListener('click', () => {
      updateStatsPanel(stateCache.data);
      openModal('stats');
    });
  }
  if (elements.openAbilitiesButton) {
    elements.openAbilitiesButton.addEventListener('click', () => {
      updateAbilitiesPanel(stateCache.data);
      openModal('abilities');
    });
  }
  if (elements.defenseResetButton) {
    elements.defenseResetButton.addEventListener('click', actions.handleDefenseReset);
  }
  if (elements.defenseApplyButton) {
    elements.defenseApplyButton.addEventListener('click', actions.handleDefenseSubstitution);
  }
  if (elements.pitcherButton) {
    elements.pitcherButton.addEventListener('click', actions.handlePitcherChange);
  }

  if (elements.enterTitleButton) {
    elements.enterTitleButton.addEventListener('click', async () => {
      if (!elements.lobbyHomeSelect || !elements.lobbyAwaySelect) {
        return;
      }
      const homeId = elements.lobbyHomeSelect.value;
      const awayId = elements.lobbyAwaySelect.value;
      if (!homeId || !awayId) {
        showStatus('ホーム・アウェイのチームを選択してください。', 'danger');
        return;
      }
      elements.enterTitleButton.disabled = true;
      try {
        await actions.handleTeamSelection(homeId, awayId);
      } catch (error) {
        // エラーハンドリングはhandleTeamSelection内で行われる
      } finally {
        elements.enterTitleButton.disabled = false;
      }
    });
  }

  if (elements.openTeamBuilder) {
    elements.openTeamBuilder.addEventListener('click', () => {
      setUIView('team-builder');
      refreshView();
      setTeamBuilderFeedback('編集するチームを選択するか、新規作成してください。', 'info');
    });
  }

  if (elements.openMatchButton) {
    elements.openMatchButton.addEventListener('click', () => {
      setUIView('team-select');
      refreshView();
    });
  }

  if (elements.openPlayerBuilder) {
    elements.openPlayerBuilder.addEventListener('click', () => {
      setUIView('player-builder');
      refreshView();
      setPlayerBuilderFeedback('区分と選手を選択するか、新規作成してください。', 'info');
      const role = updatePlayerRoleUI(elements.playerEditorRole?.value || 'batter');
      clearPlayerForm(role);
      if (elements.playerEditorSelect) {
        elements.playerEditorSelect.value = '';
      }
      (async () => {
        await populatePlayerSelect(role);
      })();
    });
  }

  if (elements.backToLobby) {
    elements.backToLobby.addEventListener('click', () => {
      setUIView('lobby');
      refreshView();
    });
  }

  if (elements.backToTeamSelect) {
    elements.backToTeamSelect.addEventListener('click', () => {
      setUIView('team-select');
      refreshView();
    });
  }

  if (elements.teamBuilderBack) {
    elements.teamBuilderBack.addEventListener('click', () => {
      setUIView('lobby');
      refreshView();
    });
  }

  if (elements.playerBuilderBack) {
    elements.playerBuilderBack.addEventListener('click', () => {
      setUIView('lobby');
      refreshView();
    });
  }

  if (elements.teamBuilderNew) {
    elements.teamBuilderNew.addEventListener('click', () => {
      loadTeamTemplate();
    });
  }

  if (elements.playerBuilderNew) {
    elements.playerBuilderNew.addEventListener('click', () => {
      const role = elements.playerEditorRole?.value || 'batter';
      loadPlayerTemplate(role);
    });
  }

  if (elements.teamEditorSelect) {
    elements.teamEditorSelect.addEventListener('change', async (event) => {
      const selectValue = event.target.value;
      if (selectValue === '__new__') {
        loadTeamTemplate();
        return;
      }
      if (!selectValue) {
        if (elements.teamEditorJson) {
          elements.teamEditorJson.value = '';
        }
        stateCache.teamBuilder.currentTeamId = null;
        stateCache.teamBuilder.editorDirty = false;
        setTeamBuilderFeedback('編集するチームを選択してください。', 'info');
        return;
      }
      setTeamBuilderFeedback('チームデータを読み込み中...', 'info');
      const teamData = await actions.fetchTeamDefinition(selectValue);
      if (teamData && elements.teamEditorJson) {
        elements.teamEditorJson.value = JSON.stringify(teamData, null, 2);
        stateCache.teamBuilder.currentTeamId = selectValue;
        stateCache.teamBuilder.editorDirty = false;
        setTeamBuilderFeedback('チームデータを読み込みました。', 'info');
      } else {
        setTeamBuilderFeedback('チームデータの読み込みに失敗しました。', 'danger');
      }
    });
  }

  if (elements.playerRoleButtons?.length) {
    elements.playerRoleButtons.forEach((button) => {
      button.addEventListener('click', () => {
        const value = String(button.dataset.roleChoice || '').toLowerCase();
        const normalized = value === 'pitcher' ? 'pitcher' : 'batter';
        if (elements.playerEditorRole?.value === normalized) {
          return;
        }
        updatePlayerRoleUI(normalized);
        clearPlayerForm(normalized);
        if (elements.playerEditorSelect) {
          elements.playerEditorSelect.value = '';
        }
        setPlayerBuilderFeedback('区分が変更されました。選手を選ぶかテンプレートを作成してください。', 'info');
        (async () => {
          await populatePlayerSelect(normalized);
        })();
      });
    });
  }

  if (elements.playerPositionButtons?.length) {
    elements.playerPositionButtons.forEach((button) => {
      button.addEventListener('click', () => {
        const role = elements.playerEditorRole?.value || 'batter';
        if (role !== 'batter' || button.disabled) {
          return;
        }
        const isActive = button.classList.contains('active');
        updateChipToggleState(button, !isActive);
      });
    });
  }

  if (elements.playerPitcherTypeButtons?.length) {
    elements.playerPitcherTypeButtons.forEach((button) => {
      button.addEventListener('click', () => {
        const role = elements.playerEditorRole?.value || 'batter';
        if (role !== 'pitcher' || button.disabled) {
          return;
        }
        elements.playerPitcherTypeButtons.forEach((btn) => {
          updateChipToggleState(btn, btn === button);
        });
      });
    });
  }

  if (elements.playerEditorSelect) {
    elements.playerEditorSelect.addEventListener('change', async (event) => {
      const selectValue = event.target.value;
      if (selectValue === '__new__') {
        const role = elements.playerEditorRole?.value || 'batter';
        loadPlayerTemplate(role);
        return;
      }
      if (!selectValue) {
        const role = elements.playerEditorRole?.value || 'batter';
        clearPlayerForm(role);
        setPlayerBuilderFeedback('編集する選手を選択してください。', 'info');
        return;
      }
      setPlayerBuilderFeedback('選手データを読み込み中...', 'info');
      const result = await actions.fetchPlayerDefinition(selectValue); // pass id
      const fetchedPlayer = result?.player || null;
      const fetchedRole = result?.role === 'pitcher' ? 'pitcher' : result?.role === 'batter' ? 'batter' : null;
      const currentRole = elements.playerEditorRole?.value || 'batter';
      const roleToUse = fetchedRole || currentRole;
      if (fetchedRole && fetchedRole !== currentRole) {
        updatePlayerRoleUI(fetchedRole);
        await populatePlayerSelect(fetchedRole, selectValue);
      }
      if (fetchedPlayer) {
        applyPlayerFormData(fetchedPlayer, roleToUse);
        // Disable delete if referenced by any team
        const hasRefs = Boolean(result?.hasReferences);
        if (elements.playerBuilderDelete) {
          elements.playerBuilderDelete.disabled = hasRefs;
        }
        if (hasRefs) {
          const refs = Array.isArray(result?.referencedBy) ? result.referencedBy : [];
          const list = refs.slice(0, 5).join(', ') + (refs.length > 5 ? ` 他${refs.length - 5}件` : '');
          setPlayerBuilderFeedback(`この選手は以下のチームに含まれているため削除できません: ${list}`, 'warning');
        } else {
          setPlayerBuilderFeedback('選手データを読み込みました。', 'info');
        }
      } else {
        clearPlayerForm(roleToUse);
        setPlayerBuilderFeedback('選手データの読み込みに失敗しました。', 'danger');
      }
    });
  }

  if (elements.playerBuilderSave) {
    elements.playerBuilderSave.addEventListener('click', async () => {
      const role = elements.playerEditorRole?.value || 'batter';
      const formData = getPlayerFormData(role);
      if (!formData) {
        return;
      }
      // If editing an existing player, capture the selected id and visible name
      const selectEl = elements.playerEditorSelect;
      const selectedValue = selectEl?.value || '';
      const selectedText = selectEl?.selectedOptions?.[0]?.textContent || null;
      const isEditing = selectedValue && selectedValue !== '__new__';
      const playerId = isEditing ? selectedValue : null;
      const originalName = isEditing ? selectedText : null;
      elements.playerBuilderSave.disabled = true;
      try {
        const saved = await actions.handlePlayerSave(formData, role, originalName, playerId);
        if (saved?.id) {
          await populatePlayerSelect(role, saved.id);
          setPlayerBuilderFeedback('選手を保存しました。', 'success');
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : '選手の保存に失敗しました。';
        setPlayerBuilderFeedback(message, 'danger');
      } finally {
        elements.playerBuilderSave.disabled = false;
      }
    });
  }

  if (elements.playerBuilderDelete) {
    elements.playerBuilderDelete.addEventListener('click', async () => {
      const role = elements.playerEditorRole?.value || 'batter';
      const selectEl = elements.playerEditorSelect;
      const idValue = selectEl?.value || '';
      const nameText = selectEl?.selectedOptions?.[0]?.textContent || '';
      if (!idValue || idValue === '__new__') {
        setPlayerBuilderFeedback('削除する選手を選択してください。', 'danger');
        return;
      }
      if (elements.playerBuilderDelete.disabled) {
        // Should not happen due to disabled state, but double-guard.
        setPlayerBuilderFeedback('この選手はチームで使用中のため削除できません。', 'warning');
        return;
      }
      const confirmed = window.confirm(`選手 '${nameText}' を削除します。よろしいですか？`);
      if (!confirmed) return;
      elements.playerBuilderDelete.disabled = true;
      try {
        await actions.handlePlayerDelete(idValue, role, nameText);
        await populatePlayerSelect(role);
        clearPlayerForm(role);
        setPlayerBuilderFeedback('選手を削除しました。', 'success');
      } catch (error) {
        const message = error instanceof Error ? error.message : '選手の削除に失敗しました。';
        setPlayerBuilderFeedback(message, 'danger');
      } finally {
        elements.playerBuilderDelete.disabled = false;
      }
    });
  }

  if (elements.teamEditorJson) {
    elements.teamEditorJson.addEventListener('input', () => {
      stateCache.teamBuilder.editorDirty = true;
    });
  }

  if (elements.teamBuilderSave) {
    elements.teamBuilderSave.addEventListener('click', async () => {
      if (!elements.teamEditorJson) return;
      const raw = elements.teamEditorJson.value;
      let parsed;
      try {
        parsed = JSON.parse(raw);
      } catch (error) {
        const message = error instanceof Error ? error.message : 'JSONを解析できません。';
        setTeamBuilderFeedback(`JSONの構文エラー: ${message}`, 'danger');
        return;
      }

      elements.teamBuilderSave.disabled = true;
      try {
        const savedId = await actions.handleTeamSave(
          stateCache.teamBuilder.currentTeamId,
          parsed,
        );
        if (savedId) {
          stateCache.teamBuilder.currentTeamId = savedId;
          stateCache.teamBuilder.lastSavedId = savedId;
          stateCache.teamBuilder.editorDirty = false;
          if (elements.teamEditorSelect) {
            elements.teamEditorSelect.value = savedId;
          }
          setTeamBuilderFeedback('チームを保存しました。', 'success');
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : 'チームの保存に失敗しました。';
        setTeamBuilderFeedback(message, 'danger');
      } finally {
        elements.teamBuilderSave.disabled = false;
      }
    });
  }

  if (elements.teamBuilderDelete) {
    elements.teamBuilderDelete.addEventListener('click', async () => {
      const teamId = stateCache.teamBuilder.currentTeamId;
      if (!teamId) {
        setTeamBuilderFeedback('削除するチームを選択してください。', 'danger');
        return;
      }
      const confirmed = window.confirm(`チーム '${teamId}' を削除します。よろしいですか？`);
      if (!confirmed) return;
      elements.teamBuilderDelete.disabled = true;
      try {
        await actions.handleTeamDelete(teamId);
        if (elements.teamEditorJson) {
          elements.teamEditorJson.value = '';
        }
        stateCache.teamBuilder.currentTeamId = null;
        stateCache.teamBuilder.lastSavedId = null;
        stateCache.teamBuilder.editorDirty = false;
        setTeamBuilderFeedback('チームを削除しました。', 'success');
        setUIView('team-builder');
        if (stateCache.data) {
          render(stateCache.data);
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : 'チームの削除に失敗しました。';
        setTeamBuilderFeedback(message, 'danger');
      } finally {
        elements.teamBuilderDelete.disabled = false;
      }
    });
  }

  elements.modalCloseButtons.forEach((button) => {
    const target = button.dataset.close;
    button.addEventListener('click', () => closeModal(target || button.closest('.modal')));
  });

  ['offense', 'defense', 'pitcher', 'stats', 'abilities'].forEach((name) => {
    const modal = resolveModal(name);
    if (modal) {
      modal.addEventListener('click', (event) => {
        if (event.target === modal) {
          closeModal(modal);
        }
      });
    }
  });

  if (elements.offenseMenu) {
    document.addEventListener('click', (event) => {
      if (elements.offenseMenu.classList.contains('hidden')) {
        return;
      }
      const clickedInsideMenu = elements.offenseMenu.contains(event.target);
      const clickedToggle =
        elements.openOffenseButton && elements.openOffenseButton.contains(event.target);
      if (!clickedInsideMenu && !clickedToggle) {
        hideOffenseMenu();
      }
    });
  }

  if (elements.defenseMenu) {
    document.addEventListener('click', (event) => {
      if (elements.defenseMenu.classList.contains('hidden')) {
        return;
      }
      const clickedInsideMenu = elements.defenseMenu.contains(event.target);
      const clickedToggle =
        elements.openDefenseButton && elements.openDefenseButton.contains(event.target);
      if (!clickedInsideMenu && !clickedToggle) {
        hideDefenseMenu();
      }
    });
  }

  if (elements.defenseField) {
    elements.defenseField.addEventListener('click', handleDefensePlayerClick);
  }
  if (elements.defenseBench) {
    elements.defenseBench.addEventListener('click', handleDefensePlayerClick);
  }
  if (elements.defenseExtras) {
    elements.defenseExtras.addEventListener('click', handleDefensePlayerClick);
  }

  elements.statsTeamButtons.forEach((button) => {
    button.addEventListener('click', () => {
      if (button.disabled) return;
      const teamKey = button.dataset.statsTeam;
      if (!teamKey) return;
      stateCache.statsView.team = teamKey;
      updateStatsPanel(stateCache.data);
    });
  });

  elements.statsTypeButtons.forEach((button) => {
    button.addEventListener('click', () => {
      const viewType = button.dataset.statsType;
      if (!viewType) return;
      stateCache.statsView.type = viewType;
      updateStatsPanel(stateCache.data);
    });
  });

  elements.abilitiesTeamButtons.forEach((button) => {
    button.addEventListener('click', () => {
      if (button.disabled) return;
      const teamKey = button.dataset.abilitiesTeam;
      if (!teamKey) return;
      stateCache.abilitiesView.team = teamKey;
      updateAbilitiesPanel(stateCache.data);
    });
  });

  elements.abilitiesTypeButtons.forEach((button) => {
    button.addEventListener('click', () => {
      if (button.disabled) return;
      const viewType = button.dataset.abilitiesType;
      if (!viewType) return;
      stateCache.abilitiesView.type = viewType;
      updateAbilitiesPanel(stateCache.data);
    });
  });

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      hideOffenseMenu();
      hideDefenseMenu();
      ['offense', 'defense', 'pitcher', 'stats', 'abilities'].forEach((name) => {
        const modal = resolveModal(name);
        if (modal && !modal.classList.contains('hidden')) {
          closeModal(modal);
        }
      });
    }
    if (event.key === 'Tab' && !event.ctrlKey && !event.altKey && !event.shiftKey) {
      event.preventDefault();
      toggleLogPanel();
    }
  });

  updateDefenseSelectionInfo();
}
