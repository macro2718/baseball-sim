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

const DEFAULT_PLAYER_TEMPLATES = {
  batter: JSON.stringify(
    {
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
    null,
    2,
  ),
  pitcher: JSON.stringify(
    {
      name: 'New Pitcher',
      pitcher_type: 'SP',
      throws: 'R',
      k_pct: 22.8,
      bb_pct: 8.5,
      hard_pct: 38.6,
      gb_pct: 44.6,
      stamina: 80,
    },
    null,
    2,
  ),
};

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

function loadPlayerTemplate(role = 'batter') {
  if (!elements.playerEditorJson) return;
  const tpl = DEFAULT_PLAYER_TEMPLATES[role] || DEFAULT_PLAYER_TEMPLATES.batter;
  elements.playerEditorJson.value = tpl;
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
      (async () => {
        const role = elements.playerEditorRole?.value || 'batter';
        const players = await actions.fetchPlayersList(role);
        const select = elements.playerEditorSelect;
        if (select) {
          const prev = select.value;
          select.innerHTML = '';
          const ph = document.createElement('option');
          ph.value = '';
          ph.textContent = '選手を選択';
          select.appendChild(ph);
          const newOpt = document.createElement('option');
          newOpt.value = '__new__';
          newOpt.textContent = '新規選手を作成';
          select.appendChild(newOpt);
          players.forEach((name) => {
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = name;
            select.appendChild(opt);
          });
          select.value = prev && players.includes(prev) ? prev : '';
        }
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

  if (elements.playerEditorRole) {
    elements.playerEditorRole.addEventListener('change', () => {
      // 切替時は選手一覧を再取得し、テンプレートを表示
      if (elements.playerEditorSelect) {
        elements.playerEditorSelect.value = '';
      }
      if (elements.playerEditorJson) {
        elements.playerEditorJson.value = '';
      }
      setPlayerBuilderFeedback('区分が変更されました。選手を選ぶかテンプレートを作成してください。', 'info');
      (async () => {
        const role = elements.playerEditorRole?.value || 'batter';
        const players = await actions.fetchPlayersList(role);
        const select = elements.playerEditorSelect;
        if (select) {
          select.innerHTML = '';
          const ph = document.createElement('option');
          ph.value = '';
          ph.textContent = '選手を選択';
          select.appendChild(ph);
          const newOpt = document.createElement('option');
          newOpt.value = '__new__';
          newOpt.textContent = '新規選手を作成';
          select.appendChild(newOpt);
          players.forEach((name) => {
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = name;
            select.appendChild(opt);
          });
        }
      })();
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
        if (elements.playerEditorJson) elements.playerEditorJson.value = '';
        setPlayerBuilderFeedback('編集する選手を選択してください。', 'info');
        return;
      }
      setPlayerBuilderFeedback('選手データを読み込み中...', 'info');
      const player = await actions.fetchPlayerDefinition(selectValue);
      if (player && elements.playerEditorJson) {
        elements.playerEditorJson.value = JSON.stringify(player, null, 2);
        setPlayerBuilderFeedback('選手データを読み込みました。', 'info');
      } else {
        setPlayerBuilderFeedback('選手データの読み込みに失敗しました。', 'danger');
      }
    });
  }

  if (elements.playerEditorJson) {
    elements.playerEditorJson.addEventListener('input', () => {
      // reserved for dirty flag if needed
    });
  }

  if (elements.playerBuilderSave) {
    elements.playerBuilderSave.addEventListener('click', async () => {
      if (!elements.playerEditorJson) return;
      const raw = elements.playerEditorJson.value;
      let parsed;
      try {
        parsed = JSON.parse(raw);
      } catch (error) {
        const message = error instanceof Error ? error.message : 'JSONを解析できません。';
        setPlayerBuilderFeedback(`JSONの構文エラー: ${message}`, 'danger');
        return;
      }
      const role = elements.playerEditorRole?.value || 'batter';
      elements.playerBuilderSave.disabled = true;
      try {
        const savedName = await actions.handlePlayerSave(parsed, role);
        if (savedName) {
          if (elements.playerEditorSelect) {
            elements.playerEditorSelect.value = savedName;
          }
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
