// ddoc serve · GUI · Round 15 (+ Round 16 viz / i18n)
//
// Single-file JS that drives the 6 form builders, validation, CLI
// command preview, and result rendering against the REST endpoints
// shipped in Round 14. No frameworks; no build step.

(() => {
  'use strict';

  // ── Round-16 i18n ───────────────────────────────────────────────
  // Tiny string table — `?lang=ko` query (or localStorage cache)
  // switches headings/tab labels to Korean. Body labels (CLI option
  // names) stay verbatim because they map 1-to-1 to ddoc CLI flags.
  const I18N = {
    en: {
      'tab.drift':     'Analyze drift',
      'tab.eda':       'Analyze EDA',
      'tab.examples':  'Examples',
      'tab.report':    'Report',
      'tab.export':    'Export',
      'tab.fetch':     'Fetch',
      'tab.recipe':    'Recipe',
      'cli.title':     'Generated CLI command',
      'cli.copy':      'Copy',
      'cli.copied':    'copied',
      'submit':        'Submit',
      'submit.fix':    'fix the chips above',
      'result.title':  'Result',
      'result.download': 'Download envelope',
      'result.raw':    'Raw JSON',
      'auth.label':    'X-API-Key',
      'auth.clear':    'clear',
      'lang.toggle':   '한국어',
      'cli.hint':      'The form on the left maps 1-to-1 to ddoc CLI options. Submit calls the equivalent REST endpoint; this panel shows what you would run in a terminal for the same effect.',
    },
    ko: {
      'tab.drift':     '드리프트 분석',
      'tab.eda':       'EDA 분석',
      'tab.examples':  '예제 데이터',
      'tab.report':    '리포트',
      'tab.export':    '외부 발신',
      'tab.fetch':     '데이터 가져오기',
      'tab.recipe':    '레시피',
      'cli.title':     '생성된 CLI 명령어',
      'cli.copy':      '복사',
      'cli.copied':    '복사됨',
      'submit':        '실행',
      'submit.fix':    '위 오류 chip 을 먼저 해결해주세요',
      'result.title':  '결과',
      'result.download': 'envelope 다운로드',
      'result.raw':    '원본 JSON',
      'auth.label':    'X-API-Key',
      'auth.clear':    '지움',
      'lang.toggle':   'English',
      'cli.hint':      '좌측 form 은 ddoc CLI 옵션과 1대1 매핑됩니다. Submit 시 동일 동작의 REST endpoint 가 호출되며, 이 패널은 터미널에서 직접 입력할 동등한 명령어를 보여줍니다.',
    },
  };

  function detectLang() {
    const params = new URLSearchParams(location.search);
    const fromQuery = params.get('lang');
    if (fromQuery && I18N[fromQuery]) return fromQuery;
    const stored = localStorage.getItem('ddoc_serve_lang');
    if (stored && I18N[stored]) return stored;
    return 'en';
  }

  let LANG = detectLang();
  const t = (key) => (I18N[LANG] && I18N[LANG][key]) || I18N.en[key] || key;

  // ── State ────────────────────────────────────────────────────────
  const STATE = {
    health: null,
    detectors: null,        // /plugins/detectors response
    examples: null,         // /examples/scenarios response
    activeTab: 'drift',
    forms: {},              // tab → {state object}
    apiKey: localStorage.getItem('ddoc_serve_api_key') || '',
  };

  // ── DOM helpers ──────────────────────────────────────────────────
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'class') node.className = v;
      else if (k === 'html') node.innerHTML = v;
      else if (k.startsWith('on') && typeof v === 'function') {
        node.addEventListener(k.slice(2).toLowerCase(), v);
      } else if (v !== false && v !== null && v !== undefined) {
        node.setAttribute(k, v === true ? '' : String(v));
      }
    }
    for (const child of children.flat()) {
      if (child == null) continue;
      node.appendChild(typeof child === 'string' ? document.createTextNode(child) : child);
    }
    return node;
  }

  // ── HTTP helpers (with auth) ────────────────────────────────────
  function authHeaders(extra = {}) {
    const out = { ...extra };
    if (STATE.apiKey) out['X-API-Key'] = STATE.apiKey;
    return out;
  }

  async function api(method, path, body) {
    const init = {
      method,
      headers: authHeaders({ 'Content-Type': 'application/json' }),
    };
    if (body !== undefined) init.body = JSON.stringify(body);
    const r = await fetch(path, init);
    let json = null;
    try { json = await r.json(); } catch (_) { /* non-JSON */ }
    return { ok: r.ok, status: r.status, json };
  }

  // ── Bootstrap ────────────────────────────────────────────────────
  async function bootstrap() {
    const [health, detectors, examples] = await Promise.all([
      api('GET', '/healthz'),
      api('GET', '/plugins/detectors'),
      api('GET', '/examples/scenarios'),
    ]);
    STATE.health = health.json || {};
    STATE.detectors = detectors.json || { count: 0, registry: [] };
    STATE.examples = examples.json || { modalities: [], scenarios: [] };

    applyI18n();
    renderTopbar();
    renderForms();
    renderActiveTab();
  }

  function applyI18n() {
    document.documentElement.lang = LANG;
    // Tab labels.
    const tabKeys = { drift: 'tab.drift', eda: 'tab.eda', examples: 'tab.examples',
                      report: 'tab.report', export: 'tab.export', fetch: 'tab.fetch' };
    $$('#tabs .tab').forEach(b => {
      const k = tabKeys[b.dataset.tab];
      if (k) b.textContent = t(k);
    });
    // CLI panel header / button.
    const cliHeader = $('.cli-panel h3');
    if (cliHeader) cliHeader.textContent = t('cli.title');
    $('#copy-cli').textContent = t('cli.copy');
    const cliHintP = $('.cli-panel .hint');
    if (cliHintP) cliHintP.innerHTML = t('cli.hint');
    // Submit button label.
    $('#submit').textContent = t('submit');
    // Result heading + download.
    $('.result-panel h3').textContent = t('result.title');
    $('#result-download').textContent = t('result.download');
    const detailsSummary = $('.result-panel details summary');
    if (detailsSummary) detailsSummary.textContent = t('result.raw');
    // Auth label / clear button.
    const authLabel = $('#auth-block label');
    if (authLabel) {
      authLabel.firstChild.nodeValue = t('auth.label') + ' ';
    }
    $('#auth-clear').textContent = t('auth.clear');
    // Lang toggle button (rendered in topbar).
    const tog = $('#lang-toggle');
    if (tog) tog.textContent = t('lang.toggle');
  }

  // ── Topbar ───────────────────────────────────────────────────────
  function renderTopbar() {
    const h = STATE.health || {};
    $('#meta').textContent = `· v${h.ddoc_version || '?'} · ${h.plugin_count ?? '?'} plugins · auth: ${h.auth_enabled ? 'ON' : 'OFF'}`;

    // Language toggle button (Round 16 — switches en ↔ ko).
    if (!$('#lang-toggle')) {
      const btn = el('button', {
        id: 'lang-toggle', class: 'link', style: 'margin-left: 0.6em;',
        onclick: () => {
          LANG = LANG === 'ko' ? 'en' : 'ko';
          localStorage.setItem('ddoc_serve_lang', LANG);
          applyI18n();
          // Re-render forms so labels update too where applicable.
          renderForms();
          renderActiveTab();
        },
      }, t('lang.toggle'));
      $('#topbar').appendChild(btn);
    }

    const authBlock = $('#auth-block');
    if (h.auth_enabled) {
      authBlock.hidden = false;
      const input = $('#api-key');
      input.value = STATE.apiKey;
      input.addEventListener('change', () => {
        STATE.apiKey = input.value.trim();
        localStorage.setItem('ddoc_serve_api_key', STATE.apiKey);
      });
      $('#auth-clear').addEventListener('click', () => {
        STATE.apiKey = '';
        input.value = '';
        localStorage.removeItem('ddoc_serve_api_key');
      });
    } else {
      authBlock.hidden = true;
    }
  }

  // ── Tab switching ────────────────────────────────────────────────
  $$('#tabs .tab').forEach(btn => {
    btn.addEventListener('click', () => {
      STATE.activeTab = btn.dataset.tab;
      $$('#tabs .tab').forEach(b => b.classList.toggle('active', b === btn));
      $$('.builder').forEach(b => b.classList.toggle('active', b.dataset.builder === STATE.activeTab));
      renderActiveTab();
    });
  });

  // ── Form schemas (one per tab) ──────────────────────────────────
  // Each schema: array of { name, label, type, opts?, default?, help?,
  // required?, depends? }. depends is a function that returns true to
  // show this field given the current form state.

  function detectorOptions() {
    const supported = new Set(['default']);
    for (const d of (STATE.detectors?.registry || [])) {
      for (const s of d.supported || []) supported.add(s);
    }
    return Array.from(supported);
  }

  // Sample recipe text shown by default in the recipe tab. Declared
  // before SCHEMAS to avoid the temporal-dead-zone gotcha.
  const SAMPLE_RECIPE_YAML = [
    'name: timeseries-drift-smoke',
    'vars:',
    '  data_root: /tmp/ddoc_recipe_smoke',
    'steps:',
    '  - id: gen_pair',
    '    run: examples.generate',
    '    with:',
    '      modality: timeseries',
    '      out: "${vars.data_root}"',
    '      scenario: shifted',
    '  - id: drift',
    '    run: analyze.drift',
    '    with:',
    '      data_path_ref: "${vars.data_root}/ref"',
    '      data_path_cur: "${vars.data_root}/cur"',
    '      quiet: true',
    '  - id: report_md',
    '    run: report.render',
    '    when: "${steps.drift.json.overall_score} > 0.1"',
    '    with:',
    '      input: "${steps.drift.output}"',
    '      out: /tmp/ddoc_recipe_smoke/report.md',
    '      format: md',
    '',
  ].join('\n');

  const SCHEMAS = {
    drift: {
      title: 'Analyze drift',
      help: 'Compare two datasets (path mode or snapshot mode). Path mode is the orchestrator-friendly form — pass concrete directories.',
      fields: [
        { name: 'data_path_ref', label: 'data-path-ref', type: 'text', help: 'Baseline directory (path mode)' },
        { name: 'data_path_cur', label: 'data-path-cur', type: 'text', help: 'Current directory (path mode)' },
        { name: 'baseline', label: 'baseline', type: 'text', help: 'Baseline snapshot ID/alias (snapshot mode — leave paths blank)' },
        { name: 'current', label: 'current', type: 'text', help: 'Current snapshot ID/alias (snapshot mode)' },
        { name: 'detector', label: 'detector', type: 'select', opts: () => detectorOptions(), default: 'default' },
        { name: 'fusion', label: 'fusion', type: 'select', opts: ['none', 'weighted', 'max', 'joint'], default: 'none' },
        { name: 'fusion_weights', label: 'fusion-weights', type: 'text', help: 'e.g. image=0.6,text=0.4 (only used when fusion ≠ none)' },
        { name: 'with_embeddings', label: 'with-embeddings', type: 'check', default: false, help: 'Path mode: load CLIP inline (vision/text)' },
        { name: 'quiet', label: 'quiet', type: 'check', default: true },
        { name: 'use_streaming', label: 'Use streaming (SSE)', type: 'check', default: false, help: 'Show NDJSON progress events live (uses /analyze/drift/stream)' },
        { name: 'timeout_sec', label: 'timeout-sec', type: 'text', default: '600' },
      ],
    },
    eda: {
      title: 'Analyze EDA',
      help: 'Run exploratory data analysis on a snapshot, the workspace, or an arbitrary path.',
      fields: [
        { name: 'snapshot', label: 'snapshot', type: 'text', help: 'Snapshot ID or alias (omit for path/workspace mode)' },
        { name: 'data_path', label: 'data-path', type: 'text', help: 'Path-mode input directory' },
        { name: 'invalidate_cache', label: 'invalidate-cache', type: 'check', default: false },
        { name: 'save_snapshot', label: 'save-snapshot', type: 'check', default: false },
        { name: 'strict_hash', label: 'strict-hash', type: 'check', default: false },
        { name: 'quiet', label: 'quiet', type: 'check', default: true },
        { name: 'timeout_sec', label: 'timeout-sec', type: 'text', default: '600' },
      ],
    },
    examples: {
      title: 'Examples · generate toy data',
      help: 'Materialize a (ref, cur) toy dataset pair for the given modality.',
      fields: [
        { name: 'modality', label: 'modality', type: 'select', opts: () => STATE.examples?.modalities || [], required: true },
        { name: 'scenario', label: 'scenario', type: 'select', opts: () => STATE.examples?.scenarios || [], default: 'shifted' },
        { name: 'out', label: 'out', type: 'text', required: true, help: 'Output dir; will contain ref/ and cur/ subdirs.' },
      ],
    },
    report: {
      title: 'Render report',
      help: 'Render a drift / EDA envelope JSON to HTML / PDF / Markdown.',
      fields: [
        { name: 'input', label: 'input', type: 'text', required: true, help: 'Path to a drift / EDA envelope JSON.' },
        { name: 'out', label: 'out', type: 'text', required: true, help: 'Output report path; format inferred from suffix unless explicit.' },
        { name: 'format', label: 'format', type: 'select', opts: ['', 'html', 'pdf', 'md'], help: 'Empty = infer from --out suffix.' },
        { name: 'title', label: 'title', type: 'text' },
        { name: 'timeout_sec', label: 'timeout-sec', type: 'text', default: '120' },
      ],
    },
    export: {
      title: 'Export drift report',
      help: 'Ship a drift envelope to file:// or keti_veritas (or any plugin-registered target).',
      fields: [
        { name: 'input', label: 'input', type: 'text', required: true, help: 'Drift envelope JSON path.' },
        { name: 'target', label: 'target', type: 'select', opts: ['file', 'keti_veritas'], default: 'file', required: true },
        { name: 'config_json', label: 'config (JSON)', type: 'textarea', help: 'file: {"out_dir":"…"}; keti_veritas: {"base_url":"http://…","api_key":"…"}' },
        { name: 'timeout_sec', label: 'timeout-sec', type: 'text', default: '120' },
      ],
    },
    fetch: {
      title: 'Fetch · materialize remote data',
      help: 'Pull data from a URI (file/s3/gs/http) into a local directory; built-in file:// fallback. Plugins may register more schemes.',
      fields: [
        { name: 'source_uri', label: 'source-uri', type: 'text', required: true, help: 'file:///path, bare path, or s3://… (plugin required for non-file).' },
        { name: 'dest', label: 'dest', type: 'text', required: true },
        { name: 'symlink', label: 'symlink', type: 'check', default: false, help: 'file:// only; create symlink instead of copy.' },
        { name: 'config_json', label: 'config (JSON)', type: 'textarea', help: 'Adapter-specific JSON; e.g. {"region":"us-east-1"} for s3.' },
        { name: 'timeout_sec', label: 'timeout-sec', type: 'text', default: '120' },
      ],
    },
    recipe: {
      title: 'Recipe · multi-step workflow',
      help: 'Paste a recipe YAML or specify a server-side path. ddoc executes the steps in order and shows per-step progress as SSE events.',
      fields: [
        { name: 'mode', label: 'input mode', type: 'select', opts: ['inline', 'path'], default: 'inline' },
        { name: 'yaml', label: 'YAML', type: 'textarea', default: SAMPLE_RECIPE_YAML, help: 'Top-level: name, vars, steps[]. Each step needs id + run + with.' },
        { name: 'path', label: 'recipe path', type: 'text', help: 'Server-side path to a YAML file (use this instead of YAML when --inline isn\'t convenient).' },
        { name: 'dry_run', label: 'dry-run', type: 'check', default: false, help: 'Resolve substitutions and print argv but skip subprocess execution.' },
        { name: 'use_streaming', label: 'Use streaming (SSE)', type: 'check', default: true, help: 'Show per-step progress live (uses /recipe/run/stream).' },
        { name: 'validate_only', label: 'validate only', type: 'check', default: false, help: 'Call /recipe/validate instead of running.' },
      ],
    },
  };

  // ── Form rendering ───────────────────────────────────────────────
  function renderForms() {
    for (const [tab, schema] of Object.entries(SCHEMAS)) {
      const root = $(`.builder[data-builder="${tab}"]`);
      root.innerHTML = '';
      root.appendChild(el('h2', {}, schema.title));
      if (schema.help) root.appendChild(el('p', { class: 'help' }, schema.help));

      const formState = {};
      STATE.forms[tab] = formState;

      const checkRow = el('div', { class: 'row-checkbox' });
      let hasCheckboxes = false;

      for (const f of schema.fields) {
        const def = (typeof f.default === 'function') ? f.default() : f.default;
        if (def !== undefined) formState[f.name] = def;

        if (f.type === 'check') {
          hasCheckboxes = true;
          const input = el('input', {
            type: 'checkbox', id: `${tab}-${f.name}`,
            checked: !!def,
          });
          input.addEventListener('change', () => {
            formState[f.name] = input.checked;
            updateCli();
            validate();
          });
          checkRow.appendChild(el('label', { for: input.id }, input, ` ${f.label}`));
          if (f.help) {
            const help = el('span', { class: 'help-inline' }, ` (${f.help})`);
            checkRow.lastChild.appendChild(help);
          }
          continue;
        }

        const fieldRow = el('div', { class: 'field' });
        fieldRow.appendChild(el('label', { for: `${tab}-${f.name}` }, f.label + (f.required ? ' *' : '')));

        let input;
        if (f.type === 'select') {
          const opts = (typeof f.opts === 'function') ? f.opts() : f.opts;
          input = el('select', { id: `${tab}-${f.name}` },
            ...opts.map(o => el('option', { value: o, selected: o === def }, o || '(infer)')));
        } else if (f.type === 'textarea') {
          input = el('textarea', { id: `${tab}-${f.name}` });
          if (def) input.value = def;
        } else {
          input = el('input', { type: 'text', id: `${tab}-${f.name}` });
          if (def !== undefined) input.value = def;
        }
        input.addEventListener('input', () => {
          formState[f.name] = input.value;
          updateCli();
          validate();
        });
        input.addEventListener('change', () => {
          formState[f.name] = input.value;
          updateCli();
          validate();
        });
        fieldRow.appendChild(input);
        if (f.help) {
          fieldRow.appendChild(el('span', { class: 'help-inline' }, f.help));
        }
        root.appendChild(fieldRow);
      }
      if (hasCheckboxes) root.appendChild(checkRow);
    }
  }

  function renderActiveTab() {
    updateCli();
    validate();
    hideResult();
  }

  // ── Validation ───────────────────────────────────────────────────
  function validateCurrent() {
    const tab = STATE.activeTab;
    const schema = SCHEMAS[tab];
    const state = STATE.forms[tab] || {};
    const errors = [];

    for (const f of schema.fields) {
      if (f.required) {
        const v = state[f.name];
        if (v === undefined || v === '' || v === null) {
          errors.push(`${f.label} is required`);
        }
      }
    }

    if (tab === 'drift') {
      const hasPaths = state.data_path_ref && state.data_path_cur;
      const hasSnaps = state.baseline && state.current;
      if (!hasPaths && !hasSnaps) {
        errors.push('Provide either both data-path-ref + data-path-cur (path mode) or both baseline + current (snapshot mode)');
      }
      if (state.fusion && state.fusion !== 'none' && state.fusion !== 'weighted' && state.fusion_weights) {
        errors.push('fusion-weights only applies when fusion=weighted');
      }
      const det = state.detector;
      if (det && det !== 'default') {
        const allow = detectorOptions();
        if (!allow.includes(det)) errors.push(`detector ${det} is not in /plugins/detectors`);
      }
    }
    if (tab === 'eda' && !state.snapshot && !state.data_path) {
      // Not an error — EDA's workspace mode (no snapshot, no path) is
      // legal — but warn if user clears both expecting a different mode.
    }
    if ((tab === 'export' || tab === 'fetch') && state.config_json) {
      try {
        const parsed = JSON.parse(state.config_json);
        if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
          errors.push('config must be a JSON object');
        }
      } catch (_) {
        errors.push('config: invalid JSON');
      }
    }
    if (state.timeout_sec !== undefined && state.timeout_sec !== '') {
      const t = Number(state.timeout_sec);
      if (!Number.isFinite(t) || t < 0) errors.push('timeout-sec must be a non-negative number');
    }

    return errors;
  }

  function validate() {
    const errors = validateCurrent();
    const errorsBox = $('#errors');
    errorsBox.innerHTML = '';
    for (const e of errors) {
      errorsBox.appendChild(el('span', { class: 'error-chip' }, e));
    }
    $('#submit').disabled = errors.length > 0;
    $('#submit-hint').textContent = errors.length ? 'fix the chips above' : '';
  }

  // ── argv builders + endpoint mappers ─────────────────────────────
  function buildArgv(tab, state) {
    const args = [];
    const pushOpt = (flag, val) => {
      if (val === undefined || val === null || val === '') return;
      args.push(flag, String(val));
    };
    const pushFlag = (flag, on) => { if (on) args.push(flag); };

    if (tab === 'drift') {
      args.push('analyze', 'drift');
      if (state.baseline) args.push(state.baseline);
      if (state.current) args.push(state.current);
      pushOpt('--data-path-ref', state.data_path_ref);
      pushOpt('--data-path-cur', state.data_path_cur);
      if (state.detector && state.detector !== 'default') pushOpt('--detector', state.detector);
      if (state.fusion && state.fusion !== 'none') pushOpt('--fusion', state.fusion);
      pushOpt('--fusion-weights', state.fusion_weights);
      pushFlag('--with-embeddings', state.with_embeddings);
      args.push('--json');
      pushFlag('--quiet', state.quiet);
      if (state.use_streaming) args.push('--ndjson-progress');
    } else if (tab === 'eda') {
      args.push('analyze', 'eda');
      if (state.snapshot) args.push(state.snapshot);
      pushOpt('--data-path', state.data_path);
      pushFlag('--invalidate-cache', state.invalidate_cache);
      pushFlag('--save-snapshot', state.save_snapshot);
      pushFlag('--strict-hash', state.strict_hash);
      args.push('--json');
      pushFlag('--quiet', state.quiet);
    } else if (tab === 'examples') {
      args.push('examples', 'generate', state.modality || '<modality>');
      pushOpt('--out', state.out);
      pushOpt('--scenario', state.scenario);
    } else if (tab === 'report') {
      args.push('report', 'render');
      pushOpt('-i', state.input);
      pushOpt('-o', state.out);
      pushOpt('--format', state.format);
      pushOpt('--title', state.title);
      args.push('--json');
    } else if (tab === 'export') {
      args.push('export', 'drift-report');
      if (state.input) args.push(state.input);
      pushOpt('--to', state.target);
      if (state.config_json) pushOpt('--config', state.config_json);
      args.push('--json');
    } else if (tab === 'fetch') {
      args.push('fetch');
      if (state.source_uri) args.push(state.source_uri);
      pushOpt('--dest', state.dest);
      pushFlag('--symlink', state.symlink);
      if (state.config_json) pushOpt('--config', state.config_json);
      args.push('--json');
    } else if (tab === 'recipe') {
      // CLI hint mirrors `ddoc recipe {validate,run} <path>` form.
      // For inline mode we suggest writing to a file first; the GUI's
      // actual submit hits the REST endpoint with the YAML body.
      const verb = state.validate_only ? 'validate' : 'run';
      args.push('recipe', verb);
      if (state.mode === 'path' && state.path) {
        args.push(state.path);
      } else if (state.mode === 'inline') {
        args.push('<recipe.yaml>');
      }
      if (verb === 'run') {
        if (state.dry_run) args.push('--dry-run');
        args.push('--json');
      } else {
        args.push('--json');
      }
    }
    return args;
  }

  function quoteArg(a) {
    if (/^[A-Za-z0-9_./:=,-]+$/.test(a)) return a;
    return `'${String(a).replace(/'/g, `'\\''`)}'`;
  }

  function updateCli() {
    const tab = STATE.activeTab;
    const args = buildArgv(tab, STATE.forms[tab] || {});
    $('#cli-hint').textContent = ['ddoc', ...args.map(quoteArg)].join(' ');
  }

  // ── Endpoint dispatch ───────────────────────────────────────────
  function buildRequest(tab, state) {
    const body = { ...state };
    if (tab === 'drift') {
      const useStream = !!body.use_streaming;
      delete body.use_streaming;
      body.timeout_sec = Number(body.timeout_sec) || 600;
      return { path: useStream ? '/analyze/drift/stream' : '/analyze/drift', body, sse: useStream };
    }
    if (tab === 'eda') {
      body.timeout_sec = Number(body.timeout_sec) || 600;
      return { path: '/analyze/eda', body, sse: false };
    }
    if (tab === 'examples') {
      return { path: '/examples/generate', body: {
        modality: body.modality, scenario: body.scenario, out: body.out,
      }, sse: false };
    }
    if (tab === 'report') {
      const out = {
        input: body.input, out: body.out,
        format: body.format || null, title: body.title || null,
        timeout_sec: Number(body.timeout_sec) || 120,
      };
      return { path: '/report/render', body: out, sse: false };
    }
    if (tab === 'export') {
      let cfg = null;
      if (body.config_json) {
        try { cfg = JSON.parse(body.config_json); } catch (_) { cfg = null; }
      }
      return {
        path: '/export/drift-report',
        body: {
          input: body.input, target: body.target,
          config: cfg, timeout_sec: Number(body.timeout_sec) || 120,
        },
        sse: false,
      };
    }
    if (tab === 'fetch') {
      let cfg = null;
      if (body.config_json) {
        try { cfg = JSON.parse(body.config_json); } catch (_) { cfg = null; }
      }
      return {
        path: '/fetch',
        body: {
          source_uri: body.source_uri, dest: body.dest,
          symlink: !!body.symlink, config: cfg,
          timeout_sec: Number(body.timeout_sec) || 120,
        },
        sse: false,
      };
    }
    if (tab === 'recipe') {
      const recipeBody = {};
      if (body.mode === 'inline') recipeBody.yaml = body.yaml;
      else if (body.mode === 'path' && body.path) recipeBody.path = body.path;
      recipeBody.dry_run = !!body.dry_run;
      if (body.validate_only) {
        return { path: '/recipe/validate', body: recipeBody, sse: false };
      }
      const useStream = !!body.use_streaming && !body.dry_run;
      return {
        path: useStream ? '/recipe/run/stream' : '/recipe/run',
        body: recipeBody,
        sse: useStream,
      };
    }
  }

  // ── Submit ──────────────────────────────────────────────────────
  $('#submit').addEventListener('click', async () => {
    const tab = STATE.activeTab;
    if (validateCurrent().length) return;
    const req = buildRequest(tab, STATE.forms[tab] || {});
    showResult({ status: 'pending', body: { status: 'submitting…' } });
    if (req.sse) {
      await streamRequest(req.path, req.body);
    } else {
      const r = await api('POST', req.path, req.body);
      renderResultBody(r);
    }
  });

  function showResult(initial) {
    $('#result-panel').hidden = false;
    $('#result-progress').hidden = true;
    $('#result-progress').innerHTML = '';
    $('#result-status').textContent = initial?.status || '';
    $('#result-status').className = '';
    $('#result-body').textContent = JSON.stringify(initial?.body ?? {}, null, 2);
    $('#result-download').hidden = true;
    $('#result-render').hidden = true;
  }
  function hideResult() { $('#result-panel').hidden = true; }

  function renderResultBody(r) {
    const ok = r.ok && r.json && r.json.status !== 'error';
    const statusEl = $('#result-status');
    statusEl.textContent = `HTTP ${r.status} ${ok ? 'OK' : (r.json?.error_code || 'error')}`;
    statusEl.className = ok ? 'ok' : 'err';
    $('#result-body').textContent = JSON.stringify(r.json ?? {}, null, 2);
    renderViz(r.json, ok);

    // Drift / EDA envelope: surface convenience actions.
    if (ok && r.json && (r.json.modality || r.json.modalities)) {
      const dl = $('#result-download');
      dl.hidden = false;
      dl.onclick = () => {
        const blob = new Blob([JSON.stringify(r.json, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = `${r.json.modality || 'envelope'}.json`;
        a.click();
        URL.revokeObjectURL(url);
      };
    } else {
      $('#result-download').hidden = true;
    }
  }

  // ── Round-16 result viz ─────────────────────────────────────────
  // Renders score / status / per-attribute bars from the envelope. Pure
  // CSS (no chart libs). Falls back to "raw only" when nothing to viz.
  function renderViz(envelope, ok) {
    const viz = $('#result-viz');
    viz.innerHTML = '';
    if (!envelope || typeof envelope !== 'object') { viz.hidden = true; return; }

    // Multi-modal stack (drift): each modality gets its own card.
    if (envelope.modalities && typeof envelope.modalities === 'object') {
      const grid = el('div', { class: 'modality-grid' });
      for (const [name, sub] of Object.entries(envelope.modalities)) {
        grid.appendChild(modalityCard(name, sub));
      }
      viz.appendChild(grid);
      if (typeof envelope.fused_score === 'number') {
        viz.appendChild(fusionCard(envelope));
      }
      viz.hidden = false;
      return;
    }

    // Single-modality drift / EDA envelope (modality + overall_score
    // or files_analyzed/series_analyzed/...).
    if (envelope.modality) {
      viz.appendChild(modalityCard(envelope.modality, envelope));
      viz.hidden = false;
      return;
    }

    // Recipe envelope: list of step results.
    if (Array.isArray(envelope.steps) && envelope.steps.length) {
      viz.appendChild(recipeCard(envelope));
      viz.hidden = false;
      return;
    }

    // Status-only success (e.g. /report/render returns {status: success,
    // format, output_path, size_bytes}).
    if (ok && envelope.status === 'success' && envelope.output_path) {
      viz.appendChild(statusCard(envelope));
      viz.hidden = false;
      return;
    }

    viz.hidden = true;
  }

  function classifyScore(score) {
    if (score == null || !isFinite(score)) return 'normal';
    if (score < 0.15) return 'normal';
    if (score < 0.25) return 'warning';
    return 'critical';
  }

  function modalityCard(name, sub) {
    const card = el('div', { class: 'viz-card' });
    card.appendChild(el('h4', {}, name));

    if (sub.status === 'error') {
      card.appendChild(el('div', { class: 'big critical' }, 'error'));
      card.appendChild(el('div', { class: 'meta' },
        el('span', {}, el('b', {}, sub.error_code || 'error_code'), ': ', String(sub.message || ''))));
      return card;
    }

    const score = sub.overall_score;
    if (typeof score === 'number') {
      const cls = sub.status?.toLowerCase?.() || classifyScore(score);
      card.appendChild(el('div', { class: `big ${cls}` }, score.toFixed(4)));
      const metaParts = [];
      if (sub.status) metaParts.push([el('span', {}, el('b', {}, 'status'), ` ${sub.status}`)]);
      if (sub.embedding_drift_detector) metaParts.push([el('span', {}, el('b', {}, 'detector'), ` ${sub.embedding_drift_detector}`)]);
      if (typeof sub.files_added === 'number') metaParts.push([el('span', {}, el('b', {}, 'files'), ` +${sub.files_added}/-${sub.files_removed}/=${sub.files_common}`)]);
      if (metaParts.length) card.appendChild(el('div', { class: 'meta' }, ...metaParts.flat()));
    } else if (sub.summary) {
      // EDA envelope shape — show summary fields.
      const meta = el('div', { class: 'meta' });
      for (const [k, v] of Object.entries(sub.summary)) {
        meta.appendChild(el('span', {}, el('b', {}, k), ` ${formatVal(v)}`));
      }
      card.appendChild(meta);
    }

    // Attribute drifts → bars.
    if (sub.attribute_drifts && typeof sub.attribute_drifts === 'object') {
      const max = Math.max(0.001, ...Object.values(sub.attribute_drifts).map(Math.abs));
      const block = el('div', {});
      for (const [k, v] of Object.entries(sub.attribute_drifts)) {
        const pct = Math.min(100, Math.abs(v) / max * 100);
        const cls = classifyScore(Math.abs(v));
        block.appendChild(el('div', { class: 'bar-row' },
          el('span', { class: 'lbl' }, k),
          el('div', { class: `bar ${cls === 'critical' ? 'crit' : cls === 'warning' ? 'warn' : 'ok'}` },
            el('span', { style: `width:${pct}%` })),
          el('span', { class: 'val' }, formatVal(v)),
        ));
      }
      card.appendChild(block);
    }

    // Round 18 — embedding ensemble breakdown.
    if (sub.embedding_drift_detailed && typeof sub.embedding_drift_detailed === 'object') {
      card.appendChild(embeddingEnsembleBlock(sub.embedding_drift_detailed));
    }

    return card;
  }

  // Render the per-component breakdown of an ensemble embedding-drift
  // result (Round-12 vision, Round-12 text). Two parts:
  //   1. Stacked contribution bar — width = weight × normalized score
  //   2. Per-metric row showing weight, normalized score, raw value
  function embeddingEnsembleBlock(detail) {
    const root = el('div', { class: 'ensemble-block' });
    root.appendChild(el('h5', {}, 'Embedding ensemble'));

    const weights = detail.weights || {};
    const normalized = detail.normalized_scores || {};
    const ensemble = +detail.ensemble_score || 0;
    const componentNames = Object.keys(weights);

    if (componentNames.length === 0) return root;

    // Stacked contribution bar.
    const stackedLabel = el('div', { class: 'ensemble-stacked-label' },
      `weighted contributions → ${ensemble.toFixed(4)}`);
    const stackedBar = el('div', { class: 'ensemble-stacked' });
    const palette = ['#2563eb', '#7c3aed', '#0891b2', '#059669', '#d97706', '#9d174d'];
    componentNames.forEach((name, i) => {
      const w = +weights[name] || 0;
      const n = +normalized[name] || 0;
      const contrib = Math.max(0, Math.min(1, w * n));
      if (contrib <= 0) return;
      stackedBar.appendChild(el('div', {
        class: 'ensemble-stacked-seg',
        style: `width:${(contrib * 100).toFixed(2)}%; background:${palette[i % palette.length]}`,
        title: `${name}: weight ${w.toFixed(2)} × normalized ${n.toFixed(3)} = ${contrib.toFixed(3)}`,
      }));
    });
    root.appendChild(stackedLabel);
    root.appendChild(stackedBar);

    // Per-metric breakdown table.
    const table = el('div', { class: 'ensemble-table' });
    table.appendChild(el('div', { class: 'ensemble-thead' },
      el('span', { class: 'lbl' }, 'metric'),
      el('span', {}, 'weight'),
      el('span', {}, 'normalized'),
      el('span', {}, 'raw'),
    ));
    componentNames.forEach((name, i) => {
      const w = +weights[name] || 0;
      const n = +normalized[name] || 0;
      const raw = detail[name];
      const dot = el('span', { class: 'ensemble-dot', style: `background:${palette[i % palette.length]}` });
      table.appendChild(el('div', { class: 'ensemble-trow' },
        el('span', { class: 'lbl' }, dot, name),
        el('span', { class: 'val' }, w.toFixed(2)),
        el('span', { class: 'val' }, n.toFixed(4)),
        el('span', { class: 'val' }, formatVal(raw)),
      ));
    });
    root.appendChild(table);

    return root;
  }

  function fusionCard(env) {
    const card = el('div', { class: 'viz-card' });
    card.appendChild(el('h4', {}, `Fusion (${env.fusion_strategy})`));
    const cls = classifyScore(env.fused_score);
    card.appendChild(el('div', { class: `big ${cls}` }, env.fused_score.toFixed(4)));
    const meta = el('div', { class: 'meta' });
    if (env.fusion_weights) {
      const w = Object.entries(env.fusion_weights).map(([k, v]) => `${k}=${(+v).toFixed(2)}`).join(' · ');
      meta.appendChild(el('span', {}, el('b', {}, 'weights'), ` ${w}`));
    }
    if (env.fusion_warnings && env.fusion_warnings.length) {
      meta.appendChild(el('span', {}, el('b', {}, '⚠'), ` ${env.fusion_warnings.length} warning(s)`));
    }
    card.appendChild(meta);
    return card;
  }

  function recipeCard(env) {
    const card = el('div', { class: 'viz-card' });
    card.appendChild(el('h4', {}, `Recipe · ${env.recipe || ''}`));
    const cls = env.status === 'success' ? 'normal' : 'critical';
    card.appendChild(el('div', { class: `big ${cls}` }, env.status));
    const meta = el('div', { class: 'meta' });
    meta.appendChild(el('span', {}, el('b', {}, 'steps'), ` ${env.steps.length}`));
    const ok = env.steps.filter(s => !s.skipped && s.returncode === 0).length;
    meta.appendChild(el('span', {}, el('b', {}, 'ok'), ` ${ok}/${env.steps.length}`));
    card.appendChild(meta);
    // Compact per-step list.
    const list = el('div', { class: 'meta', style: 'flex-direction:column;align-items:flex-start;gap:0.2em' });
    for (const s of env.steps) {
      const status = s.skipped ? 'dry' : (s.returncode === 0 ? 'ok' : 'fail');
      list.appendChild(el('span', {}, `[${status}] ${s.id} (${s.run})${s.elapsed_ms ? ` — ${s.elapsed_ms} ms` : ''}`));
    }
    card.appendChild(list);
    return card;
  }

  function statusCard(env) {
    const card = el('div', { class: 'viz-card' });
    card.appendChild(el('h4', {}, env.format ? `${env.format.toUpperCase()} report` : 'Result'));
    card.appendChild(el('div', { class: 'big normal' }, '✓'));
    const meta = el('div', { class: 'meta' });
    if (env.output_path) meta.appendChild(el('span', {}, el('b', {}, 'path'), ` ${env.output_path}`));
    if (env.size_bytes != null) meta.appendChild(el('span', {}, el('b', {}, 'size'), ` ${env.size_bytes} B`));
    if (env.target) meta.appendChild(el('span', {}, el('b', {}, 'target'), ` ${env.target}`));
    if (env.http_status) meta.appendChild(el('span', {}, el('b', {}, 'http'), ` ${env.http_status}`));
    card.appendChild(meta);
    return card;
  }

  function formatVal(v) {
    if (typeof v === 'number') {
      const abs = Math.abs(v);
      return abs >= 1000 || (abs > 0 && abs < 0.001) ? v.toExponential(3) : v.toFixed(4);
    }
    return String(v);
  }

  async function streamRequest(path, body) {
    $('#result-progress').hidden = false;
    $('#result-progress').innerHTML = '';
    const addChip = (text, cls = 'progress-chip') => {
      $('#result-progress').appendChild(el('span', { class: cls }, text));
    };
    let resp;
    try {
      resp = await fetch(path, {
        method: 'POST',
        headers: authHeaders({ 'Content-Type': 'application/json', 'Accept': 'text/event-stream' }),
        body: JSON.stringify(body),
      });
    } catch (e) {
      addChip(`network error: ${e.message}`, 'progress-chip error');
      return;
    }
    if (!resp.ok) {
      let json = null;
      try { json = await resp.json(); } catch (_) {}
      renderResultBody({ ok: false, status: resp.status, json });
      return;
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buf = '';
    let lastResult = null;
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf('\n\n')) >= 0) {
        const block = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const lines = block.split('\n');
        let event = 'message', data = '';
        for (const line of lines) {
          if (line.startsWith('event: ')) event = line.slice(7).trim();
          else if (line.startsWith('data: ')) data += line.slice(6);
        }
        if (!data) continue;
        let parsed;
        try { parsed = JSON.parse(data); } catch (_) { continue; }
        if (event === 'progress') {
          // analyze.drift/stream events: {progress, stage, message}
          // recipe/run/stream events:    {id, run, returncode, skipped, ...}
          if (typeof parsed.progress === 'number') {
            addChip(`${(parsed.progress * 100).toFixed(0)}% · ${parsed.stage}`);
          } else if (parsed.id) {
            const status = parsed.skipped
              ? `skip${parsed.skipped_reason ? '·' + parsed.skipped_reason : ''}`
              : (parsed.returncode === 0 ? 'ok' : `rc=${parsed.returncode}`);
            addChip(`${parsed.id} (${parsed.run}) — ${status}`);
          } else {
            addChip(JSON.stringify(parsed).slice(0, 80));
          }
        } else if (event === 'result') {
          lastResult = parsed;
          addChip(`100% · complete`);
        } else if (event === 'error') {
          lastResult = parsed;
          addChip(`error: ${parsed.error_code || parsed.error_type || 'unknown'}`, 'progress-chip error');
        }
      }
    }
    renderResultBody({ ok: lastResult && lastResult.status !== 'error', status: 200, json: lastResult });
  }

  // ── Copy CLI command ─────────────────────────────────────────────
  $('#copy-cli').addEventListener('click', async () => {
    const text = $('#cli-hint').textContent;
    try {
      await navigator.clipboard.writeText(text);
      const btn = $('#copy-cli');
      const orig = btn.textContent;
      btn.textContent = 'copied';
      setTimeout(() => (btn.textContent = orig), 1200);
    } catch (e) {
      alert('Clipboard API unavailable; select the text manually.');
    }
  });

  // ── Go ───────────────────────────────────────────────────────────
  bootstrap().catch(err => {
    document.body.innerHTML = `<pre style="padding:2em;color:#c4302b">bootstrap failed: ${err}</pre>`;
  });

})();
