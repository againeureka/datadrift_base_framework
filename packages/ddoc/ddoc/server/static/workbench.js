/* ddoc serve · workbench — choose two datasets, see the diagnosis.
 *
 * The front page of the GUI. Data comes in by dropping files or folders,
 * picking them, or browsing the server's data folders; each side shows
 * what it holds before anything is judged; the result opens as a verdict,
 * the pattern cards and one row per column, with the full diagnostic
 * report underneath. The forms that map 1-to-1 to CLI commands live in
 * app.js, under "Advanced tools".
 */
(function () {
  'use strict';

  const root = document.getElementById('workbench');
  if (!root || !window.ddocApp) return;
  const { api, authHeaders } = window.ddocApp;
  const lang = () => window.ddocApp.lang();

  // ── Text ─────────────────────────────────────────────────────────
  const TEXT = {
    en: {
      'view.workbench': 'Diagnose', 'view.advanced': 'Advanced tools',
      'intake.title': 'Has the data changed?',
      'intake.lead': 'Put in the data you trust and the data you want to check. ddoc says whether it drifted, what changed and what to look at first.',
      'side.ref': 'Baseline data', 'side.cur': 'Current data',
      'side.ref.tag': 'Baseline', 'side.cur.tag': 'Current',
      'side.ref.sub': 'The side you trust — training data, last month',
      'side.cur.sub': 'The side to check — what came in today',
      'drop.title': 'Drop files or a folder here',
      'drop.kinds': 'CSV · image folders · text · audio',
      'drop.files': 'Choose files', 'drop.folder': 'Choose a folder', 'drop.browse': 'Browse the server',
      'drop.type': 'Type a path instead', 'drop.path': 'Path on the server', 'drop.use': 'Use',
      'drop.uploading': 'Uploading {done} of {total} ({size})', 'drop.reading': 'Reading…',
      'card.change': 'Change', 'card.rows': 'rows', 'card.columns': 'columns', 'card.files': 'files',
      'card.period': 'Period', 'card.images': 'images', 'card.items': 'items', 'card.more': '+{n} more',
      'card.src.upload': 'Uploaded from this computer', 'card.src.case': 'Example data',
      'card.missing': 'Nothing found at this path.', 'card.unknown': 'No CSV, image, text or audio files here.',
      'kind.csv': 'CSV', 'kind.vision': 'Images', 'kind.text': 'Text', 'kind.audio': 'Audio',
      'kind.categorical': 'Distributions', 'kind.unknown': 'Unknown',
      'type.numeric': 'number', 'type.categorical': 'category', 'type.datetime': 'date',
      'pre.checking': 'Checking the pair…',
      'pre.ok': 'Ready to diagnose', 'pre.compared': '{n} columns will be compared',
      'pre.time': 'date column {cols} gives the period only',
      'pre.only_ref': 'only in the baseline: {cols}', 'pre.only_cur': 'only in the current data: {cols}',
      'pre.blind': 'This pair cannot be judged',
      'run': 'Diagnose', 'run.busy': 'Diagnosing… {s}s', 'run.need': 'Choose both sides first',
      'opt.title': 'Settings', 'opt.issuer': 'Issuer on the report', 'opt.sample': 'Raw rows shown in the report (0 = none)',
      'opt.embed': 'Compare embeddings too (images and text; slower)',
      'opt.threshold': 'Threshold, overall', 'opt.attr': 'Threshold, one column',
      'cases.title': 'Or start from an example',
      'cases.lead': 'Small stories with a known answer. One click builds the data and diagnoses it.',
      'cases.run': 'Diagnose this case', 'cases.expect': 'expected',
      'recent.title': 'Recent', 'recent.clear': 'Clear',
      'res.new': 'New diagnosis', 'res.severity': 'severity', 'res.score': 'Score', 'res.threshold': 'threshold',
      'res.what': 'What changed', 'res.patterns': 'Pattern reading', 'res.columns': 'By column',
      'res.match': 'match', 'res.near': 'near', 'res.first': 'Do first',
      'res.legend.ref': 'baseline', 'res.legend.cur': 'current',
      'res.checks': 'Check first', 'res.report': 'Diagnostic report',
      'res.open': 'Open in a new tab', 'res.html': 'Save HTML', 'res.pdf': 'Save PDF', 'res.md': 'Save Markdown',
      'res.cli': 'Copy the command', 'res.copied': 'Copied', 'res.saving': 'Preparing…',
      'res.no_patterns': 'No written-up pattern matches these measurements.',
      'res.more_columns': '{n} more columns are in the report.',
      'res.case': 'Example', 'res.case.look': 'What to notice',
      'status.pass': 'pass', 'status.caution': 'caution', 'status.fail': 'fail',
      'fs.title': 'Browse the server', 'fs.roots': 'Data folders', 'fs.up': 'Up',
      'fs.use_folder': 'Use this folder', 'fs.cancel': 'Cancel', 'fs.empty': 'This folder is empty.',
      'fs.hint': 'Pick a CSV file, or open a folder and use it as a whole.',
      'fs.truncated': 'Only the first {n} entries are shown.',
      'err.generic': 'Something went wrong: {msg}',
    },
    ko: {
      'view.workbench': '진단', 'view.advanced': '고급 도구',
      'intake.title': '데이터가 달라졌나요?',
      'intake.lead': '믿는 데이터와 확인하려는 데이터를 넣으세요. 달라졌는지, 무엇이 달라졌는지, 무엇부터 볼지 알려드립니다.',
      'side.ref': '기준 데이터', 'side.cur': '현재 데이터',
      'side.ref.tag': '기준', 'side.cur.tag': '현재',
      'side.ref.sub': '믿는 쪽 — 학습에 쓴 데이터, 지난달',
      'side.cur.sub': '확인할 쪽 — 오늘 들어온 데이터',
      'drop.title': '파일이나 폴더를 여기에 끌어다 놓으세요',
      'drop.kinds': 'CSV · 이미지 폴더 · 텍스트 · 오디오',
      'drop.files': '파일 선택', 'drop.folder': '폴더 선택', 'drop.browse': '서버에서 찾기',
      'drop.type': '경로 직접 입력', 'drop.path': '서버의 경로', 'drop.use': '사용',
      'drop.uploading': '올리는 중 {done} / {total} ({size})', 'drop.reading': '읽는 중…',
      'card.change': '바꾸기', 'card.rows': '행', 'card.columns': '열', 'card.files': '파일',
      'card.period': '기간', 'card.images': '이미지', 'card.items': '항목', 'card.more': '외 {n}개',
      'card.src.upload': '이 컴퓨터에서 올린 데이터', 'card.src.case': '예제 데이터',
      'card.missing': '이 경로에서 아무것도 찾지 못했습니다.', 'card.unknown': 'CSV·이미지·텍스트·오디오 파일이 없습니다.',
      'kind.csv': 'CSV', 'kind.vision': '이미지', 'kind.text': '텍스트', 'kind.audio': '오디오',
      'kind.categorical': '분포', 'kind.unknown': '알 수 없음',
      'type.numeric': '숫자', 'type.categorical': '범주', 'type.datetime': '날짜',
      'pre.checking': '두 데이터를 확인하는 중…',
      'pre.ok': '진단할 수 있습니다', 'pre.compared': '열 {n}개를 비교합니다',
      'pre.time': '날짜 열 {cols} 은 기간 표시에만 씁니다',
      'pre.only_ref': '기준에만 있는 열: {cols}', 'pre.only_cur': '현재에만 있는 열: {cols}',
      'pre.blind': '이 두 데이터는 판정할 수 없습니다',
      'run': '진단하기', 'run.busy': '진단하는 중… {s}초', 'run.need': '두 데이터를 먼저 넣으세요',
      'opt.title': '세부 설정', 'opt.issuer': '진단서의 발행처', 'opt.sample': '진단서에 넣을 원본 행 수 (0 = 넣지 않음)',
      'opt.embed': '임베딩도 비교 (이미지·텍스트, 느림)',
      'opt.threshold': '전체 기준', 'opt.attr': '단일 열 기준',
      'cases.title': '또는 예제로 시작하기',
      'cases.lead': '정답이 있는 작은 사례입니다. 누르면 데이터를 만들어 바로 진단합니다.',
      'cases.run': '이 사례 진단하기', 'cases.expect': '기대 판정',
      'recent.title': '최근 진단', 'recent.clear': '지우기',
      'res.new': '새 진단', 'res.severity': '심각도', 'res.score': '점수', 'res.threshold': '기준',
      'res.what': '무엇이 달라졌나', 'res.patterns': '유형 판독', 'res.columns': '열별 결과',
      'res.match': '해당', 'res.near': '가까움', 'res.first': '먼저 할 일',
      'res.legend.ref': '기준', 'res.legend.cur': '현재',
      'res.checks': '먼저 확인할 것', 'res.report': '진단서',
      'res.open': '새 탭에서 열기', 'res.html': 'HTML 저장', 'res.pdf': 'PDF 저장', 'res.md': 'Markdown 저장',
      'res.cli': '명령 복사', 'res.copied': '복사됨', 'res.saving': '만드는 중…',
      'res.no_patterns': '정리해 둔 유형 가운데 이 측정값에 해당하는 것이 없습니다.',
      'res.more_columns': '나머지 {n}개 열은 진단서에 있습니다.',
      'res.case': '예제', 'res.case.look': '눈여겨볼 점',
      'status.pass': '적합', 'status.caution': '주의', 'status.fail': '부적합',
      'fs.title': '서버에서 찾기', 'fs.roots': '데이터 폴더', 'fs.up': '위로',
      'fs.use_folder': '이 폴더 사용', 'fs.cancel': '취소', 'fs.empty': '빈 폴더입니다.',
      'fs.hint': 'CSV 파일을 고르거나, 폴더를 열어 폴더째로 사용하세요.',
      'fs.truncated': '처음 {n}개만 보여줍니다.',
      'err.generic': '문제가 생겼습니다: {msg}',
    },
  };
  const t = (key, vars) => {
    const s = (TEXT[lang()] && TEXT[lang()][key]) || TEXT.en[key] || key;
    return s.replace(/\{(\w+)\}/g, (_, k) => (vars && vars[k] != null ? vars[k] : ''));
  };

  // ── Small helpers ────────────────────────────────────────────────
  function h(tag, attrs, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === 'class') node.className = v;
      else if (k === 'html') node.innerHTML = v;            // server-built SVG only
      else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? '' : v);
    }
    for (const c of children.flat()) {
      if (c == null || c === false) continue;
      node.appendChild(typeof c === 'string' || typeof c === 'number' ? document.createTextNode(String(c)) : c);
    }
    return node;
  }
  const num = (n) => (n == null ? '—' : Number(n).toLocaleString(lang() === 'ko' ? 'ko-KR' : 'en-US'));
  const bytes = (n) => (n < 1024 ? n + ' B' : n < 1048576 ? (n / 1024).toFixed(0) + ' KB'
    : n < 1073741824 ? (n / 1048576).toFixed(1) + ' MB' : (n / 1073741824).toFixed(2) + ' GB');
  const baseName = (p) => String(p || '').replace(/[\\/]+$/, '').split(/[\\/]/).pop() || String(p || '');
  const ICONS = {
    upload: '<path d="M12 16V4m0 0L7 9m5-5 5 5M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>',
    folder: '<path d="M3 6a1 1 0 0 1 1-1h5l2 2h9a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1z"/>',
    table: '<rect x="3" y="4" width="18" height="16" rx="1.5"/><path d="M3 9h18M3 14h18M9 4v16"/>',
    image: '<rect x="3" y="4" width="18" height="16" rx="1.5"/><circle cx="8.5" cy="9.5" r="1.5"/><path d="m4 18 5-5 4 4 3-3 4 4"/>',
    file: '<path d="M6 3h8l4 4v14H6z"/><path d="M14 3v4h4"/>',
    arrow: '<path d="M4 12h16m0 0-6-6m6 6-6 6"/>',
    back: '<path d="M20 12H4m0 0 6-6m-6 6 6 6"/>',
  };
  const icon = (name, size) => h('span', {
    class: 'wb-icon', html: `<svg viewBox="0 0 24 24" width="${size || 18}" height="${size || 18}" fill="none" `
      + `stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name]}</svg>`,
  });
  const KIND_ICON = { csv: 'table', vision: 'image', image: 'image', dir: 'folder' };

  // ── State ────────────────────────────────────────────────────────
  const S = {
    sides: { ref: null, cur: null },   // { path, info, busy, error, typing }
    probe: null, probing: false,
    options: { issuer: '', sample_rows: 0, with_embeddings: false, threshold: 0.15, attr_threshold: 0.25 },
    running: false, started: 0, error: null,
    result: null, activeCase: null,
    cases: [],
    browser: null,                     // { side, data, loading, error }
  };
  const RECENT_KEY = 'ddoc_serve_recent';
  const recent = () => { try { return JSON.parse(localStorage.getItem(RECENT_KEY) || '[]'); } catch (_) { return []; } };

  // ── Getting data in ──────────────────────────────────────────────
  async function describeSide(side, path, source) {
    if (source !== 'case') S.activeCase = null;
    S.sides[side] = { path, busy: t('drop.reading') };
    S.probe = null;
    render();
    const r = await api('POST', '/describe', { path, lang: lang() });
    S.sides[side] = r.ok ? { path, info: r.json, source } : { path, error: errorText(r) };
    render();
    refreshProbe();
  }

  async function refreshProbe() {
    const { ref, cur } = S.sides;
    S.probe = null;
    if (!(ref && cur && ref.info && cur.info && ref.info.exists && cur.info.exists)) return render();
    S.probing = true;
    render();
    const r = await api('POST', '/probe', { ref: ref.path, cur: cur.path, lang: lang() });
    S.probing = false;
    S.probe = r.ok ? r.json : { blind: true, reason: errorText(r), hints: [] };
    render();
  }

  const errorText = (r) => {
    const d = r.json && (r.json.detail || r.json);
    return (d && (d.message || d.error_code)) || ('HTTP ' + r.status);
  };

  async function filesFromDrop(dt) {
    const entries = [...(dt.items || [])].map((i) => i.webkitGetAsEntry && i.webkitGetAsEntry()).filter(Boolean);
    if (!entries.length) return [...dt.files].map((f) => ({ file: f, path: f.name }));
    const out = [];
    async function walk(entry, prefix) {
      if (entry.name.startsWith('.')) return;
      if (entry.isFile) {
        out.push({ file: await new Promise((res, rej) => entry.file(res, rej)), path: prefix + entry.name });
      } else if (entry.isDirectory) {
        const reader = entry.createReader();
        for (;;) {
          const batch = await new Promise((res, rej) => reader.readEntries(res, rej));
          if (!batch.length) break;
          for (const e of batch) await walk(e, prefix + entry.name + '/');
        }
      }
    }
    for (const e of entries) await walk(e, '');
    return out;
  }

  async function uploadTo(side, items) {
    items = items.filter((it) => !it.path.split('/').some((p) => p.startsWith('.')));
    if (!items.length) return;
    const total = items.reduce((n, it) => n + it.file.size, 0);
    const show = (done) => {
      S.sides[side] = { busy: t('drop.uploading', { done, total: items.length, size: bytes(total) }) };
      render();
    };
    show(0);
    try {
      const session = await api('POST', '/workspace/uploads');
      if (!session.ok) throw new Error(errorText(session));
      let last = null;
      for (let i = 0; i < items.length; i++) {
        const url = `/workspace/uploads/${session.json.id}/${side}/` + items[i].path.split('/').map(encodeURIComponent).join('/');
        const r = await fetch(url, { method: 'PUT', body: items[i].file,
          headers: authHeaders({ 'Content-Type': 'application/octet-stream' }) });
        last = await r.json().catch(() => null);
        if (!r.ok) throw new Error(errorText({ json: last, status: r.status }));
        show(i + 1);
      }
      await describeSide(side, items.length === 1 ? last.file : last.path, 'upload');
    } catch (e) {
      S.sides[side] = { error: t('err.generic', { msg: e.message }) };
      render();
    }
  }

  function pickFiles(side, folder) {
    const input = h('input', { type: 'file', multiple: true, hidden: true });
    if (folder) input.webkitdirectory = true;
    input.addEventListener('change', () => {
      uploadTo(side, [...input.files].map((f) => ({ file: f, path: f.webkitRelativePath || f.name })));
      input.remove();
    });
    document.body.appendChild(input);
    input.click();
  }

  // ── Server browser ───────────────────────────────────────────────
  async function browse(side, path) {
    S.browser = { side, data: (S.browser && S.browser.data) || null, loading: true };
    renderBrowser();
    const r = await fetch('/fs/list' + (path ? '?path=' + encodeURIComponent(path) : ''), { headers: authHeaders() });
    const json = await r.json().catch(() => null);
    S.browser = r.ok ? { side, data: json } : { side, data: S.browser.data, error: errorText({ json, status: r.status }) };
    renderBrowser();
  }
  const closeBrowser = () => { S.browser = null; renderBrowser(); };
  function chooseFromBrowser(path) {
    const side = S.browser.side;
    closeBrowser();
    describeSide(side, path);
  }

  function renderBrowser() {
    document.querySelectorAll('.wb-modal').forEach((n) => n.remove());
    const b = S.browser;
    if (!b) return;
    const d = b.data;
    const body = h('div', { class: 'fs-list' });
    if (d) {
      if (!d.entries.length) body.appendChild(h('p', { class: 'muted pad' }, t('fs.empty')));
      for (const e of d.entries) {
        const pickable = e.type === 'dir' || e.kind !== 'other';
        body.appendChild(h('button', {
          class: 'fs-row' + (pickable ? '' : ' dim'), type: 'button', disabled: !pickable,
          onclick: () => (e.type === 'dir' ? browse(b.side, e.path) : chooseFromBrowser(e.path)),
        }, icon(KIND_ICON[e.kind] || 'file'), h('span', { class: 'fs-name' }, e.name),
        h('span', { class: 'fs-meta' }, e.type === 'dir' ? '' : bytes(e.size || 0))));
      }
      if (d.truncated) body.appendChild(h('p', { class: 'muted pad' }, t('fs.truncated', { n: d.entries.length })));
    }
    const counts = d && d.path ? Object.entries(d.counts).filter(([k]) => k !== 'dir' && k !== 'other')
      .map(([k, n]) => `${t('kind.' + (k === 'image' ? 'vision' : k))} ${num(n)}`).join(' · ') : '';
    const modal = h('div', { class: 'wb-modal', onclick: (ev) => { if (ev.target === modal) closeBrowser(); } },
      h('div', { class: 'wb-dialog', role: 'dialog', 'aria-label': t('fs.title') },
        h('header', {},
          h('h3', {}, t('fs.title') + ' · ' + t(b.side === 'ref' ? 'side.ref' : 'side.cur')),
          h('p', { class: 'muted' }, t('fs.hint'))),
        h('div', { class: 'fs-bar' },
          h('button', { class: 'wb-btn ghost', type: 'button', disabled: !(d && d.path),
            onclick: () => browse(b.side, d.parent) }, icon('back', 16), ' ' + (d && d.path && !d.parent ? t('fs.roots') : t('fs.up'))),
          h('code', { class: 'fs-path' }, '\u200e' + ((d && d.path) || t('fs.roots')))),
        b.error ? h('p', { class: 'wb-error' }, b.error) : null,
        body,
        h('footer', {},
          h('span', { class: 'muted' }, counts),
          h('button', { class: 'wb-btn ghost', type: 'button', onclick: closeBrowser }, t('fs.cancel')),
          h('button', { class: 'wb-btn primary', type: 'button', disabled: !(d && d.path),
            onclick: () => chooseFromBrowser(d.path) }, t('fs.use_folder')))));
    document.body.appendChild(modal);
  }

  // ── Diagnose ─────────────────────────────────────────────────────
  let ticker = null;
  async function diagnose() {
    const { ref, cur } = S.sides;
    if (!(ref && cur && ref.path && cur.path) || S.running) return;
    S.running = true; S.started = Date.now(); S.error = null;
    render();
    ticker = setInterval(() => {
      const b = root.querySelector('.run-btn');
      if (b) b.textContent = t('run.busy', { s: Math.round((Date.now() - S.started) / 1000) });
    }, 500);
    const o = S.options;
    const r = await api('POST', '/diff', {
      ref: ref.path, cur: cur.path, lang: lang(), issuer: o.issuer || null,
      sample_rows: Number(o.sample_rows) || 0, with_embeddings: !!o.with_embeddings,
      threshold: Number(o.threshold) || 0.15, attr_threshold: Number(o.attr_threshold) || 0.25,
      include_report: true,
    });
    clearInterval(ticker);
    S.running = false;
    if (!r.ok || !r.json || !r.json.verdict) {
      S.error = t('err.generic', { msg: errorText(r) });
      return render();
    }
    S.result = r.json;
    const list = recent().filter((x) => !(x.ref === ref.path && x.cur === cur.path));
    list.unshift({ ref: ref.path, cur: cur.path, verdict: r.json.verdict, at: Date.now(),
      refName: (ref.info && ref.info.name) || baseName(ref.path), curName: (cur.info && cur.info.name) || baseName(cur.path),
      caseTitle: S.activeCase ? S.activeCase.title : null });
    localStorage.setItem(RECENT_KEY, JSON.stringify(list.slice(0, 8)));
    render();
    window.scrollTo({ top: 0 });
  }

  async function runCase(c) {
    S.activeCase = c; S.result = null; S.error = null;
    S.sides = { ref: { busy: t('drop.reading') }, cur: { busy: t('drop.reading') } };
    render();
    const r = await api('POST', '/cases/' + encodeURIComponent(c.id));
    if (!r.ok) { S.sides = { ref: null, cur: null }; S.error = t('err.generic', { msg: errorText(r) }); return render(); }
    await Promise.all([describeSide('ref', r.json.ref, 'case'), describeSide('cur', r.json.cur, 'case')]);
    diagnose();
  }

  // ── Render: intake ───────────────────────────────────────────────
  function factList(info) {
    const facts = [];
    const p = info.profile;
    if (p) {
      facts.push([num(p.rows), t('card.rows')], [num(p.columns.length), t('card.columns')]);
      if (p.files > 1) facts.push([num(p.files), t('card.files')]);
    } else if (info.counts) {
      for (const [k, n] of Object.entries(info.counts)) facts.push([num(n), t(k === 'vision' ? 'card.images' : 'card.items')]);
    }
    return facts;
  }

  function dataCard(side, s) {
    const info = s.info;
    if (!info.exists) return h('div', { class: 'data-card' }, h('p', { class: 'wb-error' }, t('card.missing')), h('code', {}, s.path));
    const p = info.profile;
    const card = h('div', { class: 'data-card' },
      h('div', { class: 'dc-head' },
        h('span', { class: 'kind' }, icon(KIND_ICON[info.kind] || 'file', 16), ' ' + t('kind.' + (info.kind || 'unknown'))),
        h('b', { class: 'dc-name' }, info.name || baseName(s.path))),
      s.source ? h('span', { class: 'dc-src' }, t('card.src.' + s.source))
        : h('code', { class: 'dc-path', title: s.path }, '\u200e' + s.path),
      h('div', { class: 'facts' }, factList(info).map(([v, k]) => h('div', {}, h('b', {}, v), h('span', {}, k))),
        p && p.period ? h('div', { class: 'wide' }, h('b', {}, `${p.period.start} ~ ${p.period.end}`), h('span', {}, t('card.period'))) : null));
    if (info.kind === 'unknown') card.appendChild(h('p', { class: 'wb-error' }, t('card.unknown')));
    if (info.error) card.appendChild(h('p', { class: 'wb-error' }, info.error));
    if (p) {
      const cols = p.columns.slice(0, 14);
      card.appendChild(h('div', { class: 'chips' },
        cols.map((c) => h('span', { class: 'chip t-' + c.type, title: t('type.' + c.type) }, c.name)),
        p.columns.length > cols.length ? h('span', { class: 'chip more' }, t('card.more', { n: p.columns.length - cols.length })) : null));
    }
    if (info.images && info.images.length) {
      const strip = h('div', { class: 'thumbs' });
      card.appendChild(strip);
      for (const path of info.images) {
        fetch('/fs/image?path=' + encodeURIComponent(path), { headers: authHeaders() })
          .then((r) => (r.ok ? r.blob() : null))
          .then((b) => { if (b) strip.appendChild(h('img', { src: URL.createObjectURL(b), alt: '' })); })
          .catch(() => {});
      }
    }
    return card;
  }

  function slot(side) {
    const s = S.sides[side];
    const head = h('div', { class: 'slot-head' },
      h('span', { class: 'tag ' + side }, t(`side.${side}.tag`)),
      h('div', {}, h('h3', {}, t('side.' + side)), h('p', { class: 'muted' }, t(`side.${side}.sub`))),
      s && s.info ? h('button', { class: 'wb-btn ghost small', type: 'button',
        onclick: () => { S.sides[side] = null; S.probe = null; S.activeCase = null; render(); } }, t('card.change')) : null);
    let body;
    if (s && s.busy) body = h('div', { class: 'drop busy' }, h('span', { class: 'spinner' }), h('p', {}, s.busy));
    else if (s && s.info) body = dataCard(side, s);
    else {
      const typed = h('input', { type: 'text', placeholder: t('drop.path'), value: (s && s.path) || '' });
      const useTyped = () => { if (typed.value.trim()) describeSide(side, typed.value.trim()); };
      typed.addEventListener('keydown', (ev) => { if (ev.key === 'Enter') useTyped(); });
      body = h('div', { class: 'drop' },
        icon('upload', 30),
        h('p', { class: 'drop-title' }, t('drop.title')),
        h('p', { class: 'muted' }, t('drop.kinds')),
        h('div', { class: 'btns' },
          h('button', { class: 'wb-btn', type: 'button', onclick: () => pickFiles(side, false) }, t('drop.files')),
          h('button', { class: 'wb-btn', type: 'button', onclick: () => pickFiles(side, true) }, t('drop.folder')),
          h('button', { class: 'wb-btn', type: 'button', onclick: () => browse(side, null) }, icon('folder', 15), ' ' + t('drop.browse'))),
        s && s.error ? h('p', { class: 'wb-error' }, s.error) : null,
        h('details', { class: 'typed', open: !!(s && s.typing) },
          h('summary', {}, t('drop.type')),
          h('div', { class: 'typed-row' }, typed, h('button', { class: 'wb-btn', type: 'button', onclick: useTyped }, t('drop.use')))));
    }
    const node = h('section', { class: 'slot ' + side + (s && s.info ? ' filled' : '') }, head, body);
    node.addEventListener('dragover', (ev) => { ev.preventDefault(); node.classList.add('over'); });
    node.addEventListener('dragleave', () => node.classList.remove('over'));
    node.addEventListener('drop', async (ev) => {
      ev.preventDefault();
      node.classList.remove('over');
      S.activeCase = null;
      uploadTo(side, await filesFromDrop(ev.dataTransfer));
    });
    return node;
  }

  function precheck() {
    if (S.probing) return h('div', { class: 'precheck' }, h('span', { class: 'spinner small' }), ' ' + t('pre.checking'));
    const p = S.probe;
    if (!p) return null;
    if (p.blind) {
      return h('div', { class: 'precheck blind' }, h('b', {}, '◌ ' + t('pre.blind')),
        h('span', {}, p.reason || ''), (p.hints || []).map((x) => h('span', { class: 'hint' }, x)));
    }
    const bits = [];
    if (p.csv) {
      bits.push(t('pre.compared', { n: p.csv.compared.length }) + ': ' + p.csv.compared.slice(0, 8).join(', ')
        + (p.csv.compared.length > 8 ? ' …' : ''));
      if (p.csv.time.length) bits.push(t('pre.time', { cols: p.csv.time.join(', ') }));
      if (p.csv.only_ref.length) bits.push(t('pre.only_ref', { cols: p.csv.only_ref.join(', ') }));
      if (p.csv.only_cur.length) bits.push(t('pre.only_cur', { cols: p.csv.only_cur.join(', ') }));
    } else if (p.route) bits.push(p.route);
    return h('div', { class: 'precheck ok' }, h('b', {}, '✓ ' + t('pre.ok')), bits.map((b) => h('span', {}, b)));
  }

  function optionsPanel() {
    const o = S.options;
    const field = (key, label, attrs) => h('label', { class: attrs.type === 'checkbox' ? 'check' : '' },
      attrs.type === 'checkbox' ? null : h('span', {}, label),
      h('input', { ...attrs, ...(attrs.type === 'checkbox' ? { checked: o[key] } : { value: o[key] }),
        onchange: (ev) => { o[key] = attrs.type === 'checkbox' ? ev.target.checked : ev.target.value; } }),
      attrs.type === 'checkbox' ? h('span', {}, label) : null);
    return h('details', { class: 'options' }, h('summary', {}, t('opt.title')),
      h('div', { class: 'opt-grid' },
        field('issuer', t('opt.issuer'), { type: 'text', placeholder: 'ddoc' }),
        field('sample_rows', t('opt.sample'), { type: 'number', min: 0, max: 20, step: 1 }),
        field('threshold', t('opt.threshold'), { type: 'number', min: 0, max: 1, step: 0.01 }),
        field('attr_threshold', t('opt.attr'), { type: 'number', min: 0, max: 1, step: 0.01 }),
        field('with_embeddings', t('opt.embed'), { type: 'checkbox' })));
  }

  function verdictBadge(v) {
    return h('span', { class: 'vbadge v-' + String(v).toLowerCase() }, v);
  }

  function intake() {
    const { ref, cur } = S.sides;
    const ready = !!(ref && cur && ref.info && cur.info && ref.info.exists && cur.info.exists);
    const list = recent();
    return h('div', { class: 'wb-intake' },
      h('header', { class: 'wb-hero' }, h('h1', {}, t('intake.title')), h('p', {}, t('intake.lead'))),
      h('div', { class: 'pair' }, slot('ref'), h('div', { class: 'vs' }, icon('arrow', 26)), slot('cur')),
      precheck(),
      S.error ? h('p', { class: 'wb-error block' }, S.error) : null,
      h('div', { class: 'runbar' },
        h('button', { class: 'wb-btn primary big run-btn', type: 'button', disabled: !ready || S.running,
          title: ready ? '' : t('run.need'), onclick: diagnose }, S.running ? t('run.busy', { s: 0 }) : t('run')),
        optionsPanel()),
      S.cases.length ? h('section', { class: 'cases' },
        h('h2', {}, t('cases.title')), h('p', { class: 'muted' }, t('cases.lead')),
        h('div', { class: 'case-grid' }, S.cases.map((c) => h('article', { class: 'case' },
          h('div', { class: 'case-top' }, h('span', { class: 'glyph' }, c.icon),
            h('span', { class: 'expect' }, t('cases.expect') + ' ', verdictBadge(c.verdict))),
          h('h3', {}, c.title), h('p', {}, c.story),
          h('button', { class: 'wb-btn', type: 'button', disabled: S.running, onclick: () => runCase(c) }, t('cases.run')))))) : null,
      list.length ? h('section', { class: 'recent' },
        h('div', { class: 'recent-head' }, h('h2', {}, t('recent.title')),
          h('button', { class: 'link', type: 'button', onclick: () => { localStorage.removeItem(RECENT_KEY); render(); } }, t('recent.clear'))),
        list.map((x) => h('button', { class: 'recent-row', type: 'button', onclick: () => {
          S.activeCase = null; describeSide('ref', x.ref); describeSide('cur', x.cur);
        } }, verdictBadge(x.verdict), h('span', { class: 'names' }, x.caseTitle || `${x.refName} → ${x.curName}`),
        h('span', { class: 'muted' }, new Date(x.at).toLocaleString(lang() === 'ko' ? 'ko-KR' : 'en-US', { dateStyle: 'short', timeStyle: 'short' }))))) : null);
  }

  // ── Render: result ───────────────────────────────────────────────
  const STATUS_ICON = { pass: '✓', caution: '!', fail: '✕' };
  const STATUS_TOKEN = { pass: 'good', caution: 'warning', fail: 'critical' };

  function download(name, blob) {
    const a = h('a', { href: URL.createObjectURL(blob), download: name });
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 2000);
  }

  async function saveRendered(fmt, btn) {
    const label = btn.textContent;
    btn.disabled = true; btn.textContent = t('res.saving');
    const { report_html, view, exit_code, ...envelope } = S.result;
    try {
      const r = await fetch('/report/render', { method: 'POST', headers: authHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ envelope, format: fmt, lang: lang(), issuer: S.options.issuer || null }) });
      if (!r.ok) throw new Error(errorText({ json: await r.json().catch(() => null), status: r.status }));
      download('ddoc-report.' + fmt, await r.blob());
    } catch (e) {
      alert(t('err.generic', { msg: e.message }));
    }
    btn.disabled = false; btn.textContent = label;
  }

  function sideSummary(side, v) {
    const s = S.sides[side] || {};
    const d = v.sides[side];
    const bits = [];
    if (d) {
      bits.push(`${num(d.total_rows || d.rows)} ${t('card.rows')}`, `${num(d.columns)} ${t('card.columns')}`);
      if (d.period) bits.push(`${d.period.start} ~ ${d.period.end}`);
    }
    return h('div', { class: 'cmp-side' },
      h('span', { class: 'tag ' + side }, t(`side.${side}.tag`)),
      h('div', {}, h('b', {}, (s.info && s.info.name) || baseName(S.result[side])), h('span', { class: 'muted' }, bits.join(' · '))));
  }

  function result() {
    const res = S.result;
    const v = res.view || { verdict: res.verdict, token: 'warning', icon: '?', cards: [], attributes: [], sides: {}, rationale: [], checks: [], hints: res.hints || [], labels: {} };
    const scale = Math.max(1, ...v.attributes.map((a) => a.score), v.score || 0);
    const pct = (x) => Math.min(100, (100 * x) / scale).toFixed(1) + '%';

    const band = h('section', { class: 'verdict tok-' + v.token },
      h('div', { class: 'v-main' },
        h('span', { class: 'v-icon' }, v.icon),
        h('div', {},
          h('div', { class: 'v-word' }, v.verdict),
          h('div', { class: 'v-sub' }, v.verdict === 'DRIFT' ? `${t('res.severity')} ${v.severity_label}` : (v.reason || '')))),
      v.score != null ? h('div', { class: 'v-score' },
        h('div', { class: 'v-score-top' }, h('span', {}, t('res.score')), h('b', {}, Number(v.score).toFixed(3)),
          h('span', { class: 'muted' }, `${t('res.threshold')} ${v.threshold}`)),
        h('div', { class: 'gauge' }, h('i', { style: `width:${pct(v.score)}` }), h('u', { style: `left:${pct(v.threshold)}` }))) : null,
      h('div', { class: 'v-text' },
        v.headline ? h('p', { class: 'v-headline' }, v.headline) : null,
        (v.rationale || []).map((x) => h('p', { class: 'muted' }, x)),
        (v.warnings || []).map((x) => h('p', { class: 'v-warn' }, '⚠ ' + x))));

    const caseNote = S.activeCase ? h('aside', { class: 'case-note' },
      h('b', {}, `${t('res.case')}: ${S.activeCase.title}`), h('span', {}, S.activeCase.story),
      h('span', {}, h('i', {}, t('res.case.look') + ' — '), S.activeCase.look)) : null;

    const compared = h('section', { class: 'compared' }, sideSummary('ref', v), icon('arrow', 20), sideSummary('cur', v));

    const cards = h('div', { class: 'cards' },
      h('h3', {}, t('res.patterns')),
      v.cards.length ? v.cards.map((c) => h('article', { class: 'pcard ' + c.match },
        h('header', {}, h('b', {}, c.title), h('span', { class: 'mtag ' + c.match }, t('res.' + c.match)),
          c.attrs.length ? h('span', { class: 'attrs' }, c.attrs.slice(0, 4).join(', ')) : null),
        c.signals.map((s) => h('p', { class: 'signal' }, s)),
        h('p', {}, c.meaning),
        c.actions.length ? h('div', { class: 'first' }, h('span', {}, t('res.first')), h('ul', {}, c.actions.map((a) => h('li', {}, a)))) : null))
        : h('p', { class: 'muted' }, t('res.no_patterns')));

    const cols = h('div', { class: 'columns' },
      h('div', { class: 'col-head' }, h('h3', {}, t('res.columns')),
        h('span', { class: 'legend' }, h('i', { class: 'l-ref' }), v.labels.ref || t('res.legend.ref'),
          h('i', { class: 'l-cur' }), v.labels.cur || t('res.legend.cur'))),
      v.attributes.map((a) => h('div', { class: 'arow' },
        h('span', { class: 'st tok-' + STATUS_TOKEN[a.status], title: t('status.' + a.status) }, STATUS_ICON[a.status]),
        h('div', { class: 'a-main' },
          h('div', { class: 'a-top' }, h('b', {}, a.name), h('span', { class: 'a-score' }, a.score.toFixed(3))),
          h('div', { class: 'gauge thin' }, h('i', { style: `width:${pct(a.score)}` }), h('u', { style: `left:${pct(v.attr_threshold)}` })),
          a.summary ? h('p', { class: 'muted' }, a.summary) : null),
        a.spark ? h('div', { class: 'spark', html: a.spark }) : h('div', { class: 'spark' }))),
      v.attributes_total > v.attributes.length
        ? h('p', { class: 'muted' }, t('res.more_columns', { n: v.attributes_total - v.attributes.length })) : null);

    const checks = (v.checks.length || v.hints.length) ? h('section', { class: 'checks' },
      h('h3', {}, t('res.checks')),
      h('ul', {}, (v.checks.length ? v.checks : v.hints).map((c) => h('li', {}, c)))) : null;

    let report = null;
    if (res.report_html) {
      const cli = `ddoc diff '${res.ref}' '${res.cur}' --lang ${lang()} --report note.html`;
      const copy = h('button', { class: 'wb-btn ghost', type: 'button', onclick: async () => {
        try { await navigator.clipboard.writeText(cli); copy.textContent = t('res.copied'); setTimeout(() => (copy.textContent = t('res.cli')), 1200); } catch (_) { prompt('', cli); }
      } }, t('res.cli'));
      const blob = () => new Blob([res.report_html], { type: 'text/html' });
      const pdf = h('button', { class: 'wb-btn', type: 'button', onclick: () => saveRendered('pdf', pdf) }, t('res.pdf'));
      const md = h('button', { class: 'wb-btn', type: 'button', onclick: () => saveRendered('md', md) }, t('res.md'));
      report = h('section', { class: 'report' },
        h('div', { class: 'report-bar' }, h('h3', {}, t('res.report')),
          h('button', { class: 'wb-btn', type: 'button', onclick: () => window.open(URL.createObjectURL(blob()), '_blank', 'noopener') }, t('res.open')),
          h('button', { class: 'wb-btn', type: 'button', onclick: () => download('ddoc-report.html', blob()) }, t('res.html')),
          pdf, md, copy),
        // no allow-same-origin: the report cannot reach into this page
        h('iframe', { class: 'report-frame', title: t('res.report'), sandbox: 'allow-scripts allow-modals allow-popups', srcdoc: res.report_html }));
    }

    return h('div', { class: 'wb-result' },
      h('div', { class: 'res-top' },
        h('button', { class: 'wb-btn ghost', type: 'button', onclick: () => { S.result = null; render(); } }, icon('back', 16), ' ' + t('res.new'))),
      band, caseNote, compared,
      v.verdict === 'BLIND' ? null : h('section', { class: 'what' }, h('h2', {}, t('res.what')), h('div', { class: 'what-grid' }, cards, cols)),
      checks, report);
  }

  // ── Render ───────────────────────────────────────────────────────
  function render() {
    root.replaceChildren(S.result ? result() : intake());
  }

  function applyView() {
    const advanced = location.hash === '#advanced';
    document.getElementById('advanced').hidden = !advanced;
    root.hidden = advanced;
    document.querySelectorAll('#views .view').forEach((b) => {
      b.classList.toggle('active', (b.dataset.view === 'advanced') === advanced);
      b.textContent = t('view.' + b.dataset.view);
    });
  }
  document.querySelectorAll('#views .view').forEach((b) => b.addEventListener('click', () => {
    location.hash = b.dataset.view === 'advanced' ? '#advanced' : '';
  }));
  window.addEventListener('hashchange', applyView);

  async function loadCases() {
    const r = await api('GET', '/cases?lang=' + encodeURIComponent(lang()));
    S.cases = Array.isArray(r.json) ? r.json : [];
    if (S.activeCase) S.activeCase = S.cases.find((c) => c.id === S.activeCase.id) || S.activeCase;
    render();
  }
  document.addEventListener('ddoc:lang', () => { applyView(); loadCases(); refreshProbe(); });
  document.addEventListener('ddoc:auth', loadCases);        // the key was entered: load what needed it
  document.addEventListener('keydown', (ev) => { if (ev.key === 'Escape' && S.browser) closeBrowser(); });

  // Links that open on a diagnosis: ?case=<id> runs an example case,
  // ?ref=<path>&cur=<path> fills both sides (&run=1 diagnoses at once).
  async function openFromLink() {
    const q = new URLSearchParams(location.search);
    const wanted = q.get('case') && S.cases.find((c) => c.id === q.get('case'));
    if (wanted) return runCase(wanted);
    if (q.get('ref') && q.get('cur')) {
      await Promise.all([describeSide('ref', q.get('ref')), describeSide('cur', q.get('cur'))]);
      if (q.get('run') === '1') diagnose();
    }
  }

  applyView();
  render();
  loadCases().then(openFromLink);
})();
