'use strict';

const $ = id => document.getElementById(id);
const state = {
  manifest: null, tableId: null, columns: [], rows: [],
  view: [], page: 0, size: 100, sortCol: -1, sortDir: 1, q: '',
};
const cache = new Map();
const IMG = /\.(jpe?g|png|gif|webp|avif)(\?|$)/i;

// ---------- boot ----------
(async function main() {
  const res = await fetch('data/manifest.json');
  state.manifest = await res.json();
  $('origin').textContent = `${state.manifest.generated_at} · ${state.manifest.total_rows.toLocaleString()} 行 / ${state.manifest.total_tables} 张表`;

  const g = $('group');
  state.manifest.groups.forEach((grp, i) => {
    const o = document.createElement('option');
    o.value = i; o.textContent = `${grp.label}（${grp.tables.length}）`;
    g.append(o);
  });
  g.onchange = () => { fillTables(+g.value); select(state.manifest.groups[+g.value].tables[0].id); };
  $('table').onchange = () => select($('table').value);

  let qi;
  $('q').oninput = () => {
    clearTimeout(qi);
    qi = setTimeout(() => { state.q = $('q').value.trim().toLowerCase(); state.page = 0; apply(); }, 160);
  };
  $('size').onchange = () => {
    state.size = +$('size').value; state.page = 0;
    if (state.size === 0 && state.rows.length > 6000 &&
        !confirm(`这张表有 ${state.rows.length.toLocaleString()} 行，全部渲染可能让浏览器卡住。继续？`)) {
      $('size').value = '100'; state.size = 100;
    }
    render();
  };
  $('thumbs').onchange = render;
  $('first').onclick = () => { state.page = 0; render(); };
  $('prev').onclick = () => { state.page--; render(); };
  $('next').onclick = () => { state.page++; render(); };
  $('last').onclick = () => { state.page = lastPage(); render(); };

  fillTables(0);
  select(state.manifest.groups[0].tables[0].id);
  installLightbox();
})();

function fillTables(gi) {
  const sel = $('table');
  sel.innerHTML = '';
  state.manifest.groups[gi].tables.forEach(t => {
    const o = document.createElement('option');
    o.value = t.id;
    o.textContent = `${t.label}  ·  ${t.rows.toLocaleString()}×${t.cols}`;
    sel.append(o);
  });
}

async function select(id) {
  state.tableId = id;
  $('table').value = id;
  $('status').textContent = '载入中…';
  if (!cache.has(id)) {
    const r = await fetch(`data/${id}.json`);
    cache.set(id, await r.json());
  }
  const d = cache.get(id);
  state.columns = d.columns;
  state.rows = d.rows;
  state.page = 0; state.sortCol = -1; state.sortDir = 1;
  $('q').value = ''; state.q = '';
  apply();
}

// ---------- filter + sort ----------
function apply() {
  const q = state.q;
  state.view = q
    ? state.rows.filter(r => r.some(v => v != null && String(v).toLowerCase().includes(q)))
    : state.rows.slice();

  if (state.sortCol >= 0) {
    const c = state.sortCol, dir = state.sortDir;
    const numeric = state.view.length > 0 &&
      state.view.slice(0, 200).every(r => r[c] == null || r[c] === '' || !isNaN(Number(r[c])));
    state.view.sort((a, b) => {
      const x = a[c], y = b[c];
      if (x == null || x === '') return 1;          // blanks sink in both directions
      if (y == null || y === '') return -1;
      const r = numeric ? Number(x) - Number(y) : String(x).localeCompare(String(y), 'zh');
      return r * dir;
    });
  }
  render();
}

const lastPage = () => state.size === 0 ? 0 : Math.max(0, Math.ceil(state.view.length / state.size) - 1);

// ---------- render ----------
function render() {
  const size = state.size || state.view.length;
  const maxPage = lastPage();
  if (state.page > maxPage) state.page = maxPage;
  const start = size === 0 ? 0 : state.page * size;
  const slice = state.view.slice(start, start + size);

  const head = $('grid').tHead;
  head.innerHTML = '';
  const hr = document.createElement('tr');
  hr.append(th('', -1));
  state.columns.forEach((c, i) => {
    const t = th(c, i);
    if (i === state.sortCol) {
      const a = document.createElement('span');
      a.className = 'arrow'; a.textContent = state.sortDir > 0 ? '▲' : '▼';
      t.append(a);
    }
    hr.append(t);
  });
  head.append(hr);
  function th(label, i) {
    const e = document.createElement('th');
    e.textContent = label;
    if (i >= 0) e.title = `${label} — 点击排序`;
    e.onclick = i < 0 ? null : () => {
      if (state.sortCol === i) state.sortDir = -state.sortDir;
      else { state.sortCol = i; state.sortDir = 1; }
      apply();
    };
    return e;
  }

  const body = $('grid').tBodies[0];
  const frag = document.createDocumentFragment();
  const showThumbs = $('thumbs').checked;
  slice.forEach((row, k) => {
    const tr = document.createElement('tr');
    const n = document.createElement('td');
    n.className = 'rownum'; n.textContent = start + k + 1;
    tr.append(n);
    row.forEach((v, i) => {
      const td = document.createElement('td');
      if (v == null || v === '') {
        td.className = 'null'; td.textContent = '—';
      } else {
        const s = String(v);
        if (showThumbs && IMG.test(s) && (s.startsWith('images/') || s.includes('images/'))) {
          const im = document.createElement('img');
          im.className = 'thumb'; im.loading = 'lazy'; im.src = s; im.alt = '';
          im.onclick = ev => { ev.stopPropagation(); zoom(s); };
          td.append(im);
        } else {
          td.textContent = s;
          if (s.length > 90) {
            // The cell holds the whole value; CSS clips it to one line. Clicking
            // only flips `expanded`, which lets it wrap -- never truncate the
            // text itself, or a second click could not restore it.
            td.title = '点击展开 / 收起';
            td.onclick = () => td.classList.toggle('expanded');
          } else {
            td.title = s;
          }
        }
      }
      tr.append(td);
    });
    frag.append(tr);
  });
  body.replaceChildren(frag);

  $('empty').hidden = state.view.length > 0;
  $('count').textContent = state.q
    ? `筛出 ${state.view.length.toLocaleString()} / ${state.rows.length.toLocaleString()} 行`
    : `${state.rows.length.toLocaleString()} 行 · ${state.columns.length} 列`;
  $('pageinfo').textContent = state.view.length === 0 ? '—'
    : `${state.page + 1} / ${maxPage + 1}（第 ${start + 1}–${Math.min(start + size, state.view.length)} 行）`;
  $('first').disabled = $('prev').disabled = state.page === 0;
  $('last').disabled = $('next').disabled = state.page >= maxPage;
  $('status').textContent = state.tableId;
}

// ---------- lightbox ----------
function installLightbox() {
  const lb = document.createElement('div');
  lb.id = 'lightbox';
  const im = document.createElement('img');
  lb.append(im);
  lb.onclick = () => lb.classList.remove('on');
  document.addEventListener('keydown', e => { if (e.key === 'Escape') lb.classList.remove('on'); });
  document.body.append(lb);
}
function zoom(src) {
  const lb = $('lightbox');
  lb.querySelector('img').src = src;
  lb.classList.add('on');
}
