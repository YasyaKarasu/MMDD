"use strict";
const $ = id => document.getElementById(id);
const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const num = value => value == null ? "—" : Number(value).toFixed(4);
const rank = value => value == null ? "—" : `#${value}`;
const imageUrl = id => `/image/${encodeURIComponent(id)}`;
const params = new URLSearchParams(location.search);
const state = {arm: params.get("arm") || "baseline", split: params.get("split") || "test",
  generator: params.get("model") || "raw", query: params.get("q"), evidence: params.get("e"),
  tab: params.get("view") || "paths", modality: "all", catalog: null, assets: null, tables: null, data: null};
const tableCache = new Map();
let requestVersion = 0;

async function getJSON(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`数据加载失败 (${response.status})`);
  return response.json();
}

function updateURL() {
  const p = new URLSearchParams({arm: state.arm, split: state.split, model: state.generator,
    q: state.query, view: state.tab});
  if (state.evidence) p.set("e", state.evidence);
  history.replaceState(null, "", `?${p}`);
}

function toast(text) {
  $("toast").textContent = text;
  $("toast").style.display = "block";
  setTimeout(() => {$("toast").style.display = "none";}, 2500);
}

function availableQueries() {
  const ids = new Set(state.catalog.combination_queries[`${state.arm}/${state.split}/${state.generator}`] || []);
  return state.catalog.queries.filter(q => ids.has(q.id));
}

function syncControls() {
  $("arm").value = state.arm;
  const pruned = ["columns", "both"].includes(state.arm);
  $("dataset-note").textContent = `当前：${state.catalog.arms[state.arm]}（${state.arm}）。` + (pruned ?
    "上一轮已从源表、Query 和 Target 同步删除 currency / shipping_currency（USD）、language 等 15 种列；品相、装帧、库存量仍保留。" :
    "保留 currency / shipping_currency（USD）和 language；这是未删列的对照版本。查看删列结果请切换到「仅精简列」或「两者同时」。");
  $("split").value = state.split;
  const options = state.catalog.available[`${state.arm}/${state.split}`] || [];
  if (!options.includes(state.generator)) state.generator = "raw";
  $("generator").innerHTML = options.map(g => `<option value="${esc(g)}">${esc(state.catalog.generators[g])}</option>`).join("");
  $("generator").value = state.generator;
  $("mode-note").textContent = state.split === "train" ? "训练集展示已保存的 Raw 两跳结果" :
    state.generator === "raw" ? "原始 embedding 检索 · 文本 / 图片各取 top20" : "Student 检索 · Teacher 排名作为后续评分对照";
}

function renderQueries() {
  if (!state.catalog) return;
  const search = $("query-search").value.toLowerCase().trim();
  const all = availableQueries();
  const queries = all.filter(q => `${q.id} ${q.label} ${state.tables?.[q.id]?.rows.flat().join(" ") || ""}`.toLowerCase().includes(search));
  $("query-count").textContent = `${queries.length} / ${all.length}`;
  $("query-list").innerHTML = queries.map(q => `<button class="query-item ${q.id === state.query ? "active" : ""}" data-query="${esc(q.id)}" aria-pressed="${q.id === state.query}">
    <div class="q-top"><span class="q-num">Q${String(all.indexOf(q)+1).padStart(2,"0")}</span><span class="pill">${q.kind === "implicit" ? "隐式 join" : "显式 join"}</span></div>
    <div class="q-label">${esc(q.label)}</div><code>${esc(q.id)}</code></button>`).join("") || '<div class="empty">没有匹配的 Query</div>';
}

async function loadQuery() {
  const version = ++requestVersion;
  syncControls();
  const queries = availableQueries();
  if (!queries.some(q => q.id === state.query)) state.query = queries[0]?.id;
  renderQueries();
  $("status").hidden = false;
  $("status").textContent = "正在读取这条 Query 的检索路径…";
  $("content").hidden = true;
  try {
    if (!state.query) throw new Error("这个组合没有可用的查询结果。");
    const arm = state.arm;
    if (!tableCache.has(arm)) tableCache.set(arm, getJSON(`/data/${arm}/tables.json.gz`));
    const [tables, data] = await Promise.all([tableCache.get(arm),
      getJSON(`/data/${arm}/${state.split}/${state.generator}/${state.query}.json.gz`)]);
    if (version !== requestVersion) return;
    state.tables = tables;
    state.data = data;
    if (!data.evidence.some(e => e.id === state.evidence) && !data.gold.some(g => g.evidence_ids.includes(state.evidence))) state.evidence = data.evidence[0]?.id;
    $("status").hidden = true;
    $("content").hidden = false;
    renderHeader();
    renderEvidenceList(true);
    renderDirect();
    renderGold();
    setTab(state.tab);
    updateURL();
  } catch (error) {
    if (version === requestVersion) $("status").textContent = `${error.message}。请刷新页面重试。`;
  }
}

function renderHeader() {
  const table = state.tables[state.query], data = state.data;
  const query = state.catalog.queries.find(q => q.id === state.query);
  $("query-title").textContent = table.label;
  $("query-id").textContent = state.query;
  $("query-shape").textContent = `${table.rows.length} 行 × ${table.columns.length} 列 · ${table.source}`;
  $("query-kind").innerHTML = `<span class="badge ${query.kind === "implicit" ? "gold" : "dark"}">${query.kind === "implicit" ? "隐式 join · 需要证据恢复属性" : "显式 join · 已有可见连接列"}</span>`;
  $("query-columns").innerHTML = table.columns.map(c => `<span class="chip">${esc(c)}</span>`).join("") +
    table.hidden.map(c => `<span class="chip hidden-attr">待恢复 · ${esc(c)}</span>`).join("");
  const goldIds = new Set(data.gold.flatMap(g => g.evidence_ids));
  const hits = data.evidence.filter(e => goldIds.has(e.id)).length;
  const goldPairs = data.gold.filter(g => g.evidence_ids.length);
  const retained = goldPairs.filter(g => data.targets[g.target_id].bag.some(e => g.evidence_ids.includes(e))).length;
  const admitted = data.gold.filter(g => data.targets[g.target_id].c150_rank != null).length;
  $("stats").innerHTML = [
    ["首跳召回素材",data.evidence.length,`${data.evidence.filter(e=>e.modality==="text").length} 文本 / ${data.evidence.filter(e=>e.modality==="image").length} 图片`],
    ["Gold 素材首跳命中",`${hits}<span class="unit">/ ${goldIds.size}</span>`, "当前 Query 的标注证据"],
    ["正确证据被 D1 保留",`${retained}<span class="unit">/ ${goldPairs.length}</span>`, "带标注证据的正例对"],
    ["Gold target 进入 C150",`${admitted}<span class="unit">/ ${data.gold.length}</span>`, "候选截断后保留的正例"]
  ].map(([label,value,detail])=>`<div class="stat"><small>${label}</small><strong>${value}</strong><small>${detail}</small></div>`).join("");
}

function selectedEvidence() {
  return state.data.evidence.find(e => e.id === state.evidence) || (state.assets[state.evidence] ?
    {id: state.evidence, modality: state.assets[state.evidence].type, rank: null, score: null, targets: [],
     gold_target_ids: state.data.gold.filter(g=>g.evidence_ids.includes(state.evidence)).map(g=>g.target_id)} : null);
}

function renderEvidenceList(adjust = false) {
  const search = $("evidence-search").value.toLowerCase();
  const visible = state.data.evidence.filter(e => (state.modality === "all" || e.modality === state.modality) &&
    (!$("gold-evidence").checked || e.gold_target_ids.length) && `${e.id} ${state.assets[e.id]?.content || ""}`.toLowerCase().includes(search));
  const missingGold = !state.data.evidence.some(e => e.id === state.evidence) && state.data.gold.some(g => g.evidence_ids.includes(state.evidence));
  if (adjust && visible.length && !visible.some(e => e.id === state.evidence) && !missingGold) state.evidence = visible[0].id;
  if (adjust && !visible.length && !missingGold) state.evidence = null;
  $("evidence-count").textContent = `${visible.length} / ${state.data.evidence.length}`;
  $("evidence-list").innerHTML = visible.map(e => {
    const asset = state.assets[e.id];
    const retained = e.targets.filter(t=>t.retained).length;
    return `<button class="evidence-card ${e.id===state.evidence ? "active" : ""}" data-evidence="${esc(e.id)}" aria-pressed="${e.id===state.evidence}">
      ${e.modality === "image" ? `<img class="thumb" loading="lazy" src="${imageUrl(e.id)}" alt="素材 ${esc(e.id)}">` : '<div class="thumb text-thumb">Aa</div>'}
      <div class="e-card-body"><div class="e-card-top"><strong>#${e.rank} · ${e.modality === "image" ? "图片" : "文本"}</strong><span>${num(e.score)}</span></div>
      <div class="e-snippet">${esc(e.modality === "text" ? asset.content : asset.source || "书籍图片素材")}</div>
      <code>${esc(e.id)}</code><div class="badges">${e.gold_target_ids.length ? '<span class="badge gold">Gold 素材</span>' : ""}
      <span class="badge">${e.targets.length} targets</span>${retained ? `<span class="badge dark">D1 保留 ${retained}</span>` : ""}</div></div></button>`;
  }).join("") || '<div class="empty">当前筛选下没有素材。<br>可到「Gold 目标与证据」查看未命中的标注素材。</div>';
  renderEvidenceDetail();
  renderTargets();
  updateURL();
}

function renderEvidenceDetail() {
  const e = selectedEvidence();
  if (!e) {$("evidence-detail").innerHTML = '<div class="empty">选择一条素材查看第二跳</div>';return;}
  const asset = state.assets[e.id];
  const facts = `<div class="detail-facts">首跳排名 <strong>${e.rank == null ? "未进入首跳 top20" : `${e.modality==="image"?"图片":"文本"} #${e.rank}`}</strong><br>Query → 素材分数 <strong>${num(e.score)}</strong><br>来源 <strong>${esc(asset.source)}</strong><br>原表 <strong>${esc(asset.source_table)}</strong> · 行 <strong>${esc(asset.source_row)}</strong></div>`;
  $("evidence-detail").innerHTML = `<span class="eyebrow">SELECTED EVIDENCE</span><div class="detail-top"><code>${esc(e.id)}</code>${e.gold_target_ids.length?'<span class="badge gold">当前 Query 的 Gold 素材</span>':""}</div>
    ${e.modality === "image" ? `<div class="detail-image"><img src="${imageUrl(e.id)}" alt="选中的图片素材" data-open-asset="${esc(e.id)}" tabindex="0">${facts}</div>` : `<div class="detail-text">${esc(asset.content)}</div>${facts}`}
    <div class="detail-bottom"><div class="badges"><span class="badge">${e.targets.length} 个第二跳 target</span>${e.targets.some(t=>t.gold_path)?'<span class="badge teal">存在正确证据链</span>':""}</div><button class="button" data-open-asset="${esc(e.id)}">${e.modality==="image"?"查看原图":"查看全文"} ↗</button></div>`;
}

function targetCard(edge, mode = "evidence") {
  const t = state.tables[edge.id], meta = state.data.targets[edge.id];
  const c = meta.c150_rank == null ? "未入 C150" : `C150 #${meta.c150_rank}`;
  const scores = mode === "evidence" ? `素材 → T <b>${num(edge.score)}</b><span>路径 <b>${num(edge.path_score)}</b></span>` : `直接匹配 <b>${num(meta.direct_score)}</b>`;
  return `<div class="target-card" role="button" tabindex="0" data-target="${esc(edge.id)}"><span class="rank">${edge.rank ? String(edge.rank).padStart(2,"0") : "G"}</span><div class="target-content">
    <div class="target-title">${esc(t.label)}</div><div class="target-id">${esc(edge.id)} · ${t.rows.length} 行 / ${t.columns.length} 列</div>
    <div class="target-cols">${esc(t.columns.join(" · "))}</div><div class="badges">${meta.gold?'<span class="badge gold">Gold target</span>':""}${edge.gold_path?'<span class="badge teal">正确证据链</span>':""}${edge.retained?'<span class="badge dark">D1 保留</span>':""}<span class="badge">${c}</span></div>
    <div class="target-scores"><span>${scores}</span>${state.data.teacher_available?`<span>Teacher <b>${rank(meta.teacher_rank)}</b></span>`:""}</div></div><span class="target-open">↗</span></div>`;
}

function renderTargets() {
  const e = selectedEvidence();
  const filter = $("target-filter").value, search = $("target-search").value.toLowerCase();
  const edges = (e?.targets || []).filter(edge => {
    const meta = state.data.targets[edge.id], t = state.tables[edge.id];
    return (filter === "all" || filter === "gold" && meta.gold || filter === "gold_path" && edge.gold_path ||
      filter === "retained" && edge.retained || filter === "c150" && meta.c150_rank != null) &&
      `${t.id} ${t.label} ${t.columns.join(" ")} ${t.rows.flat().join(" ")}`.toLowerCase().includes(search);
  });
  $("target-count").textContent = `${edges.length} / ${e?.targets.length || 0}`;
  $("target-list").innerHTML = edges.map(edge=>targetCard(edge)).join("") ||
    `<div class="empty">${!e ? "选择一条素材查看第二跳结果。" : e.rank == null ? "这条标注素材未被首跳召回，因此没有自然第二跳结果。" : "没有符合当前筛选条件的 Target。"}</div>`;
}

function renderDirect() {
  $("direct-view").innerHTML = `<div class="eyebrow">QUERY → TARGET</div><h2>直接召回的前 ${state.data.direct.length} 张表</h2><p>这里展示 Query 直接检索 Target 的排序。点击任意结果查看整张表；此列表未经过素材路径融合。</p><div class="target-list">${state.data.direct.map((id,i)=>targetCard({id,rank:i+1},"direct")).join("")}</div>`;
}

function renderGold() {
  $("gold-view").innerHTML = `<div class="eyebrow">GROUND TRUTH</div><h2>应该召回的目标与标注证据</h2><p>“Gold target”表示目标表正确；“正确证据链”要求该素材同时是当前 Query–Target 对的 gold evidence。未标注为 gold 不等于已验证无关。</p>` +
    state.data.gold.map(g => {
      const meta = state.data.targets[g.target_id];
      const first = g.evidence_ids.filter(id=>state.data.evidence.some(e=>e.id===id));
      const second = g.evidence_ids.filter(id=>state.data.evidence.find(e=>e.id===id)?.targets.some(t=>t.id===g.target_id));
      const retained = g.evidence_ids.filter(id=>meta.bag.includes(id));
      return `<div class="gold-block">${targetCard({id:g.target_id},"direct")}<div class="gold-status">Join 列 <strong>${esc(g.join_column)}</strong> · ${g.reason==="model_recoverable_join_column"?"需要证据恢复":"Query 已可见"}<br>
      标注素材 <strong>${g.evidence_ids.length}</strong> → 首跳命中 <strong>${first.length}</strong> → 连到正确 Target <strong>${second.length}</strong> → D1 保留 <strong>${retained.length}</strong><br>
      ${meta.c150_rank == null?"目标未进入 C150，无法进入后续 Teacher 排序。":`目标进入 C150 第 ${meta.c150_rank} 名。${state.data.teacher_available?` Teacher Real：${rank(meta.teacher_rank)}；直接评分 f0：${rank(meta.f0_rank)}。`:""}`}</div>
      <div class="gold-e-list">${g.evidence_ids.map(id=>`<button data-gold-evidence="${esc(id)}">${state.assets[id]?.type==="image"?"▧":"Aa"} ${esc(id)} · ${first.includes(id)?"已召回":"未召回"} ↗</button>`).join("") || '<span class="recovery-note">该正例没有标注 gold evidence。</span>'}</div>
      ${g.recoveries.length?`<p class="recovery-note">标注的可恢复值：${esc([...new Set(g.recoveries.map(r=>`行 ${r.row} · ${r.attribute} = ${r.value}`))].join("；"))}</p>`:""}</div>`;
    }).join("");
}

function setTab(tab) {
  if (!["paths","direct","gold"].includes(tab)) tab = "paths";
  state.tab = tab;
  document.querySelectorAll("[data-tab]").forEach(el => {el.classList.toggle("active",el.dataset.tab===tab);el.setAttribute("aria-selected",el.dataset.tab===tab);});
  for (const name of ["paths","direct","gold"]) $(`${name}-view`).hidden = name !== tab;
  updateURL();
}

function openTable(id) {
  const table = state.tables[id];
  if (!table) return;
  const meta = state.data.targets[id];
  const join = state.data.gold.find(g=>g.target_id===id)?.join_column || table.join_column;
  $("modal-kicker").textContent = id === state.query ? "QUERY · 可见表格内容" : "TARGET · 完整表格";
  $("modal-title").textContent = table.label;
  $("modal-body").innerHTML = `<div class="modal-meta"><code>${esc(id)}</code><span>${esc(table.source)} · ${table.rows.length} 行 × ${table.columns.length} 列</span>${join?`<span>连接列 <strong>${esc(join)}</strong>（浅黄色）</span>`:""}</div>
    ${meta?`<div class="gold-status">${meta.gold?"Gold target · ":""}直接 ANN ${rank(meta.direct_rank)} · C150 ${rank(meta.c150_rank)} · ${state.data.teacher_available?`Teacher Real ${rank(meta.teacher_rank)} · f0 ${rank(meta.f0_rank)}`:"该视图没有 Teacher 评测排名"}</div><br>`:""}
    <div class="table-scroll"><table><thead><tr><th class="row-num">行</th>${table.columns.map(c=>`<th class="${c===join?"join-cell":""}">${esc(c)}</th>`).join("")}</tr></thead><tbody>${table.rows.map((row,i)=>`<tr><td class="row-num">${esc(table.row_ids[i])}</td>${row.map((cell,j)=>`<td class="${table.columns[j]===join?"join-cell":""}">${cell==null||cell===""?'<span style="color:#bac4bc">∅</span>':esc(cell)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  $("modal").showModal();
}

function openAsset(id) {
  const a = state.assets[id];
  $("modal-kicker").textContent = a.type === "image" ? "EVIDENCE · 原始图片" : "EVIDENCE · 完整文本";
  $("modal-title").textContent = id;
  $("modal-body").innerHTML = `<div class="modal-meta">${esc(a.source)} · ${esc(a.source_table)} · 行 ${esc(a.source_row)}</div>` +
    (a.type === "image" ? `<img class="full-image" src="${imageUrl(id)}" alt="${esc(id)}">` : `<div class="full-text">${esc(a.content)}</div>`);
  $("modal").showModal();
}

document.addEventListener("click", event => {
  const el = event.target.closest("[data-query],[data-evidence],[data-target],[data-tab],[data-open-asset],[data-gold-evidence]");
  if (!el) return;
  if (el.dataset.query) {state.query=el.dataset.query;state.evidence=null;loadQuery();}
  else if (el.dataset.evidence) {state.evidence=el.dataset.evidence;renderEvidenceList();}
  else if (el.dataset.target) openTable(el.dataset.target);
  else if (el.dataset.tab) setTab(el.dataset.tab);
  else if (el.dataset.openAsset) openAsset(el.dataset.openAsset);
  else if (el.dataset.goldEvidence) {state.evidence=el.dataset.goldEvidence;setTab("paths");renderEvidenceList();$("evidence-detail").scrollIntoView({behavior:"smooth",block:"center"});}
});
document.addEventListener("keydown", event => {
  if ((event.key === "Enter" || event.key === " ") && event.target.matches('[role="button"],img[data-open-asset]')) {event.preventDefault();event.target.click();}
});
for (const [id,key] of [["arm","arm"],["split","split"],["generator","generator"]]) $(id).addEventListener("change",()=>{state[key]=$(id).value;loadQuery();});
$("query-search").addEventListener("input",renderQueries);
$("evidence-search").addEventListener("input",()=>renderEvidenceList(true));
$("gold-evidence").addEventListener("change",()=>renderEvidenceList(true));
$("modality").addEventListener("click", event=>{const button=event.target.closest("button");if(!button)return;state.modality=button.dataset.value;$("modality").querySelectorAll("button").forEach(b=>b.classList.toggle("active",b===button));renderEvidenceList(true);});
$("target-filter").addEventListener("change",renderTargets);
$("target-search").addEventListener("input",renderTargets);
$("query-table").addEventListener("click",()=>openTable(state.query));
$("modal-close").addEventListener("click",()=>$("modal").close());
$("modal").addEventListener("click",event=>{if(event.target===$("modal")){const r=$("modal").getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)$("modal").close();}});
$("share").addEventListener("click",async()=>{try{await navigator.clipboard.writeText(location.href);toast("已复制当前 Query 和素材的访问链接");}catch{const input=document.createElement("textarea");input.value=location.href;document.body.appendChild(input);input.select();const ok=document.execCommand("copy");input.remove();if(ok)toast("已复制当前视图链接");else window.prompt("复制当前视图链接",location.href);}});

(async()=>{
  try {
    [state.catalog,state.assets]=await Promise.all([getJSON("/data/catalog.json.gz"),getJSON("/data/assets.json.gz")]);
    if (!Object.hasOwn(state.catalog.arms,state.arm)) state.arm="baseline";
    if (!["test","dev","train"].includes(state.split)) state.split="test";
    await loadQuery();
  } catch(error) {$("status").textContent=`页面数据未能加载：${error.message}`;}
})();
