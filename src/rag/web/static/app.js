// Dondoo front end. All server text is inserted with textContent, never innerHTML, so a document
// or an answer cannot inject markup or scripts into the page.
"use strict";

// Older browsers (Safari < 14) lack replaceChildren; add a small fallback.
if (!Element.prototype.replaceChildren) {
  Element.prototype.replaceChildren = function (...nodes) {
    while (this.firstChild) this.removeChild(this.firstChild);
    nodes.forEach((n) => this.appendChild(typeof n === "string" ? document.createTextNode(n) : n));
  };
}

const I18N = {
  en: {
    theme_dark: "Switch to dark mode", theme_light: "Switch to light mode",
    status_checking: "Checking…", status_ok: "Online", status_bad: "Degraded", status_off: "Offline",
    eyebrow: "Grounded answers · enforced citations",
    h1a: "Ask the documents.", h1b: "See exactly where the answer came from.",
    lede: "Every claim links to a page in a public source. If the sources don't say it, Dondoo says so instead of guessing.",
    question_label: "Your question",
    placeholder: "e.g. Within how many hours must a data breach be reported to the Data Commissioner?",
    ask: "Ask", examples: "Examples",
    library: "Library", library_sub: (n) => `${n} document${n === 1 ? "" : "s"} Dondoo can cite`,
    lib_empty: "No documents indexed yet.", lib_err: "Could not load the library.",
    passages: (n) => `${n} passages`,
    how: "How it works",
    p1t: "Hybrid search", p1: "keyword and vector search, merged by rank",
    p2t: "Rerank", p2: "a cross-encoder scores each passage against the question",
    p3t: "Generate", p3: "from the top passages only",
    p4t: "Verify", p4: "every citation must point to a retrieved passage",
    p5t: "Decline", p5: "when no passage is relevant enough",
    footer: "Answers come from public documents and may be incomplete. Check the cited source before relying on it.",
    steps: ["searching", "reranking", "writing", "checking citations"],
    answer: "Answer", badge_declined: "not in sources",
    badge_cites: (n) => `${n} citation${n === 1 ? "" : "s"}`,
    sources: "Sources", open: "Open ↗", page: (n) => `page ${n}`, retrieved: (d) => `retrieved ${d}`,
    untitled: "Untitled document", show_source: (n) => `Show source ${n}`,
    m_total: "total", m_retrieval: "retrieval", m_generation: "generation", m_passages: "passages",
    err_429: "Too many questions in a minute. Please wait a moment and try again.",
    err_http: (s) => `The server returned an error (${s}). Please try again.`,
    err_net: "Could not reach the server. It may be waking up; try again in 30 seconds.",
    sample: [
      "Within how many hours must a data controller notify the Data Commissioner of a personal data breach?",
      "What are the principles of data protection?",
      "What rights does a data subject have?",
      "Who must register as a data controller or processor?",
      "What is the capital of France?",
    ],
  },
  sw: {
    theme_dark: "Badilisha kuwa hali ya giza", theme_light: "Badilisha kuwa hali ya mwanga",
    status_checking: "Inakagua…", status_ok: "Mtandaoni", status_bad: "Kuna hitilafu", status_off: "Nje ya mtandao",
    eyebrow: "Majibu yenye ushahidi · marejeo ya lazima",
    h1a: "Uliza nyaraka.", h1b: "Ona jibu limetoka wapi hasa.",
    lede: "Kila dai linaunganishwa na ukurasa katika chanzo cha umma. Kama vyanzo havisemi hivyo, Dondoo itakwambia badala ya kubahatisha.",
    question_label: "Swali lako",
    placeholder: "mf. Ndani ya saa ngapi uvujaji wa data lazima uripotiwe kwa Kamishna wa Data?",
    ask: "Uliza", examples: "Mifano",
    library: "Maktaba", library_sub: (n) => `Nyaraka ${n} ambazo Dondoo inaweza kunukuu`,
    lib_empty: "Bado hakuna nyaraka.", lib_err: "Maktaba haikuweza kupakiwa.",
    passages: (n) => `vifungu ${n}`,
    how: "Jinsi inavyofanya kazi",
    p1t: "Utafutaji mseto", p1: "utafutaji wa maneno na wa maana, vikiunganishwa kwa mpangilio",
    p2t: "Kupanga upya", p2: "modeli hupima kila kifungu dhidi ya swali",
    p3t: "Kuandika jibu", p3: "kutoka kwa vifungu bora pekee",
    p4t: "Kuhakiki", p4: "kila rejeo lazima lielekeze kwenye kifungu kilichopatikana",
    p5t: "Kukataa", p5: "pale ambapo hakuna kifungu kinachohusika vya kutosha",
    footer: "Majibu yanatoka kwenye nyaraka za umma na huenda yasiwe kamili. Kagua chanzo kilichonukuliwa kabla ya kukitegemea.",
    steps: ["inatafuta", "inapanga", "inaandika", "inahakiki marejeo"],
    answer: "Jibu", badge_declined: "haipo kwenye vyanzo",
    badge_cites: (n) => `marejeo ${n}`,
    sources: "Vyanzo", open: "Fungua ↗", page: (n) => `ukurasa ${n}`, retrieved: (d) => `ilipakuliwa ${d}`,
    untitled: "Hati isiyo na jina", show_source: (n) => `Onyesha chanzo ${n}`,
    m_total: "jumla", m_retrieval: "utafutaji", m_generation: "uandishi", m_passages: "vifungu",
    err_429: "Maswali ni mengi mno kwa dakika moja. Subiri kidogo kisha ujaribu tena.",
    err_http: (s) => `Seva imerudisha hitilafu (${s}). Tafadhali jaribu tena.`,
    err_net: "Seva haipatikani. Huenda inaamka; jaribu tena baada ya sekunde 30.",
    sample: [
      "Ndani ya saa ngapi mdhibiti wa data lazima amjulishe Kamishna wa Data kuhusu uvujaji wa data binafsi?",
      "Kanuni za ulinzi wa data ni zipi?",
      "Mhusika wa data ana haki gani?",
      "Nani lazima ajisajili kama mdhibiti au mchakataji wa data?",
      "Mji mkuu wa Ufaransa ni upi?",
    ],
  },
};

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

// Browser storage can be unavailable (private mode, some embedded views), so every use is guarded.
function loadLang() {
  try { const v = localStorage.getItem("dondoo-lang"); if (v === "en" || v === "sw") return v; } catch { /* ignore */ }
  return (navigator.language || "").toLowerCase().startsWith("sw") ? "sw" : "en";
}
function saveLang(v) { try { localStorage.setItem("dondoo-lang", v); } catch { /* ignore */ } }

let lang = loadLang();
const t = (key, ...args) => {
  const v = I18N[lang][key];
  return typeof v === "function" ? v(...args) : v;
};

const form = $("#ask-form");
const input = $("#question");
const btn = $("#ask-btn");
const result = $("#result");
let inFlight = null;
let lastData = null;
let libraryDocs = null;
let healthState = "checking";

function safeUrl(url) {
  try {
    const u = new URL(url);
    return u.protocol === "https:" || u.protocol === "http:" ? u.href : null;
  } catch { return null; }
}

function applyLanguage() {
  document.documentElement.lang = lang;
  document.querySelectorAll("[data-i18n]").forEach((node) => { node.textContent = t(node.dataset.i18n); });
  input.placeholder = t("placeholder");
  document.querySelectorAll(".lang-btn").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.lang === lang)));
  renderExamples();
  renderStatus();
  renderLibrary();
  renderThemeButton();
  if (lastData) renderResult(lastData);
}

function currentTheme() { return document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light"; }
function renderThemeButton() {
  const b = $("#theme-btn");
  const label = currentTheme() === "dark" ? t("theme_light") : t("theme_dark");
  b.setAttribute("aria-label", label);
  b.title = label;
}
function toggleTheme() {
  const next = currentTheme() === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  try { localStorage.setItem("dondoo-theme", next); } catch { /* ignore */ }
  renderThemeButton();
}

function setLang(v) {
  if (v === lang) return;
  lang = v;
  saveLang(v);
  applyLanguage();
}

function renderExamples() {
  const box = $("#examples");
  box.replaceChildren();
  t("sample").forEach((q) => {
    const chip = el("button", "chip", q);
    chip.type = "button";
    chip.addEventListener("click", () => { input.value = q; updateCounter(); ask(q); });
    box.appendChild(chip);
  });
}

function updateCounter() { $("#counter").textContent = `${input.value.length} / 1000`; }

function renderStatus() {
  const status = $("#status");
  status.classList.toggle("ok", healthState === "ok");
  status.classList.toggle("bad", healthState === "bad" || healthState === "off");
  const key = { checking: "status_checking", ok: "status_ok", bad: "status_bad", off: "status_off" }[healthState];
  $("#status-text").textContent = t(key);
}

async function checkHealth() {
  try {
    const r = await fetch("/api/health");
    healthState = r.ok ? "ok" : "bad";
  } catch { healthState = "off"; }
  renderStatus();
}

function renderLibrary() {
  const list = $("#library");
  const sub = $("#library-sub");
  list.replaceChildren();
  if (libraryDocs === null) { sub.textContent = ""; return; }
  if (libraryDocs === "error") { list.appendChild(el("li", "empty", t("lib_err"))); sub.textContent = ""; return; }
  sub.textContent = t("library_sub", libraryDocs.length);
  if (!libraryDocs.length) { list.appendChild(el("li", "empty", t("lib_empty"))); return; }
  libraryDocs.forEach((d) => {
    const li = el("li");
    const title = el("div", "lib-title");
    const href = safeUrl(d.url);
    if (href) {
      const a = el("a", null, d.title);
      a.href = href; a.target = "_blank"; a.rel = "noopener noreferrer";
      title.appendChild(a);
    } else { title.textContent = d.title; }
    li.appendChild(title);
    li.appendChild(el("div", "lib-sub", [d.publisher, d.published, t("passages", d.chunks)].filter(Boolean).join(" · ")));
    list.appendChild(li);
  });
}

async function loadLibrary() {
  try {
    const r = await fetch("/api/documents");
    libraryDocs = r.ok ? await r.json() : "error";
  } catch { libraryDocs = "error"; }
  renderLibrary();
}

function showLoading() {
  result.hidden = false;
  const sk = el("div", "skeleton");
  sk.append(el("div", "bar"), el("div", "bar"), el("div", "bar"));
  const steps = el("div", "steps");
  const spans = t("steps").map((n) => el("span", null, n));
  steps.append(...spans);
  sk.appendChild(steps);
  result.replaceChildren(sk);
  let i = 0;
  spans[0].classList.add("on");
  return setInterval(() => { i = Math.min(i + 1, spans.length - 1); spans[i].classList.add("on"); }, 700);
}

function flashSource(marker) {
  const card = document.getElementById(`src-${marker}`);
  if (!card) return;
  const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  card.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "center" });
  card.setAttribute("tabindex", "-1");
  card.focus({ preventScroll: true });
  card.classList.remove("flash");
  void card.offsetWidth;
  card.classList.add("flash");
  setTimeout(() => card.classList.remove("flash"), 1600);
}

// Turn "text [S1] more [S2, S3]" into text nodes plus clickable citation buttons.
function renderAnswerText(text, validMarkers) {
  const p = el("p", "answer-text");
  const re = /\[((?:S\d+)(?:\s*[,;]\s*S\d+)*)\]/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) p.appendChild(document.createTextNode(text.slice(last, m.index)));
    m[1].split(/\s*[,;]\s*/).forEach((marker) => {
      if (validMarkers.has(marker)) {
        const b = el("button", "cite", marker.slice(1));
        b.type = "button";
        b.title = t("show_source", marker.slice(1));
        b.addEventListener("click", () => flashSource(marker));
        p.appendChild(b);
      }
    });
    last = re.lastIndex;
  }
  if (last < text.length) p.appendChild(document.createTextNode(text.slice(last)));
  return p;
}

function renderResult(data) {
  const frag = document.createDocumentFragment();
  const card = el("div", data.declined ? "answer-card declined-card" : "answer-card");
  const label = el("div", "answer-label", t("answer"));
  label.appendChild(el("span", data.declined ? "badge declined" : "badge grounded",
    data.declined ? t("badge_declined") : t("badge_cites", data.citations.length)));
  card.appendChild(label);

  const valid = new Set(data.citations.map((c) => c.marker));
  card.appendChild(data.declined ? el("p", "answer-text", data.answer) : renderAnswerText(data.answer, valid));
  (data.warnings || []).forEach((w) => card.appendChild(el("div", "warning", `⚠ ${w}`)));

  const metrics = el("div", "metrics");
  const tm = data.timings_ms || {};
  const add = (k, v) => { const m = el("span", "metric"); m.append(`${k} `); m.appendChild(el("b", null, v)); metrics.appendChild(m); };
  if (tm.total !== undefined) add(t("m_total"), `${Math.round(tm.total)} ms`);
  if (tm.retrieve !== undefined) add(t("m_retrieval"), `${Math.round(tm.retrieve)} ms`);
  if (tm.llm !== undefined) add(t("m_generation"), `${Math.round(tm.llm)} ms`);
  add(t("m_passages"), String((data.retrieved_chunk_ids || []).length));
  card.appendChild(metrics);
  frag.appendChild(card);

  if (data.citations.length) {
    frag.appendChild(el("h2", "sources-title", t("sources")));
    const tpl = $("#tpl-source");
    data.citations.forEach((c) => {
      const node = tpl.content.firstElementChild.cloneNode(true);
      node.id = `src-${c.marker}`;
      node.querySelector(".source-marker").textContent = c.marker.slice(1);
      node.querySelector(".source-title").textContent = c.title || t("untitled");
      const sub = [c.publisher, c.published, t("page", c.page), c.retrieved ? t("retrieved", c.retrieved) : ""].filter(Boolean);
      node.querySelector(".source-sub").textContent = sub.join(" · ");
      const link = node.querySelector(".source-link");
      link.textContent = t("open");
      const href = safeUrl(c.url);
      if (href) { link.href = href; } else { link.hidden = true; }
      node.querySelector(".source-quote").textContent = `“${c.quote}${c.quote.length >= 400 ? "…" : ""}”`;
      frag.appendChild(node);
    });
  }
  result.hidden = false;
  result.replaceChildren(frag);
}

function renderError(message) {
  lastData = null;
  result.hidden = false;
  result.replaceChildren(el("div", "error-card", message));
}

async function ask(question) {
  question = question.trim();
  if (question.length < 3) return;
  if (inFlight) inFlight.abort();
  inFlight = new AbortController();
  btn.disabled = true;
  const timer = showLoading();
  try {
    const r = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, lang }),
      signal: inFlight.signal,
    });
    if (r.status === 429) { renderError(t("err_429")); return; }
    if (!r.ok) { renderError(t("err_http", r.status)); return; }
    lastData = await r.json();
    renderResult(lastData);
    const url = new URL(location.href);
    url.searchParams.set("q", question);
    history.replaceState(null, "", url);
  } catch (e) {
    if (e.name !== "AbortError") renderError(t("err_net"));
  } finally {
    clearInterval(timer);
    btn.disabled = false;
  }
}

form.addEventListener("submit", (e) => { e.preventDefault(); ask(input.value); });
input.addEventListener("input", updateCounter);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); ask(input.value); }
});
document.querySelectorAll(".lang-btn").forEach((b) => b.addEventListener("click", () => setLang(b.dataset.lang)));
$("#theme-btn").addEventListener("click", toggleTheme);

// Links may set the language: /?lang=sw
const params = new URLSearchParams(location.search);
if (params.get("lang") === "en" || params.get("lang") === "sw") { lang = params.get("lang"); }

applyLanguage();
// Enable colour animations only after the first paint, so the page never loads mid-fade.
requestAnimationFrame(() => requestAnimationFrame(() => document.body.classList.add("ready")));
checkHealth();
loadLibrary();

// Shareable links: /?q=your+question opens the page and asks it.
const shared = params.get("q");
if (shared && shared.length <= 1000) { input.value = shared; updateCounter(); ask(shared); }
