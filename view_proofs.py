"""Local browser + annotation tool for the DeepTheorem step-numbered proof subset.

For each proof you can:
  - read the original (rendered LaTeX)
  - edit a working copy of the proof text (e.g. delete a step, inject an error)
  - save an annotation:
      SUFFICIENT: 1|0
      ERROR_TYPE: mistake|gap|truncation|none
      LOCALIZATION: step number, or N/A

Rows are optionally tagged with a `task_type` (positive|mistake|gap|truncation)
from a sample manifest built by build_annotation_sample.py (default
./Data/annotation_sample.json), keyed by dataset `id`. The detail view shows
which task each sampled row was assigned and pre-fills sensible defaults.

Annotations are stored keyed by dataset `id` in a JSON file (default
./Data/proof_annotations.json) and are separate from the original dataset,
so re-running the tool resumes exactly where you left off.

Usage:
    python view_proofs.py [--dataset PATH] [--annotations PATH] [--sample PATH] [--port 5000]

Then open http://localhost:5000 in a browser.
"""
import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from datasets import load_from_disk

STEP_RE = re.compile(r"Step\s*\d+", re.IGNORECASE)
LIST_RE = re.compile(r"(?m)^\s*\d+\.\s")

ANNOTATION_LOCK = threading.Lock()


def classify(proof: str) -> str:
    has_step = bool(STEP_RE.search(proof))
    has_list = len(LIST_RE.findall(proof)) >= 2
    if has_step and has_list:
        return "both"
    if has_step:
        return "step"
    if has_list:
        return "list"
    return "none"


def load_rows(dataset_path: str, sample: dict):
    dt = load_from_disk(dataset_path)
    if hasattr(dt, "keys"):
        dt = dt["train"]
    rows = []
    for idx in range(len(dt)):
        ex = dt[idx]
        proof = ex["proof"]
        row_id = ex.get("id")
        rows.append(
            {
                "idx": idx,
                "id": row_id,
                "domain": ex.get("domain"),
                "difficulty": ex.get("difficulty"),
                "truth_value": ex.get("truth_value"),
                "source": ex.get("source"),
                "pattern": classify(proof),
                "informal_theorem_qa": ex.get("informal_theorem_qa"),
                "proof": proof,
                "task_type": sample.get(str(row_id)),
            }
        )
    return rows


def load_json(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


load_annotations = load_json


def save_annotations(path: str, data: dict):
    with ANNOTATION_LOCK:
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=1)
        import os
        os.replace(tmp_path, path)


INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>DeepTheorem Proof Annotator</title>
<script>
window.MathJax = {
  tex: {
    inlineMath: [['\\\\(', '\\\\)'], ['$', '$']],
    displayMath: [['\\\\[', '\\\\]'], ['$$', '$$']]
  },
  options: { skipHtmlTags: ['script','noscript','style','textarea','pre','code'] }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" defer></script>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; display: flex; height: 100vh; }
  #sidebar { width: 340px; flex-shrink: 0; border-right: 1px solid #8884; padding: 12px; overflow-y: auto; box-sizing: border-box; }
  #main { flex: 1; overflow-y: auto; padding: 16px 24px; }
  h1 { font-size: 15px; margin: 0 0 12px; }
  label { display: block; font-size: 12px; margin: 10px 0 3px; opacity: 0.75; }
  select, input, textarea, button { width: 100%; box-sizing: border-box; padding: 5px 6px; font-size: 13px; font-family: inherit; }
  button { margin-top: 4px; cursor: pointer; }
  .row-item { border: 1px solid #8883; border-radius: 6px; padding: 8px 10px; margin-bottom: 8px; cursor: pointer; position: relative; }
  .row-item:hover { background: #8881; }
  .row-item.active { border-color: #4a90d9; background: #4a90d922; }
  .meta { font-size: 11px; opacity: 0.7; margin-bottom: 4px; }
  .badge { display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 10px; margin-right: 4px; }
  .badge.step { background: #4a90d955; }
  .badge.list { background: #d9944a55; }
  .badge.both { background: #6ad96a55; }
  .badge.true { background: #4ad97355; }
  .badge.false { background: #d94a4a55; }
  .badge.done { background: #6ad96a; color: #063; font-weight: bold; }
  .badge.tasktype-positive { background: #4ad97355; }
  .badge.tasktype-mistake { background: #d94a4a55; }
  .badge.tasktype-gap { background: #d9a94a55; }
  .badge.tasktype-truncation { background: #9a4ad955; }
  #task_banner { padding: 8px 12px; border-radius: 6px; font-weight: bold; margin-bottom: 12px; }
  #task_banner.positive { background: #4ad97333; border: 1px solid #4ad973; }
  #task_banner.mistake { background: #d94a4a33; border: 1px solid #d94a4a; }
  #task_banner.gap { background: #d9a94a33; border: 1px solid #d9a94a; }
  #task_banner.truncation { background: #9a4ad933; border: 1px solid #9a4ad9; }
  #task_banner.excluded { background: #8884; border: 1px solid #888; text-decoration: line-through; }
  #eligibility_bar { margin-top: 10px; padding: 10px 12px; border: 1px dashed #8888; border-radius: 6px; }
  .badge.excluded { background: #8888; color: #fff; font-weight: bold; }
  .preview { font-size: 12px; opacity: 0.85; white-space: pre-wrap; max-height: 3.6em; overflow: hidden; }
  #pager { display: flex; justify-content: space-between; align-items: center; margin-top: 10px; font-size: 12px; }
  #detail h2 { font-size: 16px; }
  .field-label { font-size: 12px; text-transform: uppercase; opacity: 0.6; margin-top: 18px; }
  .proof-text, .question-text { white-space: pre-wrap; line-height: 1.5; }
  #status { font-size: 12px; opacity: 0.7; margin-top: 8px; }
  .cols { display: flex; gap: 20px; }
  .col { flex: 1; min-width: 0; }
  textarea#edited_proof { height: 320px; font-family: ui-monospace, monospace; font-size: 12px; white-space: pre; }
  .annot-row { display: flex; gap: 12px; }
  .annot-row > div { flex: 1; }
  #save_bar { margin-top: 16px; padding: 12px; border: 1px solid #8884; border-radius: 6px; }
  #save_msg { font-size: 12px; margin-left: 10px; }
</style>
</head>
<body>
<div id="sidebar">
  <h1>DeepTheorem Proof Annotator</h1>
  <label>Search (proof / question text)</label>
  <input id="q" placeholder="e.g. Hilbert space">
  <label>Domain</label>
  <select id="domain"><option value="">(any)</option></select>
  <label>Pattern type</label>
  <select id="pattern">
    <option value="">(any)</option>
    <option value="step">Step N</option>
    <option value="list">Numbered list</option>
    <option value="both">Both</option>
  </select>
  <label>Truth value</label>
  <select id="truth">
    <option value="">(any)</option>
    <option value="true">true</option>
    <option value="false">false</option>
  </select>
  <label>Task type (annotation sample)</label>
  <select id="task_type">
    <option value="">(any row)</option>
    <option value="sampled">in sample (any type)</option>
    <option value="positive">positive</option>
    <option value="mistake">mistake</option>
    <option value="gap">gap</option>
    <option value="truncation">truncation</option>
    <option value="unsampled">not in sample</option>
  </select>
  <label>Annotated</label>
  <select id="annotated">
    <option value="">(any)</option>
    <option value="yes">already annotated</option>
    <option value="no">not yet annotated</option>
  </select>
  <label>Eligibility</label>
  <select id="eligibility">
    <option value="">(any)</option>
    <option value="excluded">excluded (not real steps)</option>
    <option value="not_excluded">not excluded</option>
  </select>
  <label>Min difficulty</label>
  <input id="min_diff" type="number">
  <label>Max difficulty</label>
  <input id="max_diff" type="number">
  <label>Jump to row id</label>
  <input id="jump_id">
  <button onclick="jumpToId()">Go</button>
  <div id="status"></div>
  <div id="list"></div>
  <div id="pager">
    <button onclick="prevPage()">&laquo; prev</button>
    <span id="pageinfo"></span>
    <button onclick="nextPage()">next &raquo;</button>
  </div>
</div>
<div id="main">
  <div id="detail">Select a row on the left.</div>
</div>
<script>
let offset = 0;
const limit = 25;
let total = 0;
let currentRows = [];
let activeIdx = null;
let activeRow = null;

function qs(params) {
  return Object.entries(params).filter(([k,v]) => v !== "" && v !== null && v !== undefined)
    .map(([k,v]) => encodeURIComponent(k) + "=" + encodeURIComponent(v)).join("&");
}

function currentFilters() {
  return {
    q: document.getElementById("q").value,
    domain: document.getElementById("domain").value,
    pattern: document.getElementById("pattern").value,
    truth: document.getElementById("truth").value,
    task_type: document.getElementById("task_type").value,
    annotated: document.getElementById("annotated").value,
    eligibility: document.getElementById("eligibility").value,
    min_diff: document.getElementById("min_diff").value,
    max_diff: document.getElementById("max_diff").value,
  };
}

async function loadFacets() {
  const res = await fetch("/api/facets");
  const data = await res.json();
  const domainSel = document.getElementById("domain");
  data.domains.forEach(d => {
    const opt = document.createElement("option");
    opt.value = d; opt.textContent = d;
    domainSel.appendChild(opt);
  });
}

async function loadRows() {
  const params = Object.assign({ offset, limit }, currentFilters());
  const res = await fetch("/api/rows?" + qs(params));
  const data = await res.json();
  total = data.total;
  currentRows = data.rows;
  renderList();
  document.getElementById("status").textContent = total + " matching rows · sample: " + data.sample_annotated + "/" + data.sample_total + " annotated · " + data.annotated_total + " annotated overall";
  document.getElementById("pageinfo").textContent = (total === 0 ? 0 : offset + 1) + "-" + Math.min(offset + limit, total) + " / " + total;
}

function renderList() {
  const listEl = document.getElementById("list");
  listEl.innerHTML = "";
  currentRows.forEach(r => {
    const div = document.createElement("div");
    div.className = "row-item" + (r.idx === activeIdx ? " active" : "");
    div.onclick = () => showDetail(r.idx);
    div.innerHTML = `<div class="meta">id ${r.id} <span style="opacity:.5">(#${r.idx})</span> · ${(r.domain || []).join(", ")} · diff ${r.difficulty}
      ${r.task_type ? `<span class="badge tasktype-${r.task_type}">${r.task_type}</span>` : ''}
      <span class="badge ${r.pattern}">${r.pattern}</span>
      ${r.excluded ? '<span class="badge excluded">excluded</span>' : (r.has_annotation ? '<span class="badge done">annotated</span>' : '')}</div>
      <div class="preview">${escapeHtml(r.proof.slice(0, 220))}</div>`;
    listEl.appendChild(div);
  });
}

function escapeHtml(s) {
  return s.replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
}

const TASK_INSTRUCTIONS = {
  positive: "TASK: POSITIVE — leave the proof unchanged. Confirm SUFFICIENT=1, ERROR_TYPE=none, LOCALIZATION=N/A and save.",
  mistake: "TASK: MISTAKE — edit the working copy to introduce a logical/computational mistake in one step, then set LOCALIZATION to that step number.",
  gap: "TASK: GAP — edit the working copy to remove a justification/reasoning step (leaving a gap), then set LOCALIZATION to that step number.",
  truncation: "TASK: TRUNCATION — cut the working copy short partway through, then set LOCALIZATION to the step number where it was cut off.",
};

function defaultsForTaskType(taskType, original) {
  if (taskType === "positive") {
    return { edited_proof: original, sufficient: "1", error_type: "none", localization: "N/A" };
  }
  if (taskType === "mistake" || taskType === "gap" || taskType === "truncation") {
    return { edited_proof: original, sufficient: "0", error_type: taskType, localization: "" };
  }
  return { edited_proof: original, sufficient: "", error_type: "", localization: "" };
}

async function showDetail(idx) {
  activeIdx = idx;
  renderList();
  const res = await fetch("/api/row/" + idx);
  const r = await res.json();
  activeRow = r;
  const hasSaved = r.annotation !== null && r.annotation !== undefined;
  const defaults = defaultsForTaskType(r.task_type, r.proof);
  const a = hasSaved ? r.annotation : defaults;
  const isExcluded = !!(hasSaved && a.excluded);
  const detail = document.getElementById("detail");
  const banner = isExcluded
    ? `<div id="task_banner" class="excluded">NOT ELIGIBLE — flagged as a regex false positive (not actually a numbered-step proof).${a.exclusion_reason ? " Reason: " + escapeHtml(a.exclusion_reason) : ""}</div>`
    : (r.task_type ? `<div id="task_banner" class="${r.task_type}">${TASK_INSTRUCTIONS[r.task_type]}</div>` : "");
  detail.innerHTML = `
    <h2>id ${r.id} <span style="opacity:.5;font-weight:normal">(#${r.idx})</span></h2>
    <div class="meta">domain: ${(r.domain || []).join(", ")} · difficulty: ${r.difficulty} · truth_value: ${r.truth_value} · source: ${r.source} · pattern: ${r.pattern}</div>
    ${banner}
    <div class="field-label">Question</div>
    <div class="question-text">${r.informal_theorem_qa}</div>
    <div class="cols">
      <div class="col">
        <div class="field-label">Original proof (rendered)</div>
        <div class="proof-text" id="orig_proof">${r.proof}</div>
      </div>
      <div class="col">
        <div class="field-label">Working copy (edit here — delete a step / inject an error)</div>
        <textarea id="edited_proof">${escapeTextarea(a.edited_proof !== undefined && a.edited_proof !== null ? a.edited_proof : r.proof)}</textarea>
        <button onclick="previewEdited()">Render preview</button>
        <div class="proof-text" id="edited_preview"></div>
      </div>
    </div>
    <div id="save_bar">
      <div class="annot-row">
        <div>
          <label>SUFFICIENT</label>
          <select id="ann_sufficient">
            <option value="">(unset)</option>
            <option value="1" ${a.sufficient === "1" ? "selected" : ""}>1 (yes, sufficient)</option>
            <option value="0" ${a.sufficient === "0" ? "selected" : ""}>0 (no, not sufficient)</option>
          </select>
        </div>
        <div>
          <label>ERROR_TYPE</label>
          <select id="ann_error_type">
            <option value="">(unset)</option>
            <option value="none" ${a.error_type === "none" ? "selected" : ""}>none</option>
            <option value="mistake" ${a.error_type === "mistake" ? "selected" : ""}>mistake</option>
            <option value="gap" ${a.error_type === "gap" ? "selected" : ""}>gap</option>
            <option value="truncation" ${a.error_type === "truncation" ? "selected" : ""}>truncation</option>
          </select>
        </div>
        <div>
          <label>LOCALIZATION (step number or N/A)</label>
          <div style="display:flex; gap:4px;">
            <input id="ann_localization" style="flex:1" value="${a.localization !== undefined && a.localization !== null ? a.localization : ''}">
            <button type="button" style="width:auto; flex-shrink:0;" onclick="document.getElementById('ann_localization').value='N/A'">N/A</button>
          </div>
        </div>
      </div>
      <button onclick="saveAnnotation()">Save annotation</button>
      <span id="save_msg"></span>
    </div>
    <div id="eligibility_bar">
      <label>Eligibility (regex false positives — this proof doesn't actually have numbered steps)</label>
      <div style="display:flex; gap:6px;">
        <input id="ann_exclude_reason" placeholder="reason (optional), e.g. citation number, decimal, not real steps" style="flex:1" value="${a.exclusion_reason || ''}">
        <button type="button" style="width:auto; flex-shrink:0;" onclick="toggleExclude()">${isExcluded ? "Un-exclude" : "Not eligible — exclude"}</button>
      </div>
      <span id="exclude_msg"></span>
    </div>
  `;
  if (window.MathJax && window.MathJax.typesetPromise) {
    MathJax.typesetPromise([document.getElementById("detail")]);
  }
}

function escapeTextarea(s) {
  return s.replace(/</g, "&lt;");
}

function previewEdited() {
  const text = document.getElementById("edited_proof").value;
  const preview = document.getElementById("edited_preview");
  preview.textContent = text;
  if (window.MathJax && window.MathJax.typesetPromise) {
    MathJax.typesetPromise([preview]);
  }
}

function buildPayload(overrides) {
  const base = {
    idx: activeRow.idx,
    id: activeRow.id,
    task_type: activeRow.task_type,
    edited_proof: document.getElementById("edited_proof").value,
    sufficient: document.getElementById("ann_sufficient").value,
    error_type: document.getElementById("ann_error_type").value,
    localization: document.getElementById("ann_localization").value,
    excluded: !!(activeRow.annotation && activeRow.annotation.excluded),
    exclusion_reason: document.getElementById("ann_exclude_reason").value,
  };
  return Object.assign(base, overrides || {});
}

async function postAnnotation(payload) {
  const res = await fetch("/api/annotate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (data.ok) {
    activeRow.annotation = payload;
    currentRows.forEach(r => {
      if (r.idx === activeRow.idx) { r.has_annotation = true; r.excluded = payload.excluded; }
    });
  }
  return data.ok;
}

async function saveAnnotation() {
  const ok = await postAnnotation(buildPayload());
  const msg = document.getElementById("save_msg");
  msg.textContent = ok ? "saved." : "error saving.";
  renderList();
  setTimeout(() => { msg.textContent = ""; }, 2000);
}

async function toggleExclude() {
  const newExcluded = !(activeRow.annotation && activeRow.annotation.excluded);
  const ok = await postAnnotation(buildPayload({ excluded: newExcluded }));
  const msg = document.getElementById("exclude_msg");
  msg.textContent = ok ? "saved." : "error saving.";
  renderList();
  if (ok) {
    showDetail(activeRow.idx);
  }
  setTimeout(() => { msg.textContent = ""; }, 2000);
}

function prevPage() { offset = Math.max(0, offset - limit); loadRows(); }
function nextPage() { if (offset + limit < total) { offset += limit; loadRows(); } }

async function jumpToId() {
  const id = document.getElementById("jump_id").value;
  if (!id) return;
  const res = await fetch("/api/find_id?id=" + encodeURIComponent(id));
  const data = await res.json();
  if (data.idx === null) { alert("id not found in current dataset"); return; }
  offset = Math.max(0, data.page_offset);
  await loadRows();
  showDetail(data.idx);
}

["q","domain","pattern","truth","task_type","annotated","eligibility","min_diff","max_diff"].forEach(id => {
  document.getElementById(id).addEventListener("change", () => { offset = 0; loadRows(); });
});
document.getElementById("q").addEventListener("keyup", (e) => { if (e.key === "Enter") { offset = 0; loadRows(); } });

loadFacets();
loadRows();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    rows = []
    domains = []
    annotations = {}
    annotations_path = "./Data/proof_annotations.json"

    def log_message(self, fmt, *args):
        pass

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html, status=200):
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _filtered(self, params):
        q = (params.get("q", [""])[0] or "").lower()
        domain = params.get("domain", [""])[0]
        pattern = params.get("pattern", [""])[0]
        truth = params.get("truth", [""])[0]
        task_type = params.get("task_type", [""])[0]
        annotated = params.get("annotated", [""])[0]
        eligibility = params.get("eligibility", [""])[0]
        min_diff = params.get("min_diff", [""])[0]
        max_diff = params.get("max_diff", [""])[0]

        result = self.rows
        if domain:
            result = [r for r in result if domain in (r["domain"] or [])]
        if pattern:
            result = [r for r in result if r["pattern"] == pattern]
        if truth in ("true", "false"):
            want = truth == "true"
            result = [r for r in result if r["truth_value"] == want]
        if task_type == "sampled":
            result = [r for r in result if r["task_type"]]
        elif task_type == "unsampled":
            result = [r for r in result if not r["task_type"]]
        elif task_type in ("positive", "mistake", "gap", "truncation"):
            result = [r for r in result if r["task_type"] == task_type]
        if annotated == "yes":
            result = [r for r in result if str(r["id"]) in self.annotations]
        elif annotated == "no":
            result = [r for r in result if str(r["id"]) not in self.annotations]
        if eligibility == "excluded":
            result = [r for r in result if self.annotations.get(str(r["id"]), {}).get("excluded")]
        elif eligibility == "not_excluded":
            result = [r for r in result if not self.annotations.get(str(r["id"]), {}).get("excluded")]
        if min_diff:
            result = [r for r in result if r["difficulty"] is not None and r["difficulty"] >= float(min_diff)]
        if max_diff:
            result = [r for r in result if r["difficulty"] is not None and r["difficulty"] <= float(max_diff)]
        if q:
            result = [r for r in result if q in r["proof"].lower() or q in (r["informal_theorem_qa"] or "").lower()]
        return result

    def _row_with_annotation(self, r):
        out = dict(r)
        out["annotation"] = self.annotations.get(str(r["id"]))
        out["has_annotation"] = str(r["id"]) in self.annotations
        return out

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if parsed.path == "/":
            self._send_html(INDEX_HTML)
        elif parsed.path == "/api/facets":
            self._send_json({"domains": self.domains})
        elif parsed.path == "/api/rows":
            filtered = self._filtered(params)
            offset = int(params.get("offset", ["0"])[0])
            limit = int(params.get("limit", ["25"])[0])
            page = filtered[offset:offset + limit]
            sample_ids = {r["id"] for r in self.rows if r["task_type"]}
            sample_annotated = sum(1 for rid in sample_ids if str(rid) in self.annotations)
            self._send_json({
                "total": len(filtered),
                "annotated_total": len(self.annotations),
                "sample_total": len(sample_ids),
                "sample_annotated": sample_annotated,
                "rows": [
                    {
                        **r,
                        "has_annotation": str(r["id"]) in self.annotations,
                        "excluded": bool(self.annotations.get(str(r["id"]), {}).get("excluded")),
                    }
                    for r in page
                ],
            })
        elif parsed.path.startswith("/api/row/"):
            idx = int(parsed.path.rsplit("/", 1)[-1])
            if 0 <= idx < len(self.rows):
                self._send_json(self._row_with_annotation(self.rows[idx]))
            else:
                self._send_json({"error": "not found"}, status=404)
        elif parsed.path == "/api/find_id":
            target = params.get("id", [""])[0]
            found = next((r for r in self.rows if str(r["id"]) == target), None)
            if found is None:
                self._send_json({"idx": None})
            else:
                filtered = self._filtered({})
                pos_in_filtered = next((i for i, r in enumerate(filtered) if r["idx"] == found["idx"]), 0)
                page_offset = (pos_in_filtered // 25) * 25
                self._send_json({"idx": found["idx"], "page_offset": page_offset})
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/annotate":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            record_id = str(body.get("id"))
            self.annotations[record_id] = {
                "idx": body.get("idx"),
                "task_type": body.get("task_type"),
                "edited_proof": body.get("edited_proof"),
                "sufficient": body.get("sufficient"),
                "error_type": body.get("error_type"),
                "localization": body.get("localization"),
                "excluded": bool(body.get("excluded")),
                "exclusion_reason": body.get("exclusion_reason"),
            }
            save_annotations(self.annotations_path, self.annotations)
            self._send_json({"ok": True})
        else:
            self._send_json({"error": "not found"}, status=404)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./Data/Processed_DeepTheorem_StepNumbered")
    parser.add_argument("--annotations", default="./Data/proof_annotations.json")
    parser.add_argument("--sample", default="./Data/annotation_sample.json")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset} ...")
    sample = load_json(args.sample)
    print(f"Loaded {len(sample)} task-type assignments from {args.sample}")
    rows = load_rows(args.dataset, sample)
    Handler.rows = rows
    Handler.domains = sorted({d for r in rows for d in (r["domain"] or [])})
    Handler.annotations_path = args.annotations
    Handler.annotations = load_annotations(args.annotations)
    print(f"Loaded {len(rows)} rows, {len(Handler.annotations)} existing annotations.")
    print(f"Serving on http://localhost:{args.port}")

    server = ThreadingHTTPServer(("localhost", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
