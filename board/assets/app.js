// AirFour finger lock web page: status from the board (socket.io "status" messages,
// with /api/status polling as a fallback), and the backup button, ROI calibration
// and live accuracy test.
const $ = (id) => document.getElementById(id);
const base = `http://${window.location.host}`;
const socket = io(base);
let lastMessage = 0;

const fmt = (v, unit = "", digits = 1) => (v === null || v === undefined ? "-" : `${Number(v).toFixed(digits)}${unit}`);

function render(s) {
  lastMessage = Date.now();
  $("state").textContent = s.state.toUpperCase() + (s.seconds_left !== null && s.state !== "locked" ? `  ${Math.ceil(s.seconds_left)} s` : "");
  $("state").style.color = s.state_color;
  $("slots").textContent = s.slots.join(" ");
  $("hint").textContent = s.hint;
  $("reading").textContent = s.reading;
  $("reading").style.color = s.reading_color;
  const st = s.stability;
  $("bar-fill").style.width = `${(100 * st.count) / st.needed}%`;
  $("bar-fill").style.background = st.count >= st.needed ? "#1baf7a" : "#aaa";
  $("stability").textContent = `stable: ${st.value ?? "-"} ${st.count}/${st.needed} frames`;
  $("diff").textContent = s.no_hand_diff === null ? "no snapshot yet" : `${s.no_hand_diff} (threshold ${s.threshold})`;
  $("wrong").textContent = `${s.wrong_in_a_row} / ${s.max_wrong}`;
  $("fps").textContent = fmt(s.fps);
  $("infer").textContent = fmt(s.infer_ms, " ms", 2);
  $("process").textContent = fmt(s.process_ms, " ms", 2);
  $("cpu").textContent = `${fmt(s.cpu_process, "%")} / ${fmt(s.cpu_board, "%")}`;
  $("mem").textContent = fmt(s.mem_mb, " MB");
  renderEval(s.eval);
}

function renderEval(e) {
  $("eval-expected").textContent = e.expected === null ? "" : `Waiting for a stable ${e.expected === 0 ? "fist (0)" : e.expected}...`;
  document.querySelectorAll("#eval-buttons button").forEach((b) => b.classList.toggle("active", Number(b.dataset.label) === e.expected));
  $("eval-table").innerHTML = Object.entries(e.per_label)
    .map(([k, v]) => `<tr><th>${k === "0" ? "0 (fist)" : k}</th><td>${v.n}</td><td>${v.correct}</td><td>${v.n ? Math.round((100 * v.correct) / v.n) + "%" : "-"}</td></tr>`)
    .join("");
  $("eval-n").textContent = e.n;
  $("eval-c").textContent = e.correct;
  $("eval-acc").textContent = e.n ? `${Math.round((100 * e.correct) / e.n)}%` : "-";
  $("eval-last").textContent = e.last.length
    ? "Last: " + e.last.map((t) => `${t.expected}→${t.predicted ?? "none"}`).join(", ")
    : "";
}

async function post(path) {
  const r = await fetch(base + path, { method: "POST" });
  return r.json();
}

for (let k = 0; k < 6; k++) {
  const b = document.createElement("button");
  b.textContent = k === 0 ? "0 (fist)" : String(k);
  b.dataset.label = k;
  b.onclick = () => post(`/api/eval/${k}`);
  $("eval-buttons").appendChild(b);
}
$("press").onclick = () => post("/api/press");
$("calibrate").onclick = async () => ($("message").textContent = (await post("/api/calibrate")).message);
$("eval-reset").onclick = () => confirm("Clear all accuracy-test results?") && post("/api/eval_reset");

socket.on("connect", () => { $("link").textContent = "live"; $("link").classList.add("ok"); });
socket.on("disconnect", () => { $("link").textContent = "disconnected"; $("link").classList.remove("ok"); });
socket.on("status", render);
setInterval(async () => {
  if (Date.now() - lastMessage < 1500) return;
  try { render(await (await fetch(base + "/api/status")).json()); } catch (e) { /* board restarting */ }
}, 1000);
