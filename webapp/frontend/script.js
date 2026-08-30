const drawBtn = document.getElementById("draw-btn");
const statusEl = document.getElementById("status");
const cardsEl = document.getElementById("cards");
const resultEl = document.getElementById("result");
const engineEl = document.getElementById("engine");
const countEl = document.getElementById("count");
const targetEl = document.getElementById("target");
const conditionEl = document.getElementById("condition");

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.hidden = !text;
  statusEl.classList.toggle("error", isError);
}

function renderCards(cards, condition) {
  cardsEl.innerHTML = "";
  for (const card of cards) {
    const div = document.createElement("div");
    div.className = "card";

    const img = document.createElement("img");
    // 条件(tilt/light)を実写データに差し替えても画像パス自体は変わらないため、
    // ブラウザキャッシュが古い画像を返し続けないようクエリでキャッシュを回避する。
    img.src = `/images/${condition}/${encodeURIComponent(card.filename)}?v=${Date.now()}`;
    img.alt = card.filename;
    div.appendChild(img);

    const pred = document.createElement("div");
    if (card.pred_rank) {
      pred.className = "pred";
      pred.textContent = `認識: ${card.pred_rank}`;
    } else {
      pred.className = "pred unknown";
      pred.textContent = "認識: UNKNOWN";
    }
    div.appendChild(pred);

    const filename = document.createElement("div");
    filename.className = "filename";
    filename.textContent = card.filename;
    div.appendChild(filename);

    cardsEl.appendChild(div);
  }
  cardsEl.hidden = cards.length === 0;
}

function renderResult(data) {
  resultEl.innerHTML = "";

  if (data.expression) {
    const expr = document.createElement("div");
    expr.className = "expression";
    expr.textContent = `${data.expression} = ${data.target}`;
    resultEl.appendChild(expr);
  } else {
    const none = document.createElement("div");
    none.className = "no-solution";
    none.textContent = `${data.target} を作る式は見つかりませんでした。`;
    resultEl.appendChild(none);
  }

  if (data.unknown_count > 0) {
    const note = document.createElement("div");
    note.className = "note";
    note.textContent = `判別できなかったカードが ${data.unknown_count} 枚あったため、計算から除外しています。`;
    resultEl.appendChild(note);
  }

  const conditionLabel = { normal: "通常", tilt: "傾き", light: "照明変化" }[data.condition] || data.condition;
  const engineNote = document.createElement("div");
  engineNote.className = "note";
  engineNote.textContent = `読み取った数字: [${data.values.join(", ")}] (エンジン: ${data.engine}, 条件: ${conditionLabel})`;
  resultEl.appendChild(engineNote);

  resultEl.hidden = false;
}

async function draw() {
  drawBtn.disabled = true;
  cardsEl.hidden = true;
  resultEl.hidden = true;
  setStatus("カードを引いて判別しています…");

  const params = new URLSearchParams({
    engine: engineEl.value,
    count: countEl.value,
    target: targetEl.value || "10",
    condition: conditionEl.value,
  });

  try {
    const res = await fetch(`/api/draw?${params.toString()}`);
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `サーバーエラー (${res.status})`);
    }
    const data = await res.json();
    setStatus("");
    renderCards(data.cards, data.condition);
    renderResult(data);
  } catch (err) {
    setStatus(err.message || "通信に失敗しました。", true);
  } finally {
    drawBtn.disabled = false;
  }
}

drawBtn.addEventListener("click", draw);
