const assetTableBody = document.querySelector("#asset-table tbody");
const frontierTableBody = document.querySelector("#frontier-table tbody");
const metaDates = document.getElementById("meta-dates");
const metaSessions = document.getElementById("meta-sessions");
const metaReturns = document.getElementById("meta-returns");
const metaUpdated = document.getElementById("meta-updated");
const canvas = document.getElementById("frontier-canvas");

function formatPercent(value, digits = 2) {
  return `${(value * 100).toFixed(digits)}%`;
}

function formatNumber(value) {
  return value.toLocaleString("ko-KR");
}

function populateAssetTable(stats) {
  if (!Array.isArray(stats) || stats.length === 0) {
    assetTableBody.innerHTML =
      '<tr><td colspan="5">표시할 자산 통계가 없습니다.</td></tr>';
    return;
  }

  const rows = stats
    .map((asset) => {
      return `
        <tr>
          <td>${asset.ticker}</td>
          <td>${formatPercent(asset.expected_return_daily)}</td>
          <td>${formatPercent(asset.volatility_daily)}</td>
          <td>${formatPercent(asset.expected_return_annual)}</td>
          <td>${formatPercent(asset.volatility_annual)}</td>
        </tr>
      `;
    })
    .join("");

  assetTableBody.innerHTML = rows;
}

function populateMeta(meta) {
  if (!meta) {
    return;
  }
  if (meta.date_start && meta.date_end) {
    metaDates.textContent = `${meta.date_start} ~ ${meta.date_end}`;
    metaUpdated.textContent = meta.date_end;
  }
  if (typeof meta.sessions === "number") {
    metaSessions.textContent = `${formatNumber(meta.sessions)}일`;
  }
  if (typeof meta.return_vectors === "number") {
    metaReturns.textContent = `${formatNumber(meta.return_vectors)}개`;
  }
}

function formatWeights(weights, tickers) {
  if (!weights) return "-";
  const order = Array.isArray(tickers) ? tickers : Object.keys(weights);
  return order
    .map((ticker) => `${ticker} ${(weights[ticker] * 100).toFixed(1)}%`)
    .join(" · ");
}

function populateFrontierTable(samples, tickers) {
  if (!Array.isArray(samples) || samples.length === 0) {
    frontierTableBody.innerHTML =
      '<tr><td colspan="4">효율적 프런티어 샘플을 찾을 수 없습니다.</td></tr>';
    return;
  }

  const labels = ["최소 변동성", "중간 지점", "최대 기대수익률"];
  const rows = samples
    .map((point, index) => {
      const label = labels[index] || `지점 ${index + 1}`;
      return `
        <tr>
          <td>${label}</td>
          <td>${formatPercent(point.expected_return_annual)}</td>
          <td>${formatPercent(point.risk_annual)}</td>
          <td>${formatWeights(point.weights, tickers)}</td>
        </tr>
      `;
    })
    .join("");

  frontierTableBody.innerHTML = rows;
}

function drawFrontier(points, frontier) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const width = canvas.width;
  const height = canvas.height;
  const margin = 60;

  const allPoints = (points || []).concat(frontier || []);
  if (allPoints.length === 0) {
    ctx.fillStyle = "#666";
    ctx.font = "16px sans-serif";
    ctx.fillText("표시할 데이터가 없습니다.", margin, height / 2);
    return;
  }

  const risks = allPoints.map((p) => p.risk);
  const returns = allPoints.map((p) => p.expected_return);
  const minRisk = Math.min(...risks);
  const maxRisk = Math.max(...risks);
  const minReturn = Math.min(...returns);
  const maxReturn = Math.max(...returns);

  const scaleX = (risk) => {
    if (Math.abs(maxRisk - minRisk) < 1e-9) return width / 2;
    return (
      margin + ((risk - minRisk) / (maxRisk - minRisk)) * (width - margin * 2)
    );
  };

  const scaleY = (ret) => {
    if (Math.abs(maxReturn - minReturn) < 1e-9) return height / 2;
    return (
      height -
      margin -
      ((ret - minReturn) / (maxReturn - minReturn)) * (height - margin * 2)
    );
  };

  ctx.strokeStyle = "#0f172a";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(margin, height - margin);
  ctx.lineTo(width - margin, height - margin);
  ctx.moveTo(margin, height - margin);
  ctx.lineTo(margin, margin);
  ctx.stroke();

  ctx.fillStyle = "#0f172a";
  ctx.font = "16px 'Noto Sans KR', sans-serif";
  ctx.fillText("Risk (σ)", width / 2 - 40, height - margin / 2);
  ctx.save();
  ctx.translate(margin / 2, height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Expected Return (μ)", -80, 0);
  ctx.restore();

  ctx.strokeStyle = "rgba(15, 23, 42, 0.15)";
  ctx.fillStyle = "rgba(15, 23, 42, 0.65)";
  ctx.lineWidth = 1;
  const ticks = 5;
  for (let i = 0; i <= ticks; i++) {
    const frac = i / ticks;
    const riskValue = minRisk + frac * (maxRisk - minRisk);
    const returnValue = minReturn + frac * (maxReturn - minReturn);
    const x = scaleX(riskValue);
    const y = scaleY(returnValue);

    ctx.beginPath();
    ctx.moveTo(x, height - margin);
    ctx.lineTo(x, height - margin + 8);
    ctx.stroke();

    ctx.fillText(riskValue.toFixed(3), x - 18, height - margin + 24);

    ctx.beginPath();
    ctx.moveTo(margin - 8, y);
    ctx.lineTo(margin, y);
    ctx.stroke();

    ctx.fillText(returnValue.toFixed(3), margin - 58, y + 4);
  }

  ctx.fillStyle = "rgba(176, 190, 197, 0.75)";
  points.forEach((point) => {
    const x = scaleX(point.risk);
    const y = scaleY(point.expected_return);
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
  });

  if (frontier && frontier.length > 0) {
    ctx.strokeStyle = "#d32f2f";
    ctx.lineWidth = 3;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();
    frontier.forEach((point, index) => {
      const x = scaleX(point.risk);
      const y = scaleY(point.expected_return);
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    ctx.fillStyle = "#d32f2f";
    frontier.forEach((point) => {
      const x = scaleX(point.risk);
      const y = scaleY(point.expected_return);
      ctx.beginPath();
      ctx.arc(x, y, 4.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "white";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(x, y, 4.5, 0, Math.PI * 2);
      ctx.stroke();
    });
  }
}

async function init() {
  try {
    const response = await fetch("efficient_frontier_data.json");
    if (!response.ok) {
      throw new Error(`데이터를 불러오지 못했습니다: ${response.status}`);
    }
    const data = await response.json();
    populateMeta(data.meta);
    populateAssetTable(data.asset_statistics);
    populateFrontierTable(data.frontier_samples || [], data.tickers);
    drawFrontier(data.points || [], data.frontier || []);
  } catch (error) {
    console.error(error);
    assetTableBody.innerHTML =
      '<tr><td colspan="5">데이터를 불러오는 중 문제가 발생했습니다.</td></tr>';
    frontierTableBody.innerHTML =
      '<tr><td colspan="4">데이터를 불러오는 중 문제가 발생했습니다.</td></tr>';
    if (canvas) {
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#d32f2f";
        ctx.font = "18px sans-serif";
        ctx.fillText("데이터 로드 실패", 80, canvas.height / 2);
      }
    }
  }
}

window.addEventListener("DOMContentLoaded", init);
