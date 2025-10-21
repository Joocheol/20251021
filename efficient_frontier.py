"""Compute efficient frontier for assets in temp.csv without external dependencies."""
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

DATA_FILE = Path("temp.csv")
OUTPUT_SVG = Path("efficient_frontier.svg")
OUTPUT_JSON = Path("efficient_frontier_data.json")


def load_close_prices(path: Path) -> Tuple[List[str], List[Dict[str, float]], List[datetime]]:
    """Return tickers, list of price rows, and matching dates (sorted)."""
    with path.open(newline="") as f:
        reader = csv.reader(f)
        try:
            categories = next(reader)
            tickers = next(reader)
            _ = next(reader)  # the "Date" row
        except StopIteration as exc:  # pragma: no cover - defensive
            raise ValueError("CSV file ended before data rows were found") from exc

        close_indices: List[int] = []
        close_tickers: List[str] = []
        for idx, (category, ticker) in enumerate(zip(categories, tickers)):
            if category.strip().lower() == "close" and ticker.strip():
                close_indices.append(idx)
                close_tickers.append(ticker.strip())

        if not close_indices:
            raise ValueError("No closing price columns detected in CSV header")

        rows: List[Dict[str, float]] = []
        dates: List[datetime] = []
        for row in reader:
            if not row or not row[0].strip():
                continue
            date = datetime.strptime(row[0].strip(), "%Y-%m-%d")
            record: Dict[str, float] = {}
            missing = False
            for idx, ticker in zip(close_indices, close_tickers):
                value = row[idx].strip()
                if not value:
                    missing = True
                    break
                record[ticker] = float(value)
            if not missing:
                rows.append(record)
                dates.append(date)

    paired = sorted(zip(dates, rows), key=lambda item: item[0])
    sorted_dates = [item[0] for item in paired]
    sorted_rows = [item[1] for item in paired]
    return close_tickers, sorted_rows, sorted_dates


def compute_returns(tickers: Sequence[str], rows: Sequence[Dict[str, float]]) -> List[List[float]]:
    """Compute daily simple returns for each ticker."""
    if len(rows) < 2:
        raise ValueError("Need at least two price observations to compute returns")
    returns: List[List[float]] = []
    for i in range(1, len(rows)):
        prev_row = rows[i - 1]
        curr_row = rows[i]
        vector: List[float] = []
        for ticker in tickers:
            prev_price = prev_row[ticker]
            curr_price = curr_row[ticker]
            if prev_price <= 0:
                raise ValueError(f"Encountered non-positive price for {ticker}")
            vector.append(curr_price / prev_price - 1.0)
        returns.append(vector)
    return returns


def mean_vector(returns: Sequence[Sequence[float]]) -> List[float]:
    n_assets = len(returns[0])
    means = [0.0] * n_assets
    count = len(returns)
    for vector in returns:
        for idx, value in enumerate(vector):
            means[idx] += value
    return [value / count for value in means]


def covariance_matrix(returns: Sequence[Sequence[float]], means: Sequence[float]) -> List[List[float]]:
    n_assets = len(means)
    count = len(returns)
    if count < 2:
        return [[0.0 for _ in range(n_assets)] for _ in range(n_assets)]
    matrix = [[0.0 for _ in range(n_assets)] for _ in range(n_assets)]
    for vector in returns:
        for i in range(n_assets):
            diff_i = vector[i] - means[i]
            for j in range(n_assets):
                matrix[i][j] += diff_i * (vector[j] - means[j])
    scale = 1.0 / (count - 1)
    for i in range(n_assets):
        for j in range(n_assets):
            matrix[i][j] *= scale
    return matrix


@dataclass
class PortfolioPoint:
    weights: Tuple[float, ...]
    expected_return: float
    risk: float


def portfolio_metrics(weights: Sequence[float], means: Sequence[float], cov: Sequence[Sequence[float]]) -> PortfolioPoint:
    expected = sum(w * mu for w, mu in zip(weights, means))
    variance = 0.0
    for i in range(len(weights)):
        for j in range(len(weights)):
            variance += weights[i] * weights[j] * cov[i][j]
    variance = max(variance, 0.0)
    return PortfolioPoint(tuple(weights), expected, math.sqrt(variance))


def generate_weight_grid(n_assets: int, step: float = 0.02) -> List[Tuple[float, ...]]:
    if n_assets < 2:
        raise ValueError("Efficient frontier requires at least two assets")
    weights: List[Tuple[float, ...]] = []
    steps = int(round(1.0 / step))
    if n_assets == 2:
        for i in range(steps + 1):
            w1 = i * step
            w2 = max(0.0, 1.0 - w1)
            weights.append((w1, w2))
    elif n_assets == 3:
        for i in range(steps + 1):
            w1 = i * step
            for j in range(steps + 1 - i):
                w2 = j * step
                w3 = 1.0 - w1 - w2
                if w3 < -1e-9:
                    continue
                w3 = max(0.0, w3)
                total = w1 + w2 + w3
                weights.append((w1 / total, w2 / total, w3 / total))
    else:
        # Simple recursive generator for higher dimensions
        def rec(prefix: List[float], remaining: int, residual: float) -> None:
            if remaining == 1:
                prefix.append(residual)
                weights.append(tuple(prefix))
                prefix.pop()
                return
            for k in range(steps + 1):
                weight = min(k * step, residual)
                if weight > residual + 1e-9:
                    break
                prefix.append(weight)
                rec(prefix, remaining - 1, residual - weight)
                prefix.pop()
        rec([], n_assets, 1.0)
    return weights


def extract_frontier(points: Sequence[PortfolioPoint]) -> List[PortfolioPoint]:
    ordered = sorted(points, key=lambda p: (p.risk, -p.expected_return))
    frontier: List[PortfolioPoint] = []
    best_return = float("-inf")
    for point in ordered:
        if point.expected_return > best_return + 1e-9:
            frontier.append(point)
            best_return = point.expected_return
    return frontier


def format_percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def create_svg(points: Sequence[PortfolioPoint], frontier: Sequence[PortfolioPoint], path: Path) -> None:
    width, height = 800, 600
    margin = 60
    risks = [p.risk for p in points]
    returns = [p.expected_return for p in points]
    min_risk, max_risk = min(risks), max(risks)
    min_return, max_return = min(returns), max(returns)

    def scale_x(risk: float) -> float:
        if math.isclose(max_risk, min_risk):
            return width / 2
        return margin + (risk - min_risk) / (max_risk - min_risk) * (width - 2 * margin)

    def scale_y(ret: float) -> float:
        if math.isclose(max_return, min_return):
            return height / 2
        return height - margin - (ret - min_return) / (max_return - min_return) * (height - 2 * margin)

    lines: List[str] = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "  <style>text { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 14px; }</style>",
        "  <rect x='0' y='0' width='{width}' height='{height}' fill='white' stroke='none'/>".format(width=width, height=height),
        "  <g stroke='black' stroke-width='1'>",
        f"    <line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}'/>",
        f"    <line x1='{margin}' y1='{height - margin}' x2='{margin}' y2='{margin}'/>",
        "  </g>",
        f"  <text x='{width / 2}' y='{height - margin / 2}' text-anchor='middle'>Risk (σ)</text>",
        f"  <text x='{margin / 2}' y='{height / 2}' text-anchor='middle' transform='rotate(-90 {margin / 2} {height / 2})'>Expected Return (μ)</text>",
    ]

    # Tick marks
    for i in range(6):
        frac = i / 5
        risk_value = min_risk + frac * (max_risk - min_risk)
        x = scale_x(risk_value)
        lines.append(f"  <line x1='{x:.2f}' y1='{height - margin}' x2='{x:.2f}' y2='{height - margin + 6}' stroke='black'/>")
        lines.append(
            f"  <text x='{x:.2f}' y='{height - margin + 24}' text-anchor='middle'>{risk_value:.3f}</text>"
        )

        return_value = min_return + frac * (max_return - min_return)
        y = scale_y(return_value)
        lines.append(f"  <line x1='{margin - 6}' y1='{y:.2f}' x2='{margin}' y2='{y:.2f}' stroke='black'/>")
        lines.append(
            f"  <text x='{margin - 12}' y='{y + 5:.2f}' text-anchor='end'>{return_value:.3f}</text>"
        )

    # All portfolio points
    for point in points:
        x = scale_x(point.risk)
        y = scale_y(point.expected_return)
        lines.append(
            f"  <circle cx='{x:.2f}' cy='{y:.2f}' r='3' fill='#B0BEC5' stroke='none' fill-opacity='0.7'/>"
        )

    # Efficient frontier
    if frontier:
        path_data = " ".join(
            f"L {scale_x(p.risk):.2f} {scale_y(p.expected_return):.2f}" for p in frontier[1:]
        )
        start = frontier[0]
        lines.append(
            f"  <path d='M {scale_x(start.risk):.2f} {scale_y(start.expected_return):.2f} {path_data}' "
            "fill='none' stroke='#D32F2F' stroke-width='3' stroke-linecap='round'/>"
        )
        for point in frontier:
            x = scale_x(point.risk)
            y = scale_y(point.expected_return)
            lines.append(
                f"  <circle cx='{x:.2f}' cy='{y:.2f}' r='4' fill='#D32F2F' stroke='white' stroke-width='1.5'/>"
            )

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def annualize_return(daily_return: float, periods: int = 252) -> float:
    return (1.0 + daily_return) ** periods - 1.0


def annualize_volatility(daily_vol: float, periods: int = 252) -> float:
    return daily_vol * math.sqrt(periods)


def main() -> None:
    tickers, rows, dates = load_close_prices(DATA_FILE)
    returns = compute_returns(tickers, rows)
    means = mean_vector(returns)
    cov = covariance_matrix(returns, means)
    points = [
        portfolio_metrics(weights, means, cov)
        for weights in generate_weight_grid(len(tickers), step=0.02)
    ]
    frontier = extract_frontier(points)
    create_svg(points, frontier, OUTPUT_SVG)

    first_date = dates[0].strftime("%Y-%m-%d")
    last_date = dates[-1].strftime("%Y-%m-%d")
    print(f"Price observations used: {len(rows)} sessions ({first_date} to {last_date})")
    print(f"Daily return vectors: {len(returns)}")

    asset_stats: List[Dict[str, float]] = []
    print("\nAsset statistics (daily and annualised):")
    for idx, (ticker, mu) in enumerate(zip(tickers, means)):
        daily_vol = math.sqrt(max(cov[idx][idx], 0.0))
        ann_return = annualize_return(mu)
        ann_vol = annualize_volatility(daily_vol)
        print(
            f"  - {ticker}: μ={format_percentage(mu)} per day ({format_percentage(ann_return)} annualised), "
            f"σ={format_percentage(daily_vol)} per day ({format_percentage(ann_vol)} annualised)"
        )
        asset_stats.append(
            {
                "ticker": ticker,
                "expected_return_daily": mu,
                "expected_return_annual": ann_return,
                "volatility_daily": daily_vol,
                "volatility_annual": ann_vol,
            }
        )

    if frontier:
        print("\nSample efficient frontier points (annualised):")
        sample_points = [frontier[0], frontier[len(frontier) // 2], frontier[-1]]
        seen: set[Tuple[float, float]] = set()
        unique_points: List[PortfolioPoint] = []
        for point in sample_points:
            key = (round(point.risk, 6), round(point.expected_return, 6))
            if key not in seen:
                unique_points.append(point)
                seen.add(key)
        for point in unique_points:
            weights_info = ", ".join(
                f"{ticker}={weight * 100:.1f}%" for ticker, weight in zip(tickers, point.weights)
            )
            ann_return = annualize_return(point.expected_return)
            ann_vol = annualize_volatility(point.risk)
            print(
                f"  - σ={format_percentage(point.risk)} ({format_percentage(ann_vol)} annualised), "
                f"μ={format_percentage(point.expected_return)} ({format_percentage(ann_return)} annualised); {weights_info}"
            )

    payload = {
        "meta": {
            "date_start": first_date,
            "date_end": last_date,
            "sessions": len(rows),
            "return_vectors": len(returns),
            "svg_path": str(OUTPUT_SVG),
        },
        "tickers": tickers,
        "asset_statistics": asset_stats,
        "points": [
            {"risk": point.risk, "expected_return": point.expected_return}
            for point in points
        ],
        "frontier": [
            {
                "weights": {ticker: weight for ticker, weight in zip(tickers, point.weights)},
                "expected_return": point.expected_return,
                "expected_return_annual": annualize_return(point.expected_return),
                "risk": point.risk,
                "risk_annual": annualize_volatility(point.risk),
            }
            for point in frontier
        ],
    }

    if frontier:
        payload["frontier_samples"] = [
            {
                "weights": {ticker: weight for ticker, weight in zip(tickers, point.weights)},
                "expected_return": point.expected_return,
                "expected_return_annual": annualize_return(point.expected_return),
                "risk": point.risk,
                "risk_annual": annualize_volatility(point.risk),
            }
            for point in unique_points
        ]

    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
