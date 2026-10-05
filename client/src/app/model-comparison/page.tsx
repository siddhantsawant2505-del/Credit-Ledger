"use client";

import { useEffect, useMemo, useState } from "react";
import {
  ApiUnavailableError,
  MODEL_DISPLAY,
  POPULATION_DEFAULT_RATE,
  TRAIN_ROWS,
  fetchBenchmarks,
  type Benchmarks,
} from "../../lib/api";

interface Row {
  key: string;
  rank: string;
  name: string;
  architecture: string;
  auc: string;
  aucNum: number;
  f1: string;
  f1Num: number;
  ks: string;
  ksNum: number;
  prAuc: string;
  brier: string;
  gini: string;
  cvStd: string;
}

function toRows(data: Benchmarks): Row[] {
  return Object.entries(data)
    .filter(([k, v]) => MODEL_DISPLAY[k] && typeof v?.auc_roc === "number")
    .sort((a, b) => b[1].auc_roc - a[1].auc_roc)
    .map(([key, m], i) => ({
      key,
      rank: String(i + 1).padStart(2, "0"),
      name: MODEL_DISPLAY[key].label,
      architecture: MODEL_DISPLAY[key].architecture,
      auc: m.auc_roc.toFixed(4),
      aucNum: m.auc_roc,
      f1: (m.optimal_threshold_metrics?.f1 ?? 0).toFixed(4),
      f1Num: m.optimal_threshold_metrics?.f1 ?? 0,
      ks: m.ks_stat.toFixed(4),
      ksNum: m.ks_stat,
      prAuc: m.pr_auc.toFixed(4),
      brier: typeof m.brier === "number" ? m.brier.toFixed(4) : "—",
      gini: `${((2 * m.auc_roc - 1) * 100).toFixed(1)}%`,
      cvStd: `± ${m.cv_auc_std.toFixed(4)}`,
    }));
}

// What each metric means and why it was / was not chosen
const METRIC_RATIONALE = [
  {
    name: "AUC-ROC",
    importance: "primary",
    label: "✓ Primary metric",
    color: "border-[#39684a] text-[#39684a] bg-[#bbefc9]/20",
    what: "Measures how well the model separates defaulters from non-defaulters across all possible decision thresholds.",
    why: "The gold standard for credit scoring. It does not depend on any single threshold and works correctly even with heavily imbalanced data (8% default rate). A score of 0.5 is random guessing; 1.0 is perfect. Our champion reaches 0.786 — meaning the model ranks a randomly chosen defaulter above a randomly chosen non-defaulter 78.6% of the time.",
    whyNot: null,
  },
  {
    name: "KS Statistic",
    importance: "primary",
    label: "✓ Primary metric",
    color: "border-[#39684a] text-[#39684a] bg-[#bbefc9]/20",
    what: "Kolmogorov-Smirnov statistic — the maximum gap between the cumulative distribution of defaulters and non-defaulters when sorted by model score.",
    why: "The industry-standard measure for credit scorecards (Basel III compliance). Tells you how sharply the model divides the population. A KS of 43% means that at the optimal cut-off, we capture 43 percentage points more of the bad accounts than we would by random selection. Regulators and risk committees typically look for KS > 30%.",
    whyNot: null,
  },
  {
    name: "PR-AUC",
    importance: "secondary",
    label: "✓ Secondary metric",
    color: "border-[#486366] text-[#486366] bg-[#ecefeb]",
    what: "Area under the Precision-Recall curve — focuses only on the minority (defaulter) class.",
    why: "Because only 8.07% of applicants default, standard accuracy is misleading. PR-AUC directly measures how well the model identifies defaulters without being distorted by the large majority of good applicants. Our models score 0.27–0.28 vs. a random baseline of 0.08 — a 3.4× improvement.",
    whyNot: null,
  },
  {
    name: "F1 Score (at optimal threshold)",
    importance: "secondary",
    label: "✓ Secondary metric",
    color: "border-[#486366] text-[#486366] bg-[#ecefeb]",
    what: "Harmonic mean of precision and recall, measured at the threshold that maximises it.",
    why: "Provides a single number that balances catching defaulters (recall) against false alarms (precision). We use the threshold-optimal F1, not the fixed 0.5 threshold F1, because the optimal cut-off reflects actual deployment behaviour.",
    whyNot: null,
  },
  {
    name: "Gini Coefficient",
    importance: "derived",
    label: "⊕ Derived metric",
    color: "border-[#414849] text-[#414849] bg-[#f1f4f1]",
    what: "Gini = 2 × AUC − 1. Converts AUC into a percentage of 'ranking power above random'. A Gini of 57% means the model is 57% better than chance at ranking applicants.",
    why: "Common in European and banking regulatory contexts as an alternative way to express AUC. Contains no new information beyond AUC but is more intuitive to communicate to non-technical stakeholders.",
    whyNot: null,
  },
  {
    name: "CV Standard Deviation",
    importance: "stability",
    label: "✓ Stability check",
    color: "border-[#B8862E] text-[#B8862E] bg-[#F5F4EF]",
    what: "Standard deviation of AUC across the 5 cross-validation folds.",
    why: "A model that performs consistently across all folds (low std) is more trustworthy than one that spikes on some folds and collapses on others. Our models all achieve ±0.003–0.005, indicating stable generalisation.",
    whyNot: null,
  },
  {
    name: "Accuracy",
    importance: "excluded",
    label: "✗ Not used",
    color: "border-[#ba1a1a] text-[#ba1a1a] bg-[#ffdad6]/20",
    what: "The percentage of all predictions that are correct.",
    why: null,
    whyNot: "Completely misleading with imbalanced data. A model that predicts 'no default' for every applicant achieves 91.9% accuracy without learning anything. Accuracy cannot distinguish between a useful model and a trivial one in this context.",
  },
  {
    name: "F1 at 0.5 threshold",
    importance: "excluded",
    label: "✗ Not used",
    color: "border-[#ba1a1a] text-[#ba1a1a] bg-[#ffdad6]/20",
    what: "F1 score measured at the arbitrary 0.5 probability cut-off.",
    why: null,
    whyNot: "The 0.5 threshold is designed for balanced datasets. With only 8% defaulters, the calibrated decision threshold should be near 0.08, not 0.5. Using 0.5 results in near-zero recall on the minority class and produces misleadingly low F1 scores (0.03–0.31) that do not reflect real model quality.",
  },
  {
    name: "Brier Score",
    importance: "excluded",
    label: "⊖ Available but deprioritised",
    color: "border-[#414849] text-[#414849] bg-[#f1f4f1]",
    what: "Mean squared error between predicted probabilities and actual outcomes. Lower is better.",
    why: null,
    whyNot: "Sensitive to calibration quality and class imbalance. Because we apply Bayesian odds re-calibration to correct for class-weight distortion in training, Brier scores are less comparable across models. We include it in the full table for completeness but do not use it for model selection.",
  },
];

export default function ModelComparisonPage() {
  const [activeTab, setActiveTab] = useState<"metrics" | "roc">("metrics");
  const [data, setData] = useState<Benchmarks | null>(null);
  const [live, setLive] = useState(true);

  useEffect(() => {
    fetchBenchmarks()
      .then((d) => {
        setData(d);
        setLive(true);
      })
      .catch((e) => {
        if (e instanceof ApiUnavailableError || e instanceof TypeError) setLive(false);
      });
  }, []);

  const rows = useMemo(() => (data ? toRows(data) : []), [data]);
  const champion = rows[0];

  return (
    <div className="flex flex-col w-full text-[#181c1a]">
      {/* Page Header */}
      <header className="pb-6">
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
          <div>
            <h1 className="font-serif text-[28px] leading-[36px] text-[#072427] font-medium">
              Model Comparison &amp; Evaluation
            </h1>
            <p className="font-sans text-[14px] text-[#414849] mt-1 max-w-[820px]">
              Seven models were trained on {TRAIN_ROWS.toLocaleString()} Home Credit loan applicants
              and evaluated using 5-fold cross-validation. Each metric below was chosen — or
              rejected — for a specific reason explained in the rationale section.
            </p>
          </div>
          <div className="flex items-center gap-2 self-start md:self-auto">
            <button
              type="button"
              onClick={() => window.print()}
              className="px-4 py-1.5 font-sans text-[13px] bg-transparent text-[#181c1a] hover:bg-[#ecefeb] border border-[#c1c8c8] rounded-[2px] transition-colors"
            >
              Print report
            </button>
            <span
              className={`px-3 py-1.5 font-mono text-[11px] border rounded-[2px] ${
                live
                  ? "text-[#39684a] bg-[#bbefc9]/30 border-[#39684a]"
                  : "text-[#414849] bg-[#ecefeb] border-[#c1c8c8]"
              }`}
            >
              {live ? "● Live model API" : "○ API offline — metrics unavailable"}
            </span>
          </div>
        </div>

        {/* Metadata strip */}
        <div className="mt-6 pt-3 pb-3 bg-[#f1f4f1] border border-[#c1c8c8] flex flex-wrap items-center justify-between gap-4 px-4">
          <div className="flex items-center gap-6 flex-wrap">
            <div>
              <span className="font-sans text-[11px] text-[#414849]">Evaluation method:</span>
              <span className="font-mono text-[12px] ml-1">5-Fold Stratified Out-of-Fold</span>
            </div>
            <div>
              <span className="font-sans text-[11px] text-[#414849]">Training cohort:</span>
              <span className="font-mono text-[12px] ml-1">{TRAIN_ROWS.toLocaleString()} applicants</span>
            </div>
            <div>
              <span className="font-sans text-[11px] text-[#414849]">Observed default rate:</span>
              <span className="font-mono text-[12px] ml-1">{(POPULATION_DEFAULT_RATE * 100).toFixed(2)}% (11.4:1 imbalance)</span>
            </div>
          </div>
          {champion && (
            <div className="flex items-center gap-1 font-mono text-[12px] text-[#39684a]">
              ✓ Champion: {champion.name} (AUC {champion.auc})
            </div>
          )}
        </div>
      </header>

      {/* ── METRIC RATIONALE SECTION ── */}
      <section className="mt-2 mb-8">
        <h2 className="font-serif text-[20px] text-[#072427] font-medium mb-1">
          Why these metrics — and why not others?
        </h2>
        <p className="font-sans text-[13px] text-[#414849] mb-4 max-w-[820px]">
          Choosing the wrong metric is one of the most common mistakes in ML for credit scoring. With
          only 8% of applicants defaulting, many standard metrics completely misrepresent how good a
          model actually is. Here is the reasoning behind each metric used in this project.
        </p>

        <div className="flex flex-col gap-3">
          {METRIC_RATIONALE.map((m) => (
            <div key={m.name} className="border border-[#c1c8c8] bg-[#ffffff]">
              <div className="px-4 py-3 flex flex-col sm:flex-row sm:items-center gap-3 border-b border-[#c1c8c8]">
                <span className="font-sans text-[14px] font-medium text-[#072427]">{m.name}</span>
                <span className={`self-start px-2 py-0.5 text-[11px] font-mono border rounded-[2px] ${m.color}`}>
                  {m.label}
                </span>
              </div>
              <div className="px-4 py-3 grid grid-cols-1 md:grid-cols-3 gap-4 text-[13px]">
                <div>
                  <span className="font-sans text-[11px] font-medium text-[#414849] uppercase tracking-wider block mb-1">
                    What it measures
                  </span>
                  <p className="font-sans text-[#414849] leading-relaxed">{m.what}</p>
                </div>
                {m.why && (
                  <div className="md:col-span-2">
                    <span className="font-sans text-[11px] font-medium text-[#39684a] uppercase tracking-wider block mb-1">
                      Why it was chosen
                    </span>
                    <p className="font-sans text-[#414849] leading-relaxed">{m.why}</p>
                  </div>
                )}
                {m.whyNot && (
                  <div className="md:col-span-2">
                    <span className="font-sans text-[11px] font-medium text-[#ba1a1a] uppercase tracking-wider block mb-1">
                      Why it was excluded / deprioritised
                    </span>
                    <p className="font-sans text-[#414849] leading-relaxed">{m.whyNot}</p>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── MODEL PERFORMANCE CHARTS ── */}
      <section className="mt-4">
        <h2 className="font-serif text-[20px] text-[#072427] font-medium mb-1">
          Model performance results
        </h2>
        <p className="font-sans text-[13px] text-[#414849] mb-4">
          All scores are out-of-fold (OOF) — computed on held-out data the model never saw during training.
        </p>

        <div className="flex items-center gap-4 border-b border-[#c1c8c8] bg-[#f7faf6]">
          <button
            type="button"
            onClick={() => setActiveTab("metrics")}
            className={`py-2 px-4 font-sans text-[13px] font-medium transition-colors border-b-2 ${
              activeTab === "metrics"
                ? "text-[#072427] border-[#1f3a3d] bg-[#f1f4f1]"
                : "text-[#414849] border-transparent hover:text-[#181c1a]"
            }`}
          >
            AUC / F1 / KS comparison
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("roc")}
            className={`py-2 px-4 font-sans text-[13px] font-medium transition-colors border-b-2 ${
              activeTab === "roc"
                ? "text-[#072427] border-[#1f3a3d] bg-[#f1f4f1]"
                : "text-[#414849] border-transparent hover:text-[#181c1a]"
            }`}
          >
            Gini (ranking power)
          </button>
        </div>

        {!rows.length && (
          <div className="mt-6 bg-[#ffffff] border border-[#c1c8c8] p-8 text-center font-sans text-[13px] text-[#414849]">
            {live
              ? "Loading benchmark metrics from the model API…"
              : "The FastAPI backend (localhost:8000) is offline — start it with `uvicorn server.api:app --reload` to display the trained-model benchmark here."}
          </div>
        )}

        {rows.length > 0 && activeTab === "metrics" && (
          <div className="mt-4 bg-[#ffffff] border border-[#c1c8c8] p-6">
            <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-2 pb-4">
              <div>
                <span className="font-serif text-[18px] text-[#072427] font-medium">
                  AUC-ROC · F1 · KS across all models
                </span>
                <p className="font-sans text-[12px] text-[#414849] mt-1">
                  Higher is better for all three. Random baseline = 0.50 AUC, 0.00 KS.
                </p>
              </div>
              <div className="flex items-center gap-4 font-mono text-[12px]">
                <div className="flex items-center gap-1">
                  <span className="w-3 h-3 bg-[#1f3a3d] inline-block" />
                  <span>AUC-ROC</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="w-3 h-3 bg-[#486366] inline-block" />
                  <span>F1 Score</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="w-3 h-3 bg-[#afcccf] inline-block" />
                  <span>KS Stat</span>
                </div>
              </div>
            </div>

            <div className="w-full overflow-x-auto pt-2">
              <svg viewBox="0 0 980 340" className="w-full h-auto min-w-[760px] font-sans" role="img">
                {[
                  { y: 20, val: "1.00" },
                  { y: 74, val: "0.80" },
                  { y: 128, val: "0.60" },
                  { y: 182, val: "0.40" },
                  { y: 236, val: "0.20" },
                  { y: 290, val: "0.00" },
                ].map((g) => (
                  <g key={g.val}>
                    <line x1="60" y1={g.y} x2="960" y2={g.y} stroke="#c1c8c8" strokeDasharray={g.val === "0.00" ? undefined : "2 2"} strokeWidth="1" />
                    <text x="50" y={g.y + 4} fill="#414849" fontSize="11" textAnchor="end" fontFamily="JetBrains Mono, monospace">
                      {g.val}
                    </text>
                  </g>
                ))}
                {rows.map((r, i) => {
                  const slot = 90 + i * (860 / Math.max(rows.length, 1));
                  const h = (v: number) => v * 270;
                  return (
                    <g key={r.key} transform={`translate(${slot - 52}, 0)`}>
                      <rect x="0" y={290 - h(r.aucNum)} width="32" height={h(r.aucNum)} fill="#1f3a3d" />
                      <text x="16" y={283 - h(r.aucNum)} fill="#1f3a3d" fontSize="10" textAnchor="middle" fontFamily="JetBrains Mono, monospace">
                        .{Math.round(r.aucNum * 1000).toString().padStart(3, "0").slice(-3)}
                      </text>
                      <rect x="36" y={290 - h(r.f1Num)} width="32" height={h(r.f1Num)} fill="#486366" />
                      <text x="52" y={283 - h(r.f1Num)} fill="#486366" fontSize="10" textAnchor="middle" fontFamily="JetBrains Mono, monospace">
                        .{Math.round(r.f1Num * 1000).toString().padStart(3, "0").slice(-3)}
                      </text>
                      <rect x="72" y={290 - h(r.ksNum)} width="32" height={h(r.ksNum)} fill="#afcccf" />
                      <text x="88" y={283 - h(r.ksNum)} fill="#181c1a" fontSize="10" textAnchor="middle" fontFamily="JetBrains Mono, monospace">
                        .{Math.round(r.ksNum * 1000).toString().padStart(3, "0").slice(-3)}
                      </text>
                      <text x="52" y="308" fill="#072427" fontSize="11" fontWeight="600" textAnchor="middle">
                        {r.name}
                      </text>
                      <text x="52" y="322" fill="#727879" fontSize="9" textAnchor="middle" fontFamily="JetBrains Mono, monospace">
                        {r.rank}
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>
          </div>
        )}

        {rows.length > 0 && activeTab === "roc" && (
          <div className="mt-4 bg-[#ffffff] border border-[#c1c8c8] p-6">
            <div className="pb-4">
              <span className="font-serif text-[18px] text-[#072427] font-medium">
                Gini coefficient — ranking power above random
              </span>
              <p className="font-sans text-[12px] text-[#414849] mt-1">
                Gini = 2 × AUC − 1. A Gini of 57% means the model ranks applicants 57% better than chance. Minimum acceptable for credit scoring: 30%.
              </p>
            </div>
            <div className="flex flex-col gap-3 pt-2">
              {rows.map((r) => {
                const gini = (2 * r.aucNum - 1) * 100;
                return (
                  <div key={r.key} className="flex items-center gap-3">
                    <span className="w-44 shrink-0 font-sans text-[13px] text-[#072427]">{r.name}</span>
                    <div className="flex-1 h-4 bg-[#f1f4f1] border border-[#c1c8c8]">
                      <div className="h-full bg-[#1f3a3d]" style={{ width: `${(gini / 60) * 100}%` }} />
                    </div>
                    <span className="w-16 text-right font-mono text-[12px] text-[#39684a] font-medium">{gini.toFixed(1)}%</span>
                  </div>
                );
              })}
              {/* Reference line label */}
              <div className="mt-2 flex items-center gap-3">
                <span className="w-44 shrink-0 font-sans text-[11px] text-[#414849]">Industry minimum</span>
                <div className="flex-1 h-0 border-t-2 border-dashed border-[#ba1a1a]" style={{ marginLeft: `${(30 / 60) * 100}%`, maxWidth: "1px" }} />
                <span className="font-mono text-[11px] text-[#ba1a1a]">30% floor</span>
              </div>
            </div>
          </div>
        )}
      </section>

      {/* ── FULL BENCHMARK TABLE ── */}
      <section className="mt-8 border border-[#c1c8c8] bg-[#ffffff]">
        <div className="px-4 py-3 bg-[#f1f4f1] border-b border-[#c1c8c8] flex items-center justify-between">
          <span className="font-sans text-[13px] text-[#181c1a] font-medium">
            Full benchmark table
          </span>
          <span className="font-mono text-[11px] text-[#414849]">
            {TRAIN_ROWS.toLocaleString()} applicants · 5-fold OOF
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-[#f7faf6] border-b border-[#c1c8c8] font-sans text-[11px] text-[#414849] uppercase tracking-wider">
                <th className="py-2.5 px-4 font-medium">#</th>
                <th className="py-2.5 px-4 font-medium">Model</th>
                <th className="py-2.5 px-4 font-medium">Architecture</th>
                <th className="py-2.5 px-4 font-medium text-right">AUC-ROC ↑</th>
                <th className="py-2.5 px-4 font-medium text-right">Stability (CV ±)</th>
                <th className="py-2.5 px-4 font-medium text-right">F1 ↑</th>
                <th className="py-2.5 px-4 font-medium text-right">KS ↑</th>
                <th className="py-2.5 px-4 font-medium text-right">PR-AUC ↑</th>
                <th className="py-2.5 px-4 font-medium text-right">Gini ↑</th>
                <th className="py-2.5 px-4 font-medium text-right">Brier ↓</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#c1c8c8] font-sans text-[13px] text-[#181c1a]">
              {rows.map((m) => (
                <tr key={m.key} className={`hover:bg-[#f1f4f1] transition-colors ${m.rank === "01" ? "bg-[#bbefc9]/10" : ""}`}>
                  <td className="py-3 px-4 font-mono text-[12px] text-[#414849]">{m.rank}</td>
                  <td className="py-3 px-4 font-medium text-[#072427]">
                    {m.name}
                    {m.rank === "01" && (
                      <span className="ml-2 px-1.5 py-0.5 font-mono text-[10px] text-[#39684a] border border-[#39684a] bg-[#bbefc9]/30 rounded-[2px]">
                        champion
                      </span>
                    )}
                  </td>
                  <td className="py-3 px-4 text-[#414849]">{m.architecture}</td>
                  <td className="py-3 px-4 font-mono text-right font-medium text-[#072427]">{m.auc}</td>
                  <td className="py-3 px-4 font-mono text-right text-[#414849]">{m.cvStd}</td>
                  <td className="py-3 px-4 font-mono text-right">{m.f1}</td>
                  <td className="py-3 px-4 font-mono text-right">{m.ks}</td>
                  <td className="py-3 px-4 font-mono text-right">{m.prAuc}</td>
                  <td className="py-3 px-4 font-mono text-right text-[#39684a] font-medium">{m.gini}</td>
                  <td className="py-3 px-4 font-mono text-right text-[#414849]">{m.brier}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Footer note */}
        <div className="px-4 py-3 border-t border-[#c1c8c8] bg-[#f7faf6]">
          <p className="font-sans text-[12px] text-[#414849] leading-relaxed">
            <span className="font-medium text-[#181c1a]">Note on F1 scores:</span> Values of 0.27–0.29 are mathematically expected with an 8.07% default rate.
            At this imbalance ratio, even a near-perfect model produces F1 in this range — it is not a sign of poor performance.
            The correct measure of discrimination quality is AUC-ROC and KS, not F1.
          </p>
        </div>
      </section>
    </div>
  );
}
