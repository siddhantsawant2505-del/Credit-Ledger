"use client";

import { useState } from "react";

export default function ModelComparisonPage() {
  const [activeTab, setActiveTab] = useState<"metrics" | "roc">("metrics");

  const models = [
    {
      rank: "01",
      name: "LightGBM v3.1",
      tag: "CHAMPION",
      architecture: "Gradient Boosted Decision Trees",
      auc: "0.842",
      f1: "0.768",
      ks: "0.541",
      gini: "68.4%",
      brier: "0.0184",
      status: "Production Champion",
      statusClass: "text-[#39684a] bg-[#bbefc9]/30 border-[#39684a]",
    },
    {
      rank: "02",
      name: "CatBoost v1.2",
      tag: "CHALLENGER A",
      architecture: "Categorical Oblivious Trees",
      auc: "0.838",
      f1: "0.761",
      ks: "0.534",
      gini: "67.6%",
      brier: "0.0189",
      status: "Hot Standby",
      statusClass: "text-[#1f3a3d] bg-[#f1f4f1] border-[#1f3a3d]",
    },
    {
      rank: "03",
      name: "XGBoost v2.0",
      tag: "CHALLENGER B",
      architecture: "Regularized Gradient Boosting",
      auc: "0.835",
      f1: "0.758",
      ks: "0.529",
      gini: "67.0%",
      brier: "0.0192",
      status: "Validated",
      statusClass: "text-[#414849] bg-[#ecefeb] border-[#c1c8c8]",
    },
    {
      rank: "04",
      name: "Random Forest",
      tag: "BENCHMARK",
      architecture: "Ensemble Bagged Trees (1000 estimators)",
      auc: "0.812",
      f1: "0.734",
      ks: "0.492",
      gini: "62.4%",
      brier: "0.0215",
      status: "Legacy Benchmark",
      statusClass: "text-[#414849] bg-[#ecefeb] border-[#c1c8c8]",
    },
    {
      rank: "05",
      name: "Neural Net (MLP)",
      tag: "EXPERIMENTAL",
      architecture: "Deep Multi-Layer Perceptron (3 hidden)",
      auc: "0.798",
      f1: "0.718",
      ks: "0.468",
      gini: "59.6%",
      brier: "0.0238",
      status: "Research Only",
      statusClass: "text-[#414849] bg-[#ecefeb] border-[#c1c8c8]",
    },
    {
      rank: "06",
      name: "Logistic Regression",
      tag: "BASE LINE",
      architecture: "L2 Penalized GLM (Binomial)",
      auc: "0.765",
      f1: "0.684",
      ks: "0.412",
      gini: "53.0%",
      brier: "0.0271",
      status: "Regulatory Baseline",
      statusClass: "text-[#414849] bg-[#ecefeb] border-[#c1c8c8]",
    },
  ];

  return (
    <div className="flex flex-col w-full text-[#181c1a]">
      {/* Header & Actuarial Summary */}
      <header className="pb-6">
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
          <div>
            <div className="font-mono text-[11px] text-[#414849] tracking-wider uppercase mb-1">
              Validation Registry · Section 04 / Ledger Entry
            </div>
            <h1 className="font-serif text-[28px] leading-[36px] text-[#072427] font-medium">
              Model Comparison &amp; Discrimination Benchmark
            </h1>
            <p className="font-sans text-[14px] text-[#414849] mt-1">
              Quantitative comparison across six production candidate architectures validated against the Q3 holdout sample.
            </p>
          </div>

          <div className="flex items-center gap-2 self-start md:self-auto">
            <button
              type="button"
              onClick={() => window.print()}
              className="px-4 py-1.5 font-sans text-[13px] bg-transparent text-[#181c1a] hover:bg-[#ecefeb] border border-[#c1c8c8] rounded-[2px] transition-colors"
            >
              Print ledger extract
            </button>
            <button
              type="button"
              className="px-4 py-1.5 font-sans text-[13px] bg-[#1f3a3d] text-[#ffffff] hover:bg-[#072427] rounded-[2px] transition-colors"
            >
              Export validation pack (.csv)
            </button>
          </div>
        </div>

        {/* Actuarial Metadata Strip */}
        <div className="mt-6 pt-3 pb-3 bg-[#f1f4f1] border border-[#c1c8c8] flex flex-wrap items-center justify-between gap-4 px-4 text-[#181c1a]">
          <div className="flex items-center gap-6 flex-wrap">
            <div>
              <span className="font-sans text-[11px] text-[#414849]">Holdout cohort:</span>
              <span className="font-mono text-[12px] ml-1">2024-Q3 (N = 142,850)</span>
            </div>
            <div>
              <span className="font-sans text-[11px] text-[#414849]">Target definition:</span>
              <span className="font-mono text-[12px] ml-1">90+ DPD within 12M</span>
            </div>
            <div>
              <span className="font-sans text-[11px] text-[#414849]">Prior default rate:</span>
              <span className="font-mono text-[12px] ml-1">3.42%</span>
            </div>
          </div>
          <div className="flex items-center gap-1 font-mono text-[12px] text-[#39684a]">
            ✓ Independent validation signed
          </div>
        </div>
      </header>

      {/* View Switcher Tabs */}
      <section className="mt-4">
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
            Grouped metrics (AUC / F1 / KS)
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
            Overlaid ROC curves (FPR vs TPR)
          </button>
        </div>

        {/* TAB 1: Grouped Discrimination Metrics Chart */}
        {activeTab === "metrics" && (
          <div className="mt-6 bg-[#ffffff] border border-[#c1c8c8] p-6">
            <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-2 pb-4">
              <div>
                <span className="font-serif text-[18px] text-[#072427] font-medium">
                  Holdout discrimination metrics across candidates
                </span>
                <p className="font-sans text-[12px] text-[#414849] mt-1">
                  Measured on out-of-time sample. Baseline random guess = 0.50 AUC, 0.00 KS.
                </p>
              </div>

              {/* Legend */}
              <div className="flex items-center gap-4 font-mono text-[12px]">
                <div className="flex items-center gap-1">
                  <span className="w-3 h-3 bg-[#1f3a3d] inline-block"></span>
                  <span>AUC-ROC</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="w-3 h-3 bg-[#486366] inline-block"></span>
                  <span>F1 Score</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="w-3 h-3 bg-[#afcccf] inline-block"></span>
                  <span>KS Stat</span>
                </div>
              </div>
            </div>

            {/* SVG Bar Chart */}
            <div className="w-full overflow-x-auto pt-2">
              <svg viewBox="0 0 980 340" className="w-full h-auto min-w-[760px] font-sans" role="img">
                {/* Grid Lines */}
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

                {/* Model 1: LightGBM v3.1 */}
                <g transform="translate(90, 0)">
                  <rect x="0" y="62.6" width="32" height="227.4" fill="#1f3a3d" />
                  <text x="16" y="55" fill="#1f3a3d" fontSize="10" textAnchor="middle" fontFamily="JetBrains Mono, monospace">.842</text>
                  <rect x="36" y="82.6" width="32" height="207.4" fill="#486366" />
                  <text x="52" y="75" fill="#486366" fontSize="10" textAnchor="middle" fontFamily="JetBrains Mono, monospace">.768</text>
                  <rect x="72" y="143.9" width="32" height="146.1" fill="#afcccf" />
                  <text x="88" y="137" fill="#181c1a" fontSize="10" textAnchor="middle" fontFamily="JetBrains Mono, monospace">.541</text>
                  <text x="52" y="308" fill="#072427" fontSize="12" fontWeight="600" textAnchor="middle">LightGBM v3.1</text>
                  <text x="52" y="322" fill="#727879" fontSize="10" textAnchor="middle" fontFamily="JetBrains Mono, monospace">CHAMPION</text>
                </g>

                {/* Model 2: CatBoost v1.2 */}
                <g transform="translate(236, 0)">
                  <rect x="0" y="63.7" width="32" height="226.3" fill="#1f3a3d" />
                  <text x="16" y="56" fill="#1f3a3d" fontSize="10" textAnchor="middle" fontFamily="JetBrains Mono, monospace">.838</text>
                  <rect x="36" y="84.5" width="32" height="205.5" fill="#486366" />
                  <text x="52" y="77" fill="#486366" fontSize="10" textAnchor="middle" fontFamily="JetBrains Mono, monospace">.761</text>
                  <rect x="72" y="145.8" width="32" height="144.2" fill="#afcccf" />
                  <text x="88" y="139" fill="#181c1a" fontSize="10" textAnchor="middle" fontFamily="JetBrains Mono, monospace">.534</text>
                  <text x="52" y="308" fill="#181c1a" fontSize="12" fontWeight="500" textAnchor="middle">CatBoost v1.2</text>
                  <text x="52" y="322" fill="#727879" fontSize="10" textAnchor="middle" fontFamily="JetBrains Mono, monospace">CANDIDATE</text>
                </g>

                {/* Model 3: XGBoost v2.0 */}
                <g transform="translate(382, 0)">
                  <rect x="0" y="64.55" width="32" height="225.45" fill="#1f3a3d" />
                  <text x="16" y="57" fill="#1f3a3d" fontSize="10" textAnchor="middle" fontFamily="JetBrains Mono, monospace">.835</text>
                  <rect x="36" y="85.34" width="32" height="204.66" fill="#486366" />
                  <text x="52" y="78" fill="#486366" fontSize="10" textAnchor="middle" fontFamily="JetBrains Mono, monospace">.758</text>
                  <rect x="72" y="147.17" width="32" height="142.83" fill="#afcccf" />
                  <text x="88" y="140" fill="#181c1a" fontSize="10" textAnchor="middle" fontFamily="JetBrains Mono, monospace">.529</text>
                  <text x="52" y="308" fill="#181c1a" fontSize="12" fontWeight="500" textAnchor="middle">XGBoost v2.0</text>
                  <text x="52" y="322" fill="#727879" fontSize="10" textAnchor="middle" fontFamily="JetBrains Mono, monospace">CANDIDATE</text>
                </g>

                {/* Model 4: Random Forest */}
                <g transform="translate(528, 0)">
                  <rect x="0" y="70.76" width="32" height="219.24" fill="#1f3a3d" opacity="0.8" />
                  <text x="16" y="63" fill="#1f3a3d" fontSize="10" textAnchor="middle" fontFamily="JetBrains Mono, monospace">.812</text>
                  <rect x="36" y="91.82" width="32" height="198.18" fill="#486366" opacity="0.8" />
                  <text x="52" y="84" fill="#486366" fontSize="10" textAnchor="middle" fontFamily="JetBrains Mono, monospace">.734</text>
                  <rect x="72" y="157.16" width="32" height="132.84" fill="#afcccf" opacity="0.8" />
                  <text x="88" y="150" fill="#181c1a" fontSize="10" textAnchor="middle" fontFamily="JetBrains Mono, monospace">.492</text>
                  <text x="52" y="308" fill="#181c1a" fontSize="12" fontWeight="500" textAnchor="middle">Random Forest</text>
                  <text x="52" y="322" fill="#727879" fontSize="10" textAnchor="middle" fontFamily="JetBrains Mono, monospace">BENCHMARK</text>
                </g>

                {/* Model 5: Neural Net */}
                <g transform="translate(674, 0)">
                  <rect x="0" y="74.54" width="32" height="215.46" fill="#1f3a3d" opacity="0.6" />
                  <text x="16" y="67" fill="#1f3a3d" fontSize="10" textAnchor="middle" fontFamily="JetBrains Mono, monospace">.798</text>
                  <rect x="36" y="96.14" width="32" height="193.86" fill="#486366" opacity="0.6" />
                  <text x="52" y="88" fill="#486366" fontSize="10" textAnchor="middle" fontFamily="JetBrains Mono, monospace">.718</text>
                  <rect x="72" y="163.64" width="32" height="126.36" fill="#afcccf" opacity="0.6" />
                  <text x="88" y="156" fill="#181c1a" fontSize="10" textAnchor="middle" fontFamily="JetBrains Mono, monospace">.468</text>
                  <text x="52" y="308" fill="#181c1a" fontSize="12" fontWeight="500" textAnchor="middle">Neural Net (MLP)</text>
                  <text x="52" y="322" fill="#727879" fontSize="10" textAnchor="middle" fontFamily="JetBrains Mono, monospace">RESEARCH</text>
                </g>

                {/* Model 6: Logistic Regression */}
                <g transform="translate(820, 0)">
                  <rect x="0" y="83.45" width="32" height="206.55" fill="#1f3a3d" opacity="0.4" />
                  <text x="16" y="76" fill="#1f3a3d" fontSize="10" textAnchor="middle" fontFamily="JetBrains Mono, monospace">.765</text>
                  <rect x="36" y="105.32" width="32" height="184.68" fill="#486366" opacity="0.4" />
                  <text x="52" y="98" fill="#486366" fontSize="10" textAnchor="middle" fontFamily="JetBrains Mono, monospace">.684</text>
                  <rect x="72" y="178.76" width="32" height="111.24" fill="#afcccf" opacity="0.4" />
                  <text x="88" y="171" fill="#181c1a" fontSize="10" textAnchor="middle" fontFamily="JetBrains Mono, monospace">.412</text>
                  <text x="52" y="308" fill="#181c1a" fontSize="12" fontWeight="500" textAnchor="middle">Logistic Reg</text>
                  <text x="52" y="322" fill="#727879" fontSize="10" textAnchor="middle" fontFamily="JetBrains Mono, monospace">BASELINE</text>
                </g>
              </svg>
            </div>
          </div>
        )}

        {/* TAB 2: Overlaid ROC Curves */}
        {activeTab === "roc" && (
          <div className="mt-6 bg-[#ffffff] border border-[#c1c8c8] p-6">
            <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-2 pb-4">
              <div>
                <span className="font-serif text-[18px] text-[#072427] font-medium">
                  Overlaid Receiver Operating Characteristic (ROC) Curves
                </span>
                <p className="font-sans text-[12px] text-[#414849] mt-1">
                  True Positive Rate (Sensitivity) vs False Positive Rate (1 - Specificity) across operating thresholds.
                </p>
              </div>

              <div className="flex items-center gap-4 font-mono text-[11px]">
                <div className="flex items-center gap-1 text-[#072427]">
                  <span className="w-3 h-0.5 bg-[#072427] inline-block"></span>
                  <span>LightGBM (.842)</span>
                </div>
                <div className="flex items-center gap-1 text-[#1f3a3d]">
                  <span className="w-3 h-0.5 bg-[#1f3a3d] inline-block"></span>
                  <span>CatBoost (.838)</span>
                </div>
                <div className="flex items-center gap-1 text-[#39684a]">
                  <span className="w-3 h-0.5 bg-[#39684a] inline-block"></span>
                  <span>XGBoost (.835)</span>
                </div>
                <div className="flex items-center gap-1 text-[#727879]">
                  <span className="w-3 h-0.5 bg-[#727879] inline-block"></span>
                  <span>LogReg (.765)</span>
                </div>
              </div>
            </div>

            {/* SVG Line Chart for ROC */}
            <div className="w-full overflow-x-auto pt-2">
              <svg viewBox="0 0 600 400" className="w-full h-auto max-w-[700px] mx-auto font-sans" role="img">
                {/* Background & Axes */}
                <rect x="50" y="20" width="500" height="320" fill="#f7faf6" stroke="#c1c8c8" strokeWidth="1" />

                {/* Diagonal Random Baseline */}
                <line x1="50" y1="340" x2="550" y2="20" stroke="#c1c8c8" strokeDasharray="4 4" strokeWidth="1" />

                {/* LightGBM ROC Curve */}
                <path
                  d="M 50 340 Q 70 80, 200 45 T 550 20"
                  fill="none"
                  stroke="#072427"
                  strokeWidth="2.5"
                />

                {/* CatBoost ROC Curve */}
                <path
                  d="M 50 340 Q 80 95, 220 52 T 550 20"
                  fill="none"
                  stroke="#1f3a3d"
                  strokeWidth="1.8"
                />

                {/* XGBoost ROC Curve */}
                <path
                  d="M 50 340 Q 90 110, 240 60 T 550 20"
                  fill="none"
                  stroke="#39684a"
                  strokeWidth="1.5"
                />

                {/* Logistic Regression ROC Curve */}
                <path
                  d="M 50 340 Q 150 180, 320 100 T 550 20"
                  fill="none"
                  stroke="#727879"
                  strokeWidth="1.2"
                />

                {/* Axes Labels */}
                <text x="300" y="380" fill="#414849" fontSize="12" textAnchor="middle" fontFamily="IBM Plex Sans, sans-serif">
                  False Positive Rate (FPR)
                </text>
                <text x="15" y="180" fill="#414849" fontSize="12" textAnchor="middle" fontFamily="IBM Plex Sans, sans-serif" transform="rotate(-90 15 180)">
                  True Positive Rate (TPR)
                </text>

                {/* Tick Labels */}
                <text x="50" y="355" fill="#414849" fontSize="10" textAnchor="middle" fontFamily="JetBrains Mono">0.0</text>
                <text x="175" y="355" fill="#414849" fontSize="10" textAnchor="middle" fontFamily="JetBrains Mono">0.25</text>
                <text x="300" y="355" fill="#414849" fontSize="10" textAnchor="middle" fontFamily="JetBrains Mono">0.50</text>
                <text x="425" y="355" fill="#414849" fontSize="10" textAnchor="middle" fontFamily="JetBrains Mono">0.75</text>
                <text x="550" y="355" fill="#414849" fontSize="10" textAnchor="middle" fontFamily="JetBrains Mono">1.0</text>

                <text x="40" y="344" fill="#414849" fontSize="10" textAnchor="end" fontFamily="JetBrains Mono">0.0</text>
                <text x="40" y="260" fill="#414849" fontSize="10" textAnchor="end" fontFamily="JetBrains Mono">0.25</text>
                <text x="40" y="180" fill="#414849" fontSize="10" textAnchor="end" fontFamily="JetBrains Mono">0.50</text>
                <text x="40" y="100" fill="#414849" fontSize="10" textAnchor="end" fontFamily="JetBrains Mono">0.75</text>
                <text x="40" y="24" fill="#414849" fontSize="10" textAnchor="end" fontFamily="JetBrains Mono">1.0</text>
              </svg>
            </div>
          </div>
        )}
      </section>

      {/* Comprehensive Benchmark Table */}
      <section className="mt-8 border border-[#c1c8c8] bg-[#ffffff]">
        <div className="px-4 py-3 bg-[#f1f4f1] border-b border-[#c1c8c8] flex items-center justify-between">
          <span className="font-sans text-[13px] text-[#181c1a] font-medium">
            Candidate Discrimination Matrix &amp; Gini Ledger
          </span>
          <span className="font-mono text-[11px] text-[#414849]">
            Cohort Validation: 142,850 Records
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-[#f7faf6] border-b border-[#c1c8c8] font-sans text-[11px] text-[#414849] uppercase tracking-wider">
                <th className="py-2.5 px-4 font-medium">Rank</th>
                <th className="py-2.5 px-4 font-medium">Model Candidate</th>
                <th className="py-2.5 px-4 font-medium">Architecture Specification</th>
                <th className="py-2.5 px-4 font-medium text-right">AUC-ROC</th>
                <th className="py-2.5 px-4 font-medium text-right">F1 Score</th>
                <th className="py-2.5 px-4 font-medium text-right">KS Stat</th>
                <th className="py-2.5 px-4 font-medium text-right">Gini Coeff</th>
                <th className="py-2.5 px-4 font-medium text-right">Brier Loss</th>
                <th className="py-2.5 px-4 font-medium text-right">Deployment Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#c1c8c8] font-sans text-[13px] text-[#181c1a]">
              {models.map((m) => (
                <tr key={m.rank} className="hover:bg-[#f1f4f1] transition-colors">
                  <td className="py-3 px-4 font-mono text-[12px] text-[#414849]">{m.rank}</td>
                  <td className="py-3 px-4 font-medium text-[#072427]">{m.name}</td>
                  <td className="py-3 px-4 text-[#414849]">{m.architecture}</td>
                  <td className="py-3 px-4 font-mono text-right font-medium text-[#072427]">{m.auc}</td>
                  <td className="py-3 px-4 font-mono text-right">{m.f1}</td>
                  <td className="py-3 px-4 font-mono text-right">{m.ks}</td>
                  <td className="py-3 px-4 font-mono text-right text-[#39684a] font-medium">{m.gini}</td>
                  <td className="py-3 px-4 font-mono text-right text-[#414849]">{m.brier}</td>
                  <td className="py-3 px-4 text-right">
                    <span className={`inline-block px-2.5 py-0.5 border font-mono text-[11px] font-medium uppercase ${m.statusClass}`}>
                      {m.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
