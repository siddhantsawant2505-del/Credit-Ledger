export default function ExplainabilityPage() {
  const shapFeatures = [
    {
      name: "Debt-to-income ratio",
      value: "28.4%",
      delta: -0.042,
      deltaText: "-0.042",
      type: "decrease",
      widthPct: 42,
      baseline: "35.0% cohort avg",
      directionNote: "Sub-30% DTI reduces default risk by 4.2 bps",
    },
    {
      name: "Delinquencies last 24 months",
      value: "0",
      delta: -0.038,
      deltaText: "-0.038",
      type: "decrease",
      widthPct: 38,
      baseline: "0.42 prior delinq",
      directionNote: "Clean 24m payment history reduces risk",
    },
    {
      name: "Employment tenure",
      value: "7.5 years",
      delta: -0.026,
      deltaText: "-0.026",
      type: "decrease",
      widthPct: 26,
      baseline: "3.2 years avg",
      directionNote: "Stable long-term employment tenure",
    },
    {
      name: "Home ownership status",
      value: "Mortgage",
      delta: -0.015,
      deltaText: "-0.015",
      type: "decrease",
      widthPct: 15,
      baseline: "Tenant lease",
      directionNote: "Mortgage tenure demonstrates equity stability",
    },
    {
      name: "Revolving credit utilization",
      value: "41.2%",
      delta: 0.031,
      deltaText: "+0.031",
      type: "increase",
      widthPct: 31,
      baseline: "28.0% target",
      directionNote: "Utilization above 40% threshold adds 3.1 bps risk",
    },
    {
      name: "Recent credit inquiries",
      value: "3 in 6m",
      delta: 0.018,
      deltaText: "+0.018",
      type: "increase",
      widthPct: 18,
      baseline: "1 inquiry avg",
      directionNote: "Multiple inquiries indicate short-term credit seeking",
    },
    {
      name: "Requested loan amount",
      value: "$24,000",
      delta: 0.006,
      deltaText: "+0.006",
      type: "increase",
      widthPct: 6,
      baseline: "$18,500 mean",
      directionNote: "Slight principal exposure elevation",
    },
  ];

  return (
    <div className="flex flex-col w-full text-[#181c1a]">
      {/* Dossier Header & Context */}
      <div className="pb-6 border-b border-[#c1c8c8]">
        <div className="flex flex-col md:flex-row md:items-baseline justify-between gap-2 mb-1">
          <h1 className="font-serif text-[28px] leading-[36px] text-[#072427] font-medium tracking-tight">
            Feature Attribution &amp; Local SHAP Analysis
          </h1>
          <span className="font-mono text-[12px] text-[#414849]">
            Model: LightGBM v3.1 | Seed: #4092
          </span>
        </div>
        <p className="font-sans text-[14px] text-[#414849] max-w-[850px] mb-4">
          Deconstruction of default risk factors for Application #CR-948201 using Shapley additive explanations (LightGBM v3.1).
        </p>

        {/* Actuarial Baseline Metric Block */}
        <div className="p-4 bg-[#ecefeb] border border-[#c1c8c8] flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-6 flex-wrap">
            <div>
              <span className="font-sans text-[11px] text-[#414849] block">
                Base default expectation E[f(x)]
              </span>
              <span className="font-mono text-[14px] text-[#181c1a] font-medium">
                21.40%
              </span>
            </div>
            <div className="h-8 w-px bg-[#c1c8c8] hidden sm:block"></div>
            <div>
              <span className="font-sans text-[11px] text-[#414849] block">
                Applicant calculated score
              </span>
              <span className="font-mono text-[14px] text-[#072427] font-medium">
                14.80%
              </span>
            </div>
            <div className="h-8 w-px bg-[#c1c8c8] hidden sm:block"></div>
            <div>
              <span className="font-sans text-[11px] text-[#414849] block">
                Total net displacement
              </span>
              <span className="font-mono text-[14px] text-[#39684a] font-medium">
                –6.60% (–0.0660)
              </span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-[#39684a]"></span>
            <span className="font-sans text-[11px] text-[#414849] font-medium">
              Adjudication posture: Favorable
            </span>
          </div>
        </div>
      </div>

      {/* Legend & Section Title */}
      <div className="pt-6 pb-2 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div className="font-serif text-[18px] text-[#072427] font-medium">
          Diverging SHAP Vector Graph
        </div>
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-1.5">
            <div className="w-3.5 h-2 bg-[#4B7A5B]"></div>
            <span className="font-sans text-[11px] text-[#414849]">
              Decreases default risk
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3.5 h-2 bg-[#A8432E]"></div>
            <span className="font-sans text-[11px] text-[#414849]">
              Increases default risk
            </span>
          </div>
        </div>
      </div>

      {/* Diverging Bar Chart Matrix */}
      <div className="py-4 border-b border-[#c1c8c8] overflow-x-auto">
        <div className="min-w-[760px] flex flex-col">
          {/* Scale Header Guidelines */}
          <div className="grid grid-cols-12 gap-0 pb-2 font-mono text-[12px] text-[#414849] border-b border-[#c1c8c8]">
            <div className="col-span-5 text-left font-sans text-[11px] uppercase tracking-wider font-medium">
              Feature identifier &amp; raw value
            </div>
            <div className="col-span-7 relative">
              <div className="flex justify-between w-full text-center">
                <span className="w-12 text-left">–0.05</span>
                <span className="w-12">–0.03</span>
                <span className="w-12">–0.01</span>
                <span className="w-12 font-medium text-[#181c1a]">0.00</span>
                <span className="w-12">+0.01</span>
                <span className="w-12">+0.03</span>
                <span className="w-12 text-right">+0.05</span>
              </div>
            </div>
          </div>

          {/* Diverging Rows */}
          <div className="divide-y divide-[#c1c8c8] relative">
            {/* Center hairline through canvas */}
            <div className="absolute left-[calc(5/12*100%+7/12*50%)] top-0 bottom-0 w-px bg-[#c1c8c8] pointer-events-none z-10"></div>

            {shapFeatures.map((f, i) => (
              <div key={i} className="grid grid-cols-12 items-center py-3 hover:bg-[#f1f4f1] transition-colors">
                <div className="col-span-5 pr-4 flex items-baseline justify-between">
                  <span className="font-sans text-[14px] text-[#181c1a]">{f.name}</span>
                  <span className="font-mono text-[12px] text-[#414849]">{f.value}</span>
                </div>
                <div className="col-span-7 relative h-7 flex items-center">
                  {f.type === "decrease" ? (
                    <>
                      <div
                        className="absolute right-[50%] h-[8px] bg-[#4B7A5B]"
                        style={{ width: `${f.widthPct * 0.85}%` }}
                      ></div>
                      <span
                        className="absolute font-mono text-[12px] text-[#4B7A5B] font-medium"
                        style={{ right: `calc(50% + ${f.widthPct * 0.85}% + 8px)` }}
                      >
                        {f.deltaText}
                      </span>
                    </>
                  ) : (
                    <>
                      <div
                        className="absolute left-[50%] h-[8px] bg-[#A8432E]"
                        style={{ width: `${f.widthPct * 0.85}%` }}
                      ></div>
                      <span
                        className="absolute font-mono text-[12px] text-[#A8432E] font-medium"
                        style={{ left: `calc(50% + ${f.widthPct * 0.85}% + 8px)` }}
                      >
                        {f.deltaText}
                      </span>
                    </>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Feature Attribution Table */}
      <section className="mt-8 border border-[#c1c8c8] bg-[#ffffff]">
        <div className="px-4 py-3 bg-[#f1f4f1] border-b border-[#c1c8c8] flex items-center justify-between">
          <span className="font-sans text-[13px] text-[#181c1a] font-medium">
            Detailed Shapley Value Ledger
          </span>
          <span className="font-mono text-[11px] text-[#414849]">
            7 Contributing Vectors Analyzed
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-[#f7faf6] border-b border-[#c1c8c8] font-sans text-[11px] text-[#414849] uppercase tracking-wider">
                <th className="py-2.5 px-4 font-medium">Feature Identifier</th>
                <th className="py-2.5 px-4 font-medium">Applicant Value</th>
                <th className="py-2.5 px-4 font-medium text-right">SHAP Delta</th>
                <th className="py-2.5 px-4 font-medium">Cohort Benchmark</th>
                <th className="py-2.5 px-4 font-medium">Directional Rationale</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#c1c8c8] font-sans text-[13px] text-[#181c1a]">
              {shapFeatures.map((f, i) => (
                <tr key={i} className="hover:bg-[#f1f4f1] transition-colors">
                  <td className="py-3 px-4 font-medium text-[#072427]">{f.name}</td>
                  <td className="py-3 px-4 font-mono text-[12px]">{f.value}</td>
                  <td className={`py-3 px-4 font-mono text-right font-medium ${f.type === "decrease" ? "text-[#4B7A5B]" : "text-[#A8432E]"}`}>
                    {f.deltaText}
                  </td>
                  <td className="py-3 px-4 font-mono text-[12px] text-[#414849]">{f.baseline}</td>
                  <td className="py-3 px-4 text-[#414849] text-[12px]">{f.directionNote}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
