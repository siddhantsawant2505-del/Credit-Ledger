"use client";

import { useEffect, useState } from "react";
import {
  ApiUnavailableError,
  fetchExplainability,
  type ExplainabilityPayload,
} from "../../lib/api";

// Human-readable labels and plain-English explanations for known features
const FEATURE_META: Record<string, { label: string; plain: string }> = {
  EXT_SOURCES_MEAN: {
    label: "External Credit Score (Average)",
    plain: "Average of three external credit bureau scores. The single most predictive signal — higher is better. Think of it like a combined credit score from Experian, Equifax, and TransUnion.",
  },
  EXT_SOURCE_2: {
    label: "External Credit Score 2",
    plain: "One of three credit bureau assessments. Reflects repayment history across existing loans and credit lines.",
  },
  EXT_SOURCE_3: {
    label: "External Credit Score 3",
    plain: "A second credit bureau signal. Strongly correlated with likelihood of default — applicants with low scores here are 3–4× more likely to default.",
  },
  EXT_SOURCE_1: {
    label: "External Credit Score 1",
    plain: "Third external bureau score. Slightly less correlated than sources 2 and 3, but still among the top predictors.",
  },
  EXT_SOURCES_WEIGHTED: {
    label: "Weighted Credit Score",
    plain: "A single combined score giving more weight to the most predictive bureau sources. Engineered from all three bureau signals.",
  },
  EXT_SOURCES_PROD_1_2_3: {
    label: "Credit Score Interaction",
    plain: "The product of all three bureau scores. Captures cases where all three are simultaneously low — which is far riskier than any one being low alone.",
  },
  DAYS_BIRTH: {
    label: "Applicant Age",
    plain: "Older applicants statistically default less often. Younger applicants (especially under 25) show elevated risk — likely due to shorter credit histories.",
  },
  DAYS_EMPLOYED: {
    label: "Employment Duration",
    plain: "How long the applicant has been continuously employed. Longer tenures signal stability. Short durations (under 1 year) are a risk flag.",
  },
  AMT_CREDIT: {
    label: "Loan Amount Requested",
    plain: "The total credit amount being applied for. Larger loans carry more exposure, especially when income or external scores are low.",
  },
  AMT_ANNUITY: {
    label: "Monthly Repayment Amount",
    plain: "The monthly instalment for the loan. High annuity relative to income is a strong indicator of repayment stress.",
  },
  AMT_INCOME_TOTAL: {
    label: "Annual Income",
    plain: "Total declared annual income. Higher income generally lowers default risk, though it must be evaluated against the loan size.",
  },
  PAYMENT_RATE: {
    label: "Payment-to-Loan Ratio",
    plain: "Monthly repayment divided by total loan amount. A high ratio means the applicant repays the loan faster — lower default risk. Engineered feature.",
  },
  DTI_RATIO: {
    label: "Debt-to-Income Ratio",
    plain: "Loan annuity as a fraction of income. Values above 40% are a recognised risk flag in credit underwriting globally.",
  },
  EMPLOYED_TO_AGE_RATIO: {
    label: "Career Stability Index",
    plain: "Proportion of working life spent in current employment. A value near 1.0 means continuous employment — very low risk.",
  },
  INCOME_PER_PERSON: {
    label: "Income per Family Member",
    plain: "Household income divided by family size. Accounts for financial obligations to dependants — lower values increase repayment stress.",
  },
};

function getFeatureMeta(rawName: string) {
  if (FEATURE_META[rawName]) return FEATURE_META[rawName];
  // Fallback: clean up raw name into readable form
  const label = rawName
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
  return {
    label,
    plain: "Derived from the Home Credit application data. Contributes to the model's prediction of repayment likelihood.",
  };
}

export default function ExplainabilityPage() {
  const [data, setData] = useState<ExplainabilityPayload | null>(null);
  const [live, setLive] = useState(true);

  useEffect(() => {
    fetchExplainability()
      .then((d) => {
        setData(d);
        setLive(true);
      })
      .catch((e) => {
        if (e instanceof ApiUnavailableError || e instanceof TypeError) setLive(false);
      });
  }, []);

  const maxShare = data?.features?.[0]?.gainShare ?? 1;

  return (
    <div className="flex flex-col w-full text-[#181c1a]">
      {/* Page Header */}
      <div className="pb-6 border-b border-[#c1c8c8]">
        <h1 className="font-serif text-[28px] leading-[36px] text-[#072427] font-medium tracking-tight">
          What drives the model's decisions?
        </h1>
        <p className="font-sans text-[15px] text-[#414849] max-w-[820px] mt-2 leading-relaxed">
          The LightGBM champion model was trained on 307,511 loan applications. The chart below shows
          which pieces of information it relies on most heavily — and by how much — when estimating
          the probability that an applicant will default.
        </p>
        <p className="font-sans text-[13px] text-[#414849] max-w-[820px] mt-2 leading-relaxed">
          <span className="font-medium text-[#181c1a]">How to read this:</span> Each bar represents
          a feature (a piece of data about the applicant). The longer the bar, the more that feature
          influences the model's output. The percentage is the feature's share of the total
          &quot;learning signal&quot; across all 450+ decision trees.
        </p>

        {/* Live model benchmark strip */}
        {data?.benchmark && (
          <div className="mt-4 p-4 bg-[#ecefeb] border border-[#c1c8c8] flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-6 flex-wrap">
              <div>
                <span className="font-sans text-[11px] text-[#414849] block">Model accuracy (AUC)</span>
                <span className="font-mono text-[14px] text-[#072427] font-medium">
                  {data.benchmark.aucRoc?.toFixed(4) ?? "—"}
                  <span className="font-sans text-[11px] text-[#414849] ml-1 font-normal">out of 1.0</span>
                </span>
              </div>
              <div className="h-8 w-px bg-[#c1c8c8] hidden sm:block" />
              <div>
                <span className="font-sans text-[11px] text-[#414849] block">Separation power (KS)</span>
                <span className="font-mono text-[14px] text-[#072427] font-medium">
                  {data.benchmark.ksStat ? `${(data.benchmark.ksStat * 100).toFixed(1)}%` : "—"}
                  <span className="font-sans text-[11px] text-[#414849] ml-1 font-normal">gap between good/bad</span>
                </span>
              </div>
              <div className="h-8 w-px bg-[#c1c8c8] hidden sm:block" />
              <div>
                <span className="font-sans text-[11px] text-[#414849] block">Precision on defaulters (PR-AUC)</span>
                <span className="font-mono text-[14px] text-[#39684a] font-medium">
                  {data.benchmark.prAuc?.toFixed(4) ?? "—"}
                  <span className="font-sans text-[11px] text-[#414849] ml-1 font-normal">vs 0.08 random baseline</span>
                </span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[#39684a]" />
              <span className="font-sans text-[11px] text-[#414849] font-medium">
                Live from trained model
              </span>
            </div>
          </div>
        )}
      </div>

      {!data && (
        <div className="mt-8 bg-[#ffffff] border border-[#c1c8c8] p-8 text-center font-sans text-[13px] text-[#414849]">
          {live
            ? "Loading feature importances from the model API…"
            : "The model API (localhost:8000) is offline. Start it with `uvicorn server.api:app --reload` from the project root to see the real feature importances here."}
        </div>
      )}

      {data && (
        <>
          {/* Feature Importance Bar Chart */}
          <div className="pt-8">
            <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-2 mb-4">
              <h2 className="font-serif text-[20px] text-[#072427] font-medium">
                Top {data.features.length} most influential factors
              </h2>
              <span className="font-sans text-[12px] text-[#414849]">
                Ranked by share of total learning signal
              </span>
            </div>

            <div className="flex flex-col divide-y divide-[#c1c8c8] border border-[#c1c8c8] bg-[#ffffff]">
              {data.features.map((f, i) => {
                const meta = getFeatureMeta(f.name);
                const barPct = (f.gainShare / maxShare) * 100;
                return (
                  <div key={f.name} className="p-4 hover:bg-[#f7faf6] transition-colors">
                    {/* Top row: rank, label, bar, percentage */}
                    <div className="flex items-center gap-4">
                      <span className="font-mono text-[12px] text-[#414849] w-6 shrink-0">
                        {String(i + 1).padStart(2, "0")}
                      </span>
                      <span className="font-sans text-[14px] font-medium text-[#072427] w-56 shrink-0">
                        {meta.label}
                      </span>
                      <div className="flex-1 h-4 bg-[#f1f4f1] border border-[#c1c8c8] min-w-[80px]">
                        <div
                          className="h-full bg-[#1f3a3d] transition-all"
                          style={{ width: `${barPct}%` }}
                        />
                      </div>
                      <span className="font-mono text-[13px] text-[#39684a] font-medium w-12 text-right shrink-0">
                        {(f.gainShare * 100).toFixed(1)}%
                      </span>
                    </div>
                    {/* Plain-English explanation */}
                    <p className="mt-1.5 ml-10 font-sans text-[12px] text-[#414849] leading-relaxed max-w-[680px]">
                      {meta.plain}
                    </p>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Plain-English Summary */}
          <div className="mt-8 p-5 border border-[#c1c8c8] bg-[#f7faf6]">
            <h3 className="font-sans text-[13px] font-medium text-[#181c1a] mb-2">Key takeaways</h3>
            <ul className="font-sans text-[13px] text-[#414849] leading-relaxed space-y-1.5 list-disc list-inside">
              <li>
                <span className="font-medium text-[#181c1a]">External credit bureau scores</span> are by far the strongest signal — they alone account for the majority of the model's predictive power.
              </li>
              <li>
                <span className="font-medium text-[#181c1a]">Age and employment duration</span> are the next most important — they capture financial stability over time.
              </li>
              <li>
                <span className="font-medium text-[#181c1a]">Loan size and income</span> matter in combination — a large loan relative to income is a strong risk indicator.
              </li>
              <li>
                <span className="font-medium text-[#181c1a]">Engineered ratios</span> (Payment Rate, DTI, Career Stability Index) were added specifically because they capture relationships the raw numbers alone miss.
              </li>
            </ul>
          </div>
        </>
      )}
    </div>
  );
}
