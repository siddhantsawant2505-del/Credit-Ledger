import Link from "next/link";

export default function OverviewPage() {
  return (
    <div className="flex flex-col w-full">
      {/* Dossier Header */}
      <div className="flex flex-col">
        <div className="flex items-center gap-2 mb-2">
          <span className="font-sans text-[11px] leading-[14px] text-[#414849]">
            Dossier Module 01
          </span>
          <span className="text-[#c1c8c8]">/</span>
          <span className="font-mono text-[12px] leading-[16px] text-[#39684a] font-medium">
            Registry Synchronized
          </span>
        </div>

        <h1 className="font-serif text-[36px] leading-[44px] tracking-tight text-[#072427] font-medium">
          Model Validation &amp; Default Scoring
        </h1>

        <p className="mt-4 font-sans text-[16px] leading-[24px] text-[#414849] max-w-[860px]">
          An empirical benchmark suite assessing six candidate machine learning models against 142,000 historical consumer credit agreements. This workbench provides institutional risk committees with standardized discrimination metrics, calibrated probability estimates, and feature-attribution diagnostics.
        </p>
      </div>

      <div className="w-full h-[1px] bg-[#c1c8c8] my-10"></div>

      {/* Key Metric Highlights Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 divide-y md:divide-y-0 md:divide-x divide-[#c1c8c8] border-y border-[#c1c8c8]">
        {/* Metric 1 */}
        <div className="py-6 md:pr-6 flex flex-col justify-between">
          <div className="flex items-baseline justify-between mb-2">
            <span className="font-sans text-[11px] text-[#414849] uppercase tracking-wider font-medium">
              Metric 01
            </span>
            <span className="font-mono text-[12px] text-[#39684a] font-medium">
              Primary Rank
            </span>
          </div>
          <div className="font-mono text-[44px] leading-none text-[#072427] font-medium tracking-tight">
            0.842
          </div>
          <div className="mt-4">
            <div className="font-sans text-[14px] text-[#181c1a] font-medium">
              Champion AUC-ROC (LightGBM v3.1)
            </div>
            <div className="font-sans text-[12px] text-[#414849] mt-1">
              Calibrated against 2024 holdout cohort
            </div>
          </div>
        </div>

        {/* Metric 2 */}
        <div className="py-6 md:px-6 flex flex-col justify-between">
          <div className="flex items-baseline justify-between mb-2">
            <span className="font-sans text-[11px] text-[#414849] uppercase tracking-wider font-medium">
              Metric 02
            </span>
            <span className="font-mono text-[12px] text-[#414849]">
              Holdout Scope
            </span>
          </div>
          <div className="font-mono text-[44px] leading-none text-[#072427] font-medium tracking-tight">
            142,850
          </div>
          <div className="mt-4">
            <div className="font-sans text-[14px] text-[#181c1a] font-medium">
              Total validation accounts
            </div>
            <div className="font-sans text-[12px] text-[#414849] mt-1">
              Observed default rate 4.18%
            </div>
          </div>
        </div>

        {/* Metric 3 */}
        <div className="py-6 md:pl-6 flex flex-col justify-between">
          <div className="flex items-baseline justify-between mb-2">
            <span className="font-sans text-[11px] text-[#414849] uppercase tracking-wider font-medium">
              Metric 03
            </span>
            <span className="font-mono text-[12px] text-[#39684a] font-medium">
              Tolerance Pass
            </span>
          </div>
          <div className="font-mono text-[44px] leading-none text-[#072427] font-medium tracking-tight">
            18.4 bps
          </div>
          <div className="mt-4">
            <div className="font-sans text-[14px] text-[#181c1a] font-medium">
              Brier score calibration loss
            </div>
            <div className="font-sans text-[12px] text-[#414849] mt-1">
              Within Basel committee tolerance
            </div>
          </div>
        </div>
      </div>

      <div className="w-full h-[1px] bg-[#c1c8c8] my-10"></div>

      {/* Dossier Sections Grid */}
      <div className="flex items-baseline justify-between mb-4">
        <div className="font-serif text-[18px] text-[#072427] font-medium">
          Dossier Verification Sections
        </div>
        <div className="font-mono text-[12px] text-[#414849]">
          Index: SEC-01 to SEC-03
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Card 1: Predict & Results */}
        <div className="border border-[#c1c8c8] bg-[#ffffff] p-6 rounded-[2px] flex flex-col justify-between transition-colors hover:bg-[#f1f4f1]">
          <div>
            <div className="flex items-center justify-between border-b border-[#c1c8c8] pb-2 mb-4">
              <span className="font-sans text-[11px] text-[#414849]">Section 01</span>
              <span className="font-mono text-[12px] text-[#414849]">CR-EVAL-A</span>
            </div>
            <h2 className="font-serif text-[18px] text-[#072427] font-medium mb-2">
              Predict &amp; results
            </h2>
            <p className="font-sans text-[14px] text-[#414849] leading-relaxed">
              Evaluate applicant loan parameters across personal, financial, and credit history inputs with immediate risk-verdict classification.
            </p>
          </div>
          <div className="pt-6 mt-6 border-t border-[#c1c8c8]">
            <Link
              href="/predict-results"
              className="inline-flex items-center justify-center px-4 py-2 bg-[#1f3a3d] text-[#ffffff] rounded-[2px] font-sans text-[13px] font-medium hover:bg-[#072427] transition-colors"
            >
              Open scoring engine
            </Link>
          </div>
        </div>

        {/* Card 2: Model Comparison */}
        <div className="border border-[#c1c8c8] bg-[#ffffff] p-6 rounded-[2px] flex flex-col justify-between transition-colors hover:bg-[#f1f4f1]">
          <div>
            <div className="flex items-center justify-between border-b border-[#c1c8c8] pb-2 mb-4">
              <span className="font-sans text-[11px] text-[#414849]">Section 02</span>
              <span className="font-mono text-[12px] text-[#414849]">CR-EVAL-B</span>
            </div>
            <h2 className="font-serif text-[18px] text-[#072427] font-medium mb-2">
              Model comparison
            </h2>
            <p className="font-sans text-[14px] text-[#414849] leading-relaxed">
              Cross-examine AUC-ROC, F1 score, and Kolmogorov-Smirnov discrimination power across 6 candidate models alongside overlaid ROC curves.
            </p>
          </div>
          <div className="pt-6 mt-6 border-t border-[#c1c8c8]">
            <Link
              href="/model-comparison"
              className="inline-flex items-center justify-center px-4 py-2 bg-[#1f3a3d] text-[#ffffff] rounded-[2px] font-sans text-[13px] font-medium hover:bg-[#072427] transition-colors"
            >
              View benchmark tables
            </Link>
          </div>
        </div>

        {/* Card 3: Explainability */}
        <div className="border border-[#c1c8c8] bg-[#ffffff] p-6 rounded-[2px] flex flex-col justify-between transition-colors hover:bg-[#f1f4f1]">
          <div>
            <div className="flex items-center justify-between border-b border-[#c1c8c8] pb-2 mb-4">
              <span className="font-sans text-[11px] text-[#414849]">Section 03</span>
              <span className="font-mono text-[12px] text-[#414849]">CR-EVAL-C</span>
            </div>
            <h2 className="font-serif text-[18px] text-[#072427] font-medium mb-2">
              Explainability
            </h2>
            <p className="font-sans text-[14px] text-[#414849] leading-relaxed">
              Deconstruct decision rationale with localized SHAP feature attribution bars and tabular feature delta quantification.
            </p>
          </div>
          <div className="pt-6 mt-6 border-t border-[#c1c8c8]">
            <Link
              href="/explainability"
              className="inline-flex items-center justify-center px-4 py-2 bg-[#1f3a3d] text-[#ffffff] rounded-[2px] font-sans text-[13px] font-medium hover:bg-[#072427] transition-colors"
            >
              Inspect attribution ledger
            </Link>
          </div>
        </div>
      </div>

      {/* Audit Footer Strip */}
      <div className="w-full border-t border-[#c1c8c8] pt-4 mt-10 flex flex-col md:flex-row items-start md:items-center justify-between gap-2">
        <div className="font-sans text-[12px] text-[#414849]">
          Quarterly Audit Registry #CR-2024-Q3. Signed off by Model Risk Management Committee on November 12, 2024.
        </div>
        <div className="font-mono text-[12px] text-[#181c1a] flex items-center gap-2">
          <span>
            Checksum: <span className="text-[#072427] font-medium">9a4f-88b1-ec02</span>
          </span>
          <span className="text-[#c1c8c8]">|</span>
          <span>Ledger Status: Sealed</span>
        </div>
      </div>
    </div>
  );
}
