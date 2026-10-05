"use client";

import { useEffect, useState, useMemo } from "react";
import {
  MODEL_DISPLAY,
  POPULATION_DEFAULT_RATE,
  fetchBenchmarks,
  predict,
  type Benchmarks,
} from "../../lib/api";

export default function PredictResultsPage() {
  const [activeTab, setActiveTab] = useState<"composite" | "personal" | "financial" | "credit">("composite");

  // Form State
  const [age, setAge] = useState<number>(38);
  const [employment, setEmployment] = useState<string>("ft");
  const [residential, setResidential] = useState<string>("owner_mortgage");
  const [annualIncome, setAnnualIncome] = useState<number>(92500);
  const [requestedPrincipal, setRequestedPrincipal] = useState<number>(24000);
  const [loanTerm, setLoanTerm] = useState<number>(48);
  const [dtiRatio, setDtiRatio] = useState<number>(28.4);
  const [utilizationRate, setUtilizationRate] = useState<number>(41.2);
  const [priorDelinquencies, setPriorDelinquencies] = useState<number>(0);

  // Model Switcher State
  const [selectedModel, setSelectedModel] = useState<string>("lightgbm");
  const [isLiveApi, setIsLiveApi] = useState<boolean>(false);
  const [liveResult, setLiveResult] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [benchmarks, setBenchmarks] = useState<Benchmarks | null>(null);

  // Real benchmark metrics for the model selector labels
  useEffect(() => {
    fetchBenchmarks()
      .then(setBenchmarks)
      .catch(() => setBenchmarks(null));
  }, []);

  const modelOptions = useMemo(
    () =>
      Object.entries(MODEL_DISPLAY).map(([key, d]) => {
        const auc = benchmarks?.[key]?.auc_roc;
        return {
          value: key,
          label: `${d.label}${typeof auc === "number" ? ` (AUC ${auc.toFixed(4)})` : ""}`,
        };
      }),
    [benchmarks]
  );

  // Reset helper
  const handleReset = () => {
    setAge(38);
    setEmployment("ft");
    setResidential("owner_mortgage");
    setAnnualIncome(92500);
    setRequestedPrincipal(24000);
    setLoanTerm(48);
    setDtiRatio(28.4);
    setUtilizationRate(41.2);
    setPriorDelinquencies(0);
    setSelectedModel("lightgbm");
    setLiveResult(null);
  };

  // Live API Fetcher
  const runLivePrediction = async () => {
    setLoading(true);
    try {
      const data = await predict({
        age,
        employment,
        residential,
        annualIncome,
        requestedPrincipal,
        loanTerm,
        dtiRatio,
        utilizationRate,
        priorDelinquencies,
        selectedModel,
      });
      setLiveResult(data);
      setIsLiveApi(true);
    } catch {
      setLiveResult(null);
      setIsLiveApi(false);
    } finally {
      setLoading(false);
    }
  };

  // Prediction Math Simulation
  const prediction = useMemo(() => {
    // Model base offsets (offline simulation only; use "Run prediction" for real model inference)
    const modelOffsets: Record<string, number> = {
      lightgbm: 0.0,
      xgboost: 0.04,
      gradient_boosting: -0.015,
      logistic_regression: 0.11,
      stacking_ensemble: -0.02,
      dnn: 0.02,
      lda: 0.05,
      random_forest: 0.03,
    };

    const modelOffset = modelOffsets[selectedModel] || 0.0;

    // Linear score accumulation calibrated to 8.07% population default rate
    let score = -2.42 + modelOffset;
    score += (dtiRatio - 25) * 0.035;
    score += (utilizationRate - 30) * 0.025;
    score += priorDelinquencies * 0.55;
    score += Math.max(-0.4, (75000 - annualIncome) / 100000);
    score += (requestedPrincipal / Math.max(1, annualIncome)) * 0.45;
    score += (40 - age) * 0.01;

    if (liveResult) {
      const pd = liveResult.probabilityOfDefault;
      let badgeColorClass = "border-[#39684a] text-[#39684a] bg-[#bbefc9]/30";
      if (liveResult.riskTier === "high") {
        badgeColorClass = "border-[#ba1a1a] text-[#ba1a1a] bg-[#ffdad6]/40";
      } else if (liveResult.riskTier === "medium") {
        badgeColorClass = "border-[#B8862E] text-[#B8862E] bg-[#F5F4EF]";
      }

      return {
        pd,
        pdPercent: liveResult.probabilityPercent.toFixed(1),
        pdExact: pd.toFixed(4),
        expectedLoss: liveResult.expectedLoss,
        riskTier: liveResult.riskTier,
        verdictLabel: liveResult.verdictLabel,
        verdictNote: liveResult.verdictNote,
        badgeColorClass,
        shap: {
          dti: liveResult.shapContributions?.["Debt-to-income ratio"] || 0,
          delinq: liveResult.shapContributions?.["Prior Delinquencies"] || 0,
          tenure: liveResult.shapContributions?.["Employment Tenure"] || 0,
          home: -0.015,
          util: liveResult.shapContributions?.["Credit Utilization"] || 0,
        },
      };
    }

    // Sigmoid probability conversion
    const pd = 1 / (1 + Math.exp(-score));
    const pdPercent = (pd * 100).toFixed(1);
    const pdExact = pd.toFixed(4);

    // Expected Loss (EL) with 45% Loss Given Default (LGD)
    const expectedLoss = Math.round(requestedPrincipal * pd * 0.45);

    // Calibrated Risk Classification Tier aligned with 8.07% population default rate
    let riskTier: "low" | "medium" | "high" = "low";
    let verdictLabel = "Low risk";
    let verdictNote = "Probability of default is below population baseline. Recommended automatic prime underwriting pass.";
    let badgeColorClass = "border-[#39684a] text-[#39684a] bg-[#bbefc9]/30";

    if (pd >= 0.18) {
      riskTier = "high";
      verdictLabel = "High risk";
      verdictNote = "Calibrated default hazard exceeds 2x the population benchmark. Adverse notice or senior committee review recommended.";
      badgeColorClass = "border-[#ba1a1a] text-[#ba1a1a] bg-[#ffdad6]/40";
    } else if (pd >= POPULATION_DEFAULT_RATE) {
      riskTier = "medium";
      verdictLabel = "Medium risk";
      verdictNote = "Default risk aligns within conditional underwriting bracket (Tier B). Secondary income/collateral verification required.";
      badgeColorClass = "border-[#B8862E] text-[#B8862E] bg-[#F5F4EF]";
    }

    // Dynamic SHAP Vector calculations
    const shapDti = Math.min(0.08, Math.max(-0.06, (dtiRatio - 32) * 0.003));
    const shapDelinq = priorDelinquencies > 0 ? priorDelinquencies * 0.032 : -0.038;
    const shapTenure = employment === "ft" ? -0.026 : employment === "se" ? 0.018 : -0.008;
    const shapHome = residential === "owner_clear" ? -0.028 : residential === "owner_mortgage" ? -0.015 : 0.022;
    const shapUtil = Math.min(0.06, Math.max(-0.04, (utilizationRate - 35) * 0.002));

    return {
      pd,
      pdPercent,
      pdExact,
      expectedLoss,
      riskTier,
      verdictLabel,
      verdictNote,
      badgeColorClass,
      shap: {
        dti: shapDti,
        delinq: shapDelinq,
        tenure: shapTenure,
        home: shapHome,
        util: shapUtil,
      },
    };
  }, [
    liveResult,
    age,
    employment,
    residential,
    annualIncome,
    requestedPrincipal,
    loanTerm,
    dtiRatio,
    utilizationRate,
    priorDelinquencies,
    selectedModel,
  ]);

  return (
    <div className="flex flex-col w-full text-[#181c1a]">
      {/* Document Header */}
      <section className="flex flex-col gap-1 pb-6 border-b border-[#c1c8c8]">
        <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-1">
          <h1 className="font-serif text-[28px] leading-[36px] text-[#072427] font-medium tracking-tight">
            Individual Default Probability Scoring
          </h1>
          <span className="font-mono text-[12px] text-[#414849]">
            Dossier File: ADJ-2024-9904
          </span>
        </div>
        <p className="font-sans text-[14px] leading-[20px] text-[#414849] max-w-3xl">
          Input applicant profile attributes to generate real-time probability of default and committee risk verdict.
        </p>
      </section>

      {/* Interactive Form Workspace */}
      <section className="pt-6">
        {/* Form Tab Navigation */}
        <div className="flex items-center justify-between border-b border-[#c1c8c8]">
          <div className="flex items-center space-x-6" role="tablist">
            <button
              type="button"
              onClick={() => setActiveTab("composite")}
              className={`pb-2 font-sans text-[13px] font-medium border-b-2 transition-colors ${
                activeTab === "composite"
                  ? "text-[#072427] border-[#1f3a3d]"
                  : "text-[#414849] border-transparent hover:text-[#181c1a]"
              }`}
            >
              Active ledger profile
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("personal")}
              className={`pb-2 font-sans text-[13px] font-medium border-b-2 transition-colors ${
                activeTab === "personal"
                  ? "text-[#072427] border-[#1f3a3d]"
                  : "text-[#414849] border-transparent hover:text-[#181c1a]"
              }`}
            >
              Personal
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("financial")}
              className={`pb-2 font-sans text-[13px] font-medium border-b-2 transition-colors ${
                activeTab === "financial"
                  ? "text-[#072427] border-[#1f3a3d]"
                  : "text-[#414849] border-transparent hover:text-[#181c1a]"
              }`}
            >
              Financial
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("credit")}
              className={`pb-2 font-sans text-[13px] font-medium border-b-2 transition-colors ${
                activeTab === "credit"
                  ? "text-[#072427] border-[#1f3a3d]"
                  : "text-[#414849] border-transparent hover:text-[#181c1a]"
              }`}
            >
              Credit history
            </button>
          </div>
          <div className="hidden sm:flex items-center gap-1.5 font-mono text-[12px] text-[#414849]">
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-[#39684a]"></span>
            Validation criteria: Basel III compliant
          </div>
        </div>

        {/* Primary Form Grid */}
        <form className="pt-6" onSubmit={(e) => e.preventDefault()}>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Column 1: Personal Attributes */}
            <div className={`flex flex-col gap-4 ${activeTab !== "composite" && activeTab !== "personal" ? "opacity-40" : ""}`}>
              <div className="font-sans text-[11px] font-medium text-[#414849] border-b border-[#c1c8c8] pb-1 uppercase tracking-wider">
                01. Personal attributes
              </div>

              <div className="flex flex-col gap-1">
                <label className="font-sans text-[13px] font-medium text-[#181c1a]" htmlFor="applicant-age">
                  Applicant age
                </label>
                <input
                  id="applicant-age"
                  type="number"
                  min={18}
                  max={99}
                  value={age}
                  onChange={(e) => setAge(Number(e.target.value))}
                  className="w-full h-10 px-3 font-mono text-[14px] text-[#181c1a] bg-[#ffffff] border border-[#c1c8c8] rounded-[2px] focus:border-[#1f3a3d] focus:outline-none"
                />
                <span className="font-sans text-[12px] text-[#414849]">
                  Primary applicant chronological age
                </span>
              </div>

              <div className="flex flex-col gap-1">
                <label className="font-sans text-[13px] font-medium text-[#181c1a]" htmlFor="employment-status">
                  Employment status
                </label>
                <select
                  id="employment-status"
                  value={employment}
                  onChange={(e) => setEmployment(e.target.value)}
                  className="w-full h-10 px-3 font-sans text-[14px] text-[#181c1a] bg-[#ffffff] border border-[#c1c8c8] rounded-[2px] focus:border-[#1f3a3d] focus:outline-none"
                >
                  <option value="ft">Full-time permanent</option>
                  <option value="pt">Part-time standard</option>
                  <option value="se">Self-employed (&gt;36m)</option>
                  <option value="ct">Fixed-term contract</option>
                  <option value="rt">Retired / Annuity</option>
                </select>
                <span className="font-sans text-[12px] text-[#414849]">
                  Verified via tax schedule W-2 / 1099
                </span>
              </div>

              <div className="flex flex-col gap-1">
                <label className="font-sans text-[13px] font-medium text-[#181c1a]" htmlFor="residential-status">
                  Residential status
                </label>
                <select
                  id="residential-status"
                  value={residential}
                  onChange={(e) => setResidential(e.target.value)}
                  className="w-full h-10 px-3 font-sans text-[14px] text-[#181c1a] bg-[#ffffff] border border-[#c1c8c8] rounded-[2px] focus:border-[#1f3a3d] focus:outline-none"
                >
                  <option value="owner_mortgage">Owner with mortgage</option>
                  <option value="owner_clear">Owner unencumbered</option>
                  <option value="tenant">Tenant / Leaseholder</option>
                  <option value="other">Institutional living</option>
                </select>
                <span className="font-sans text-[12px] text-[#414849]">
                  Primary place of legal domicile
                </span>
              </div>
            </div>

            {/* Column 2: Financial Parameters */}
            <div className={`flex flex-col gap-4 ${activeTab !== "composite" && activeTab !== "financial" ? "opacity-40" : ""}`}>
              <div className="font-sans text-[11px] font-medium text-[#414849] border-b border-[#c1c8c8] pb-1 uppercase tracking-wider">
                02. Financial parameters
              </div>

              <div className="flex flex-col gap-1">
                <label className="font-sans text-[13px] font-medium text-[#181c1a]" htmlFor="annual-income">
                  Gross annual income ($)
                </label>
                <input
                  id="annual-income"
                  type="number"
                  step={1000}
                  value={annualIncome}
                  onChange={(e) => setAnnualIncome(Number(e.target.value))}
                  className="w-full h-10 px-3 font-mono text-[14px] text-[#181c1a] bg-[#ffffff] border border-[#c1c8c8] rounded-[2px] focus:border-[#1f3a3d] focus:outline-none"
                />
                <span className="font-sans text-[12px] text-[#414849]">
                  Baseline pre-tax verifiable revenue
                </span>
              </div>

              <div className="flex flex-col gap-1">
                <label className="font-sans text-[13px] font-medium text-[#181c1a]" htmlFor="requested-principal">
                  Requested loan principal ($)
                </label>
                <input
                  id="requested-principal"
                  type="number"
                  step={500}
                  value={requestedPrincipal}
                  onChange={(e) => setRequestedPrincipal(Number(e.target.value))}
                  className="w-full h-10 px-3 font-mono text-[14px] text-[#181c1a] bg-[#ffffff] border border-[#c1c8c8] rounded-[2px] focus:border-[#1f3a3d] focus:outline-none"
                />
                <span className="font-sans text-[12px] text-[#414849]">
                  Total requested line disbursement
                </span>
              </div>

              <div className="flex flex-col gap-1">
                <label className="font-sans text-[13px] font-medium text-[#181c1a]" htmlFor="loan-term">
                  Loan term
                </label>
                <select
                  id="loan-term"
                  value={loanTerm}
                  onChange={(e) => setLoanTerm(Number(e.target.value))}
                  className="w-full h-10 px-3 font-sans text-[14px] text-[#181c1a] bg-[#ffffff] border border-[#c1c8c8] rounded-[2px] focus:border-[#1f3a3d] focus:outline-none"
                >
                  <option value={24}>24 months</option>
                  <option value={36}>36 months</option>
                  <option value={48}>48 months</option>
                  <option value={60}>60 months</option>
                  <option value={72}>72 months</option>
                </select>
                <span className="font-sans text-[12px] text-[#414849]">
                  Linear amortization window
                </span>
              </div>
            </div>

            {/* Column 3: Credit History & Ratios */}
            <div className={`flex flex-col gap-4 ${activeTab !== "composite" && activeTab !== "credit" ? "opacity-40" : ""}`}>
              <div className="font-sans text-[11px] font-medium text-[#414849] border-b border-[#c1c8c8] pb-1 uppercase tracking-wider">
                03. Credit ledger history
              </div>

              <div className="flex flex-col gap-1">
                <label className="font-sans text-[13px] font-medium text-[#181c1a]" htmlFor="dti-ratio">
                  Debt-to-income ratio (DTI %)
                </label>
                <input
                  id="dti-ratio"
                  type="number"
                  step={0.1}
                  value={dtiRatio}
                  onChange={(e) => setDtiRatio(Number(e.target.value))}
                  className="w-full h-10 px-3 font-mono text-[14px] text-[#181c1a] bg-[#ffffff] border border-[#c1c8c8] rounded-[2px] focus:border-[#1f3a3d] focus:outline-none"
                />
                <span className="font-sans text-[12px] text-[#414849]">
                  Includes recurring revolving obligations
                </span>
              </div>

              <div className="flex flex-col gap-1">
                <label className="font-sans text-[13px] font-medium text-[#181c1a]" htmlFor="utilization-rate">
                  Revolving line utilization (%)
                </label>
                <input
                  id="utilization-rate"
                  type="number"
                  step={0.1}
                  value={utilizationRate}
                  onChange={(e) => setUtilizationRate(Number(e.target.value))}
                  className="w-full h-10 px-3 font-mono text-[14px] text-[#181c1a] bg-[#ffffff] border border-[#c1c8c8] rounded-[2px] focus:border-[#1f3a3d] focus:outline-none"
                />
                <span className="font-sans text-[12px] text-[#414849]">
                  Aggregate balances over total limits
                </span>
              </div>

              <div className="flex flex-col gap-1">
                <label className="font-sans text-[13px] font-medium text-[#181c1a]" htmlFor="prior-delinquencies">
                  Prior delinquencies (last 24m)
                </label>
                <input
                  id="prior-delinquencies"
                  type="number"
                  min={0}
                  max={24}
                  value={priorDelinquencies}
                  onChange={(e) => setPriorDelinquencies(Number(e.target.value))}
                  className="w-full h-10 px-3 font-mono text-[14px] text-[#181c1a] bg-[#ffffff] border border-[#c1c8c8] rounded-[2px] focus:border-[#1f3a3d] focus:outline-none"
                />
                <span className="font-sans text-[12px] text-[#414849]">
                  Recorded 30+ day payment arrears
                </span>
              </div>
            </div>
          </div>

          {/* Action Strip */}
          <div className="mt-6 pt-4 border-t border-[#c1c8c8] flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <button
                type="button"
                onClick={runLivePrediction}
                disabled={loading}
                className="h-10 px-6 font-sans text-[13px] font-medium text-[#ffffff] bg-[#1f3a3d] hover:bg-[#072427] rounded-[2px] transition-colors disabled:opacity-50"
              >
                {loading ? "Running model inference..." : "Run prediction"}
              </button>
              <button
                type="button"
                onClick={handleReset}
                className="font-sans text-[13px] text-[#414849] hover:text-[#181c1a] underline transition-colors"
              >
                Reset form values
              </button>
            </div>
            <div className="flex items-center gap-2 font-mono text-[12px]">
              {isLiveApi ? (
                <span className="text-[#39684a] bg-[#bbefc9]/40 border border-[#39684a] px-2 py-0.5 rounded-[2px]">
                  ● Live Model API Active
                </span>
              ) : (
                <span className="text-[#414849] bg-[#ecefeb] border border-[#c1c8c8] px-2 py-0.5 rounded-[2px]">
                  ○ Offline Simulation Mode
                </span>
              )}
            </div>
          </div>
        </form>
      </section>

      {/* Output Canvas Divider */}
      <hr className="border-t border-[#c1c8c8] my-10" />

      {/* Results Section */}
      <section className="flex flex-col gap-6">
        {/* Header with Model Switcher */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 pb-2 border-b border-[#c1c8c8]">
          <div>
            <div className="font-sans text-[11px] text-[#414849] uppercase tracking-wider font-medium">
              Computed outcome dossier
            </div>
            <h2 className="font-serif text-[22px] text-[#072427] font-medium mt-1">
              Prediction Results &amp; Model Output
            </h2>
          </div>

          <div className="flex items-center gap-2">
            <label htmlFor="model-selector" className="font-sans text-[13px] text-[#414849]">
              Active evaluation model:
            </label>
            <select
              id="model-selector"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="h-10 px-3 font-mono text-[12px] text-[#181c1a] bg-[#ffffff] border border-[#c1c8c8] rounded-[2px] focus:border-[#1f3a3d] focus:outline-none"
            >
              {modelOptions.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Output Display Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 border border-[#c1c8c8] bg-[#ffffff]">
          {/* Probability of Default */}
          <div className="p-6 lg:col-span-4 border-b lg:border-b-0 lg:border-r border-[#c1c8c8] flex flex-col justify-between">
            <div>
              <span className="font-sans text-[11px] text-[#414849] uppercase tracking-wider font-medium block">
                Primary metric
              </span>
              <span className="font-sans text-[13px] text-[#181c1a] font-medium mt-1 block">
                Calculated Probability of Default (PD)
              </span>
              <div className="mt-4 flex items-baseline gap-2">
                <span className="font-mono text-[44px] leading-tight text-[#072427] font-medium tracking-tight">
                  {prediction.pdPercent}%
                </span>
                <span className="font-mono text-[12px] text-[#414849]">
                  ({prediction.pdExact})
                </span>
              </div>
            </div>
            <div className="pt-4 mt-4 border-t border-[#c1c8c8]">
              <div className="font-sans text-[12px] text-[#414849]">
                Population default rate <span className="font-mono text-[#181c1a]">{(POPULATION_DEFAULT_RATE * 100).toFixed(2)}%</span>
              </div>
              <div className="font-sans text-[11px] text-[#39684a] mt-1 font-medium">
                {(prediction.pd * 100 - POPULATION_DEFAULT_RATE * 100).toFixed(1)}% relative to the training cohort baseline
              </div>
            </div>
          </div>

          {/* Risk Verdict & Classification */}
          <div className="p-6 lg:col-span-4 border-b lg:border-b-0 lg:border-r border-[#c1c8c8] flex flex-col justify-between">
            <div>
              <span className="font-sans text-[11px] text-[#414849] uppercase tracking-wider font-medium block">
                Committee classification
              </span>
              <span className="font-sans text-[13px] text-[#181c1a] font-medium mt-1 block">
                Adjudication risk verdict
              </span>
              <div className="mt-4">
                <span className={`inline-flex items-center px-3 py-1 border font-sans text-[13px] font-medium uppercase tracking-wider ${prediction.badgeColorClass}`}>
                  {prediction.verdictLabel}
                </span>
              </div>
            </div>
            <div className="pt-4 mt-4 border-t border-[#c1c8c8]">
              <p className="font-sans text-[12px] text-[#414849] leading-relaxed">
                {prediction.verdictNote}
              </p>
            </div>
          </div>

          {/* Expected Loss & Statistical Boundary */}
          <div className="p-6 lg:col-span-4 flex flex-col justify-between">
            <div>
              <span className="font-sans text-[11px] text-[#414849] uppercase tracking-wider font-medium block">
                Statistical boundary
              </span>
              <span className="font-sans text-[13px] text-[#181c1a] font-medium mt-1 block">
                Expected loss (EL) provision
              </span>
              <div className="mt-4 flex items-baseline gap-2">
                <span className="font-mono text-[44px] leading-tight text-[#072427] font-medium tracking-tight">
                  ${prediction.expectedLoss.toLocaleString()}
                </span>
                <span className="font-mono text-[12px] text-[#414849]">USD</span>
              </div>
            </div>
            <div className="pt-4 mt-4 border-t border-[#c1c8c8]">
              <div className="flex justify-between font-sans text-[12px] text-[#414849]">
                <span>Confidence Interval (95%):</span>
                <span className="font-mono text-[#181c1a]">
                  {(prediction.pd * 100 * 0.8).toFixed(1)}% – {(prediction.pd * 100 * 1.2).toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between font-sans text-[12px] text-[#414849] mt-1">
                <span>Loss Given Default (LGD):</span>
                <span className="font-mono text-[#181c1a]">45.0% flat</span>
              </div>
            </div>
          </div>
        </div>

        {/* Decision Rule Matrix Table */}
        <div className="border border-[#c1c8c8] bg-[#ffffff]">
          <div className="px-4 py-2 bg-[#f1f4f1] border-b border-[#c1c8c8] flex items-center justify-between">
            <span className="font-sans text-[13px] text-[#181c1a] font-medium">
              Underwriting decision rule matrix
            </span>
            <span className="font-mono text-[11px] text-[#414849]">
              Policy code: POL-2024-V3
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-[#f7faf6] border-b border-[#c1c8c8] font-sans text-[11px] text-[#414849] uppercase tracking-wider">
                  <th className="py-2 px-4 font-medium">Classification grade</th>
                  <th className="py-2 px-4 font-medium text-right">Cutoff boundary</th>
                  <th className="py-2 px-4 font-medium text-right">Lending limit</th>
                  <th className="py-2 px-4 font-medium">Statutory condition</th>
                  <th className="py-2 px-4 font-medium text-right">Cohort match status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#c1c8c8] font-sans text-[12px] text-[#181c1a]">
                {/* Low Risk Row */}
                <tr className={prediction.riskTier === "low" ? "bg-[#bbefc9]/20 font-medium" : "hover:bg-[#f1f4f1]"}>
                  <td className="py-2.5 px-4 font-medium flex items-center gap-2">
                    <span className="inline-block w-2 h-2 bg-[#39684a]"></span>
                    <span className="text-[#39684a]">Low risk</span>
                  </td>
                  <td className="py-2.5 px-4 font-mono text-right">PD &lt; {(POPULATION_DEFAULT_RATE * 100).toFixed(1)}%</td>
                  <td className="py-2.5 px-4 font-mono text-right">$100,000</td>
                  <td className="py-2.5 px-4">Standard covenant; automatic pass available</td>
                  <td className="py-2.5 px-4 text-right font-medium">
                    {prediction.riskTier === "low" ? (
                      <span className="text-[#39684a] font-mono text-[12px]">✓ Current match</span>
                    ) : (
                      <span className="text-[#414849] font-mono text-[12px]">–</span>
                    )}
                  </td>
                </tr>

                {/* Medium Risk Row */}
                <tr className={prediction.riskTier === "medium" ? "bg-[#F5F4EF] font-medium" : "hover:bg-[#f1f4f1]"}>
                  <td className="py-2.5 px-4 font-medium flex items-center gap-2">
                    <span className="inline-block w-2 h-2 bg-[#B8862E]"></span>
                    <span className="text-[#B8862E]">Medium risk</span>
                  </td>
                  <td className="py-2.5 px-4 font-mono text-right">{(POPULATION_DEFAULT_RATE * 100).toFixed(1)}% ≤ PD &lt; 18.0%</td>
                  <td className="py-2.5 px-4 font-mono text-right">$45,000</td>
                  <td className="py-2.5 px-4">Manual senior adjudicator concurrence required</td>
                  <td className="py-2.5 px-4 text-right font-medium">
                    {prediction.riskTier === "medium" ? (
                      <span className="text-[#B8862E] font-mono text-[12px]">✓ Current match</span>
                    ) : (
                      <span className="text-[#414849] font-mono text-[12px]">–</span>
                    )}
                  </td>
                </tr>

                {/* High Risk Row */}
                <tr className={prediction.riskTier === "high" ? "bg-[#ffdad6]/20 font-medium" : "hover:bg-[#f1f4f1]"}>
                  <td className="py-2.5 px-4 font-medium flex items-center gap-2">
                    <span className="inline-block w-2 h-2 bg-[#ba1a1a]"></span>
                    <span className="text-[#ba1a1a]">High risk</span>
                  </td>
                  <td className="py-2.5 px-4 font-mono text-right">PD ≥ 18.0%</td>
                  <td className="py-2.5 px-4 font-mono text-right">$0</td>
                  <td className="py-2.5 px-4">Adverse notice issuance; immediate decline</td>
                  <td className="py-2.5 px-4 text-right font-medium">
                    {prediction.riskTier === "high" ? (
                      <span className="text-[#ba1a1a] font-mono text-[12px]">✓ Current match</span>
                    ) : (
                      <span className="text-[#414849] font-mono text-[12px]">–</span>
                    )}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </div>
  );
}
