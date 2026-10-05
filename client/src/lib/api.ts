// Shared typed access to the Credit Ledger FastAPI backend.
// The API serves REAL trained-model artifacts: benchmark metrics come from
// models/evaluation_results.json (5-fold OOF protocol, 307,511 train rows).

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface ThresholdMetrics {
  threshold: number;
  f1: number;
  precision: number;
  recall: number;
  confusion_matrix: number[][];
  accuracy: number;
  error_rate: number;
  balanced_accuracy: number;
  mcc: number;
}

export interface ModelMetrics {
  auc_roc: number;
  pr_auc: number;
  ks_stat: number;
  optimal_threshold: number;
  default_threshold: ThresholdMetrics;
  optimal_threshold_metrics: ThresholdMetrics;
  cv_fold_aucs: number[];
  cv_auc_mean: number;
  cv_auc_std: number;
  fit_time_sec?: number;
  brier?: number;
  oof_folds_done?: number;
}

export type Benchmarks = Record<string, ModelMetrics>;

export interface PredictResult {
  model: string;
  probabilityOfDefault: number;
  probabilityPercent: number;
  expectedLoss: number;
  riskTier: "low" | "medium" | "high";
  verdictLabel: string;
  verdictNote: string;
  thresholdUsed: number;
  shapContributions: Record<string, number>;
  shapBaseValue: number;
  shapSource: "treeshap" | "heuristic";
}

export interface ExplainabilityPayload {
  model: string;
  source: string;
  features: { name: string; gainShare: number }[];
  benchmark: { aucRoc: number | null; ksStat: number | null; prAuc: number | null };
}

export class ApiUnavailableError extends Error {}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new ApiUnavailableError(`${path} -> ${res.status}`);
  return (await res.json()) as T;
}

export function fetchBenchmarks(): Promise<Benchmarks> {
  return getJson<Benchmarks>("/api/models");
}

export function fetchExplainability(): Promise<ExplainabilityPayload> {
  return getJson<ExplainabilityPayload>("/api/explainability");
}

export async function predict(req: Record<string, unknown>): Promise<PredictResult> {
  const res = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new ApiUnavailableError(`/api/predict -> ${res.status}`);
  return (await res.json()) as PredictResult;
}

export const MODEL_DISPLAY: Record<
  string,
  { label: string; architecture: string }
> = {
  stacking_ensemble: {
    label: "Stacking Ensemble",
    architecture: "Logistic meta-learner over 7 base models",
  },
  lightgbm: {
    label: "LightGBM",
    architecture: "Histogram GBDT (451 trees, tuned at 100k rows)",
  },
  xgboost: {
    label: "XGBoost",
    architecture: "Regularized hist boosting (659 trees, depth 4)",
  },
  gradient_boosting: {
    label: "Gradient Boosting",
    architecture: "sklearn GBC (159 estimators, depth 2)",
  },
  dnn: {
    label: "Deep Neural Network",
    architecture: "PyTorch GELU MLP (256-128-64, BatchNorm)",
  },
  random_forest: {
    label: "Random Forest",
    architecture: "Bagged trees (154 estimators, depth 16)",
  },
  logistic_regression: {
    label: "Logistic Regression",
    architecture: "L2 GLM (C=0.59, L-BFGS)",
  },
  lda: {
    label: "Linear Discriminant",
    architecture: "LSQR solver, automatic shrinkage",
  },
};

/** Population prior from the training data (24,825 / 307,511). */
export const POPULATION_DEFAULT_RATE = 0.0807;
export const TRAIN_ROWS = 307511;
export const TEST_ROWS = 48744;
