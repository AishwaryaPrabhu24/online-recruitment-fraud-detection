
import { useNavigate } from "react-router-dom";
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip,
} from "recharts";
import "../styles/Results.css";

function Results() {
  const navigate = useNavigate();

  let result = null;
  let submittedJob = {};

  try {
    result = JSON.parse(sessionStorage.getItem("predictionResult") || "null");
    submittedJob = JSON.parse(sessionStorage.getItem("jobForm") || "{}");
  } catch {
    result = null;
    submittedJob = {};
  }

  const prediction = result?.prediction;
  const conclusion = prediction?.final_conclusion;
  const model = prediction?.model;

  const label = conclusion?.label;
  const fraudProbability = Number(conclusion?.probability_fraud);
  const legitimateProbability = Number(conclusion?.probability_legitimate);

  const hasProbabilities =
    Number.isFinite(fraudProbability) &&
    Number.isFinite(legitimateProbability);

  const chartData = hasProbabilities
    ? [
        { name: "Fraud risk", value: fraudProbability * 100 },
        { name: "Legitimate", value: legitimateProbability * 100 },
      ]
    : [];

  const redFlags = prediction?.red_flags || [];
  const summary = prediction?.red_flag_summary || {};

  const topWords =
    model?.top_words || prediction?.top_words || [];

  const isFraudulent = label?.toLowerCase() === "fraudulent";
  const isLegitimate = label?.toLowerCase() === "legitimate";

  const isUncertain =
    hasProbabilities &&
    Math.max(fraudProbability, legitimateProbability) < 0.65;

  function formatPercent(value) {
    return `${(value * 100).toFixed(1)}%`;
  }

  function readableFeature(feature) {
    return feature
      .replaceAll("_", " ")
      .replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function riskLevel(level) {
    return ["high", "medium", "low"].includes(
      String(level).toLowerCase()
    )
      ? String(level).toLowerCase()
      : "unknown";
  }

  return (
    <main className="results-page">
      <header className="results-header">
        <button
          className="results-brand"
          onClick={() => navigate("/")}
          type="button"
        >
          <span className="results-brand-icon">F</span>
          <span>FraudShield</span>
        </button>

        <button
          className="results-home-link"
          type="button"
          onClick={() => navigate("/")}
        >
          Home
        </button>
      </header>

      <section className="results-main">
        <div className="results-eyebrow">
          <span className="results-eyebrow-dot"></span>
          JOB ANALYSIS REPORT
        </div>

        <h1>
          Analysis <span>results.</span>
        </h1>

        <p className="results-subtitle">
          Review the model prediction, estimated probabilities, and
          potential warning indicators for this job posting.
        </p>

        {!result || !conclusion ? (
          <section className="no-result-card">
            <div className="no-result-icon">!</div>
            <h2>No prediction available yet</h2>
            <p>
              Your job details have been collected, but the machine learning
              prediction has not been received. Connect the form to the
              Flask prediction API to generate the actual results.
            </p>
            {submittedJob.title && (
              <p className="submitted-job">
                Submitted job: <strong>{submittedJob.title}</strong>
              </p>
            )}
            <button
              className="results-primary-button"
              onClick={() => navigate("/analyze")}
              type="button"
            >
              Return to job form <span>→</span>
            </button>
          </section>
        ) : (
          <>
            <section
              className={`conclusion-card ${
                isFraudulent
                  ? "conclusion-fraud"
                  : isLegitimate
                    ? "conclusion-legitimate"
                    : ""
              }`}
            >
              <div className="conclusion-top">
                <div>
                  <span className="card-overline">MODEL CONCLUSION</span>
                  <h2>{label || "Prediction returned"}</h2>
                </div>

                <div className="conclusion-symbol" aria-hidden="true">
                  {isFraudulent ? "!" : isLegitimate ? "✓" : "?"}
                </div>
              </div>

              <p className="conclusion-message">
                {conclusion.message ||
                  "The model has returned a classification for this job posting."}
              </p>

              {isUncertain && (
                <div className="uncertainty-note">
                  The model's probabilities are relatively close. Treat this
                  result as uncertain and verify the employer independently.
                </div>
              )}

              <div className="conclusion-meta">
                <span>Model used</span>
                <strong>{model?.name || result.model || "XGBoost"}</strong>
              </div>
            </section>

            <section className="result-section">
              <div className="result-section-heading">
                <div>
                  <span className="card-overline">PROBABILITY BREAKDOWN</span>
                  <h2>What the model estimates</h2>
                </div>
              </div>

              <div className="probability-layout">
                <div className="probability-chart-card">
                  {hasProbabilities ? (
                    <>
                      <div className="probability-chart">
                        <ResponsiveContainer width="100%" height="100%">
                          <PieChart>
                            <Pie
                              data={chartData}
                              dataKey="value"
                              nameKey="name"
                              innerRadius="65%"
                              outerRadius="88%"
                              paddingAngle={4}
                              stroke="none"
                            >
                              <Cell fill="#f15b70" />
                              <Cell fill="#50c5a1" />
                            </Pie>
                            <Tooltip
                              formatter={(value) => [
                                `${Number(value).toFixed(1)}%`,
                                "Estimated probability",
                              ]}
                              contentStyle={{
                                background: "#151b28",
                                border: "1px solid #303849",
                                borderRadius: "8px",
                                color: "#f4f6fb",
                              }}
                            />
                          </PieChart>
                        </ResponsiveContainer>
                      </div>

                      <div className="chart-center-label">
                        <strong>{formatPercent(fraudProbability)}</strong>
                        <span>Fraud probability</span>
                      </div>
                    </>
                  ) : (
                    <div className="chart-unavailable">
                      Probability values are unavailable.
                    </div>
                  )}
                </div>

                <div className="probability-stats">
                  <div className="probability-stat">
                    <span className="stat-label">
                      <span className="legend-dot fraud-dot"></span>
                      Fraud probability
                    </span>
                    <strong className="fraud-value">
                      {hasProbabilities
                        ? formatPercent(fraudProbability)
                        : "N/A"}
                    </strong>
                    <div className="stat-track">
                      <div
                        className="stat-fill fraud-fill"
                        style={{
                          width: hasProbabilities
                            ? `${Math.min(100, Math.max(0, fraudProbability * 100))}%`
                            : "0%",
                        }}
                      />
                    </div>
                  </div>

                  <div className="probability-stat">
                    <span className="stat-label">
                      <span className="legend-dot legitimate-dot"></span>
                      Legitimate probability
                    </span>
                    <strong className="legitimate-value">
                      {hasProbabilities
                        ? formatPercent(legitimateProbability)
                        : "N/A"}
                    </strong>
                    <div className="stat-track">
                      <div
                        className="stat-fill legitimate-fill"
                        style={{
                          width: hasProbabilities
                            ? `${Math.min(100, Math.max(0, legitimateProbability * 100))}%`
                            : "0%",
                        }}
                      />
                    </div>
                  </div>

                  <p className="probability-caption">
                    These percentages represent the model's estimates, not
                    verified facts about the employer or job.
                  </p>
                </div>
              </div>
            </section>

            <section className="result-section">
              <div className="result-section-heading">
                <div>
                  <span className="card-overline">RISK INDICATORS</span>
                  <h2>Potential warning signs</h2>
                  <p>
                    Indicators detected by the system, based on the submitted
                    posting and its available fields.
                  </p>
                </div>
              </div>

              <div className="risk-summary-grid">
                <div className="risk-summary-item">
                  <span>Detected</span>
                  <strong>{summary.total_detected ?? redFlags.length}</strong>
                </div>
                <div className="risk-summary-item risk-summary-high">
                  <span>High</span>
                  <strong>{summary.high ?? 0}</strong>
                </div>
                <div className="risk-summary-item risk-summary-medium">
                  <span>Medium</span>
                  <strong>{summary.medium ?? 0}</strong>
                </div>
                <div className="risk-summary-item risk-summary-low">
                  <span>Low</span>
                  <strong>{summary.low ?? 0}</strong>
                </div>
              </div>

              {redFlags.length > 0 ? (
                <div className="risk-list">
                  {redFlags.map((flag, index) => {
                    const level = riskLevel(flag.level);

                    return (
                      <article
                        className="risk-item"
                        key={`${flag.feature || "flag"}-${index}`}
                      >
                        <span className={`risk-indicator risk-${level}`}>
                          {level === "unknown" ? "?" : "!"}
                        </span>

                        <div className="risk-item-content">
                          <h3>{readableFeature(flag.feature || "Risk indicator")}</h3>
                          <p>
                            {flag.description ||
                              "This indicator was detected by the system. Review the posting and verify the relevant information independently."}
                          </p>
                        </div>

                        <span className={`risk-level risk-level-${level}`}>
                          {level}
                        </span>
                      </article>
                    );
                  })}
                </div>
              ) : (
                <div className="empty-risk-list">
                  No risk indicators were returned by the API. This does not
                  guarantee that the job is safe.
                </div>
              )}
            </section>

            <section className="result-section">
              <div className="result-section-heading">
                <div>
                  <span className="card-overline">MODEL-ASSOCIATED TERMS</span>
                  <h2>Important words and phrases</h2>
                  <p>
                    Terms associated with the model output. These are not
                    necessarily causal explanations for the prediction.
                  </p>
                </div>
              </div>

              {topWords.length > 0 ? (
                <div className="terms-list">
                  {topWords.slice(0, 10).map((item, index) => (
                    <div
                      className="term-row"
                      key={`${item.word || "term"}-${index}`}
                    >
                      <span className="term-rank">
                        {String(index + 1).padStart(2, "0")}
                      </span>

                      <span className="term-name">
                        {item.word || "Unknown term"}
                      </span>

                      <span
                        className={`term-direction ${
                          item.direction === "fraudulent" ||
                          item.direction === "fraud"
                            ? "term-fraud"
                            : "term-legitimate"
                        }`}
                      >
                        {item.direction || "Associated"}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="empty-risk-list">
                  No important terms were returned by the API.
                </div>
              )}
            </section>

            <div className="results-disclaimer">
              <strong>Important:</strong> This system is a decision-support
              tool, not a guarantee. A job may be fraudulent even when
              classified as legitimate, and legitimate employers can have
              incomplete postings. Verify the company through its official
              website, never pay recruitment fees, and do not share sensitive
              personal or financial information with unverified recruiters.
            </div>

            <div className="results-actions">
              <button
                className="results-secondary-button"
                type="button"
                onClick={() => navigate("/")}
              >
                Back to Home
              </button>

              <button
                className="results-primary-button"
                type="button"
                onClick={() => navigate("/analyze")}
              >
                Analyze another job <span>→</span>
              </button>
            </div>
          </>
        )}
      </section>

      <footer className="results-footer">
        FraudShield · Machine Learning-Based Online Recruitment Fraud Detection
      </footer>
    </main>
  );
}

export default Results;
