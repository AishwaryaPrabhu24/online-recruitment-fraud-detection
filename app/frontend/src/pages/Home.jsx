
import { useNavigate } from "react-router-dom";
import "../styles/Home.css";

function Home() {
  const navigate = useNavigate();

  return (
    <main className="home-page">
      <nav className="home-navbar">
        <a className="brand" href="/" aria-label="FraudShield home">
          <span className="brand-icon">F</span>
          <span>FraudShield</span>
        </a>

        <span className="nav-status">
          <span className="status-dot"></span>
          AI-Powered Job Safety
        </span>
      </nav>

      <section className="home-hero">
        <div className="hero-content">
          <div className="hero-badge">
            <span className="hero-badge-dot"></span>
            RECRUITMENT FRAUD AWARENESS
          </div>

          <h1>
            Not Every Job
            <br />
            Offer Is <span className="highlight-text">Real.</span>
          </h1>

          <p className="hero-description">
            Fake job postings can cost you your money, time, and personal
            information. Analyze a job description with machine learning
            before you decide to trust the opportunity.
          </p>

          <button
            className="analyze-button"
            onClick={() => navigate("/analyze")}
          >
            Analyze the Job
            <span aria-hidden="true">→</span>
          </button>

          <p className="hero-note">
            <span aria-hidden="true">✦</span>
            AI-assisted analysis · Risk indicators · Model prediction
          </p>
        </div>

        <div className="hero-visual" aria-hidden="true">
          <div className="visual-glow"></div>

          <div className="scanner-ring ring-one"></div>
          <div className="scanner-ring ring-two"></div>

          <div className="shield-card">
            <div className="shield-topline">
              <span>THREAT MONITOR</span>
              <span className="monitor-dot"></span>
            </div>

            <div className="shield-symbol">
              <svg
                viewBox="0 0 100 112"
                role="presentation"
                focusable="false"
              >
                <path
                  d="M50 5 L88 20 V49 C88 75 72 94 50 106 C28 94 12 75 12 49 V20 Z"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="3"
                />
                <path
                  d="M34 53 L45 64 L67 40"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="4"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </div>

            <div className="shield-divider"></div>

            <div className="monitor-row">
              <span className="monitor-line"></span>
              <span className="monitor-line short"></span>
              <span className="monitor-square"></span>
            </div>

            <div className="monitor-row">
              <span className="monitor-line medium"></span>
              <span className="monitor-square danger-square"></span>
            </div>

            <div className="monitor-footer">
              <span>JOB SECURITY SCAN</span>
              <span>ACTIVE</span>
            </div>
          </div>

          <div className="floating-tag tag-top">
            <span className="tag-alert">!</span>
            Verify before you trust
          </div>

          <div className="floating-tag tag-bottom">
            <span className="tag-check">✓</span>
            Analyze suspicious offers
          </div>
        </div>
      </section>

      <footer className="home-footer">
        <span>FraudShield · Recruitment Fraud Detection</span>
        <span>Machine Learning Project</span>
      </footer>
    </main>
  );
}

export default Home;
