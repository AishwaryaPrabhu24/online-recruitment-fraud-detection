
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { analyzeJob } from "../services/api";
import "../styles/AnalyzeJob.css";

const initialForm = {
  title: "",
  company_profile: "",
  description: "",
  requirements: "",
  benefits: "",
  salary_range: "",
  employment_type: "",
  required_experience: "",
  required_education: "",
  industry: "",
  function: "",
  department: "",
  telecommuting: 0,
  has_company_logo: 0,
  has_questions: 0,
};

function AnalyzeJob() {
  const navigate = useNavigate();

  const [mode, setMode] = useState("form");
  const [form, setForm] = useState(initialForm);
  const [jobText, setJobText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function handleModeChange(nextMode) {
    setMode(nextMode);
    setError("");
  }

  function handleChange(event) {
    const { name, value, type, checked } = event.target;

    setForm((previous) => ({
      ...previous,
      [name]:
        type === "checkbox"
          ? Number(checked)
          : value,
    }));

    if (error) setError("");
  }

  async function handleSubmit(event) {
    event.preventDefault();

    if (loading) return;

    let payload;

    if (mode === "form") {
      if (!form.title.trim() && !form.description.trim()) {
        setError("Please enter at least a job title or description.");
        return;
      }

      payload = {
        ...form,
        title: form.title.trim(),
        company_profile: form.company_profile.trim(),
        description: form.description.trim(),
        requirements: form.requirements.trim(),
        benefits: form.benefits.trim(),
        salary_range: form.salary_range.trim(),
        employment_type: form.employment_type,
        required_experience: form.required_experience,
        required_education: form.required_education,
        industry: form.industry.trim(),
        function: form.function.trim(),
        department: form.department.trim(),
      };
    } else {
      const message = jobText.trim();

      if (!message) {
        setError("Please paste the complete job message.");
        return;
      }

      if (message.length > 50000) {
        setError("The message cannot exceed 50,000 characters.");
        return;
      }

      payload = { job_text: message };
    }

    setLoading(true);
    setError("");

    try {
      const result = await analyzeJob(payload);

      const submittedJob =
        mode === "form"
          ? form
          : {
              ...initialForm,
              title:
                jobText
                  .split(/\r?\n/)
                  .find((line) => line.trim())
                  ?.trim()
                  .slice(0, 250) || "Full job message",
              description: jobText.trim(),
              job_text: jobText.trim(),
              input_mode: "full_message",
            };

      sessionStorage.setItem(
        "jobForm",
        JSON.stringify(submittedJob)
      );

      sessionStorage.setItem(
        "predictionResult",
        JSON.stringify(result)
      );

      navigate("/results");
    } catch (err) {
      setError(
        err.message || "Unable to analyze this job posting."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="analyze-page">
      <header className="analyze-header">
        <button
          type="button"
          className="back-button"
          onClick={() => navigate("/")}
          disabled={loading}
        >
          <span aria-hidden="true">←</span> Home
        </button>

        <div className="analyze-brand">
          <span className="brand-icon">F</span>
          <span>FraudShield</span>
        </div>

        <span className="step-indicator">STEP 01 / 02</span>
      </header>

      <section className="form-intro">
        <div className="form-eyebrow">
          <span className="form-eyebrow-dot" />
          JOB POSTING ANALYSIS
        </div>

        <h1>
          Check before you <span>trust.</span>
        </h1>

        <p>
          Analyze a recruitment opportunity using individual job
          details or by pasting the complete job message.
        </p>
      </section>

      <section className="job-form">
        <div
          className="analysis-mode-selector"
          role="group"
          aria-label="Choose job analysis input method"
        >
          <button
            type="button"
            className={`analysis-mode-button ${
              mode === "form" ? "active" : ""
            }`}
            onClick={() => handleModeChange("form")}
            aria-pressed={mode === "form"}
            disabled={loading}
          >
            <span aria-hidden="true">☷</span>
            Form-Based Analysis
          </button>

          <button
            type="button"
            className={`analysis-mode-button ${
              mode === "message" ? "active" : ""
            }`}
            onClick={() => handleModeChange("message")}
            aria-pressed={mode === "message"}
            disabled={loading}
          >
            <span aria-hidden="true">▤</span>
            Entire Job Message
          </button>
        </div>

        {error && (
          <div className="form-error" role="alert">
            <strong>Analysis failed</strong>
            <p>{error}</p>
          </div>
        )}

        <form onSubmit={handleSubmit}>
          {mode === "form" ? (
            <>
              <div className="section-heading">
                <span className="section-number">01</span>
                <div>
                  <h2>Enter job details</h2>
                  <p>
                    Fill in the available information. Optional fields
                    can be left blank.
                  </p>
                </div>
              </div>

              <div className="form-grid">
                <label className="field">
                  <span>Job Title <span className="required">*</span></span>
                  <input
                    name="title"
                    value={form.title}
                    onChange={handleChange}
                    placeholder="e.g. Software Developer"
                    disabled={loading}
                  />
                </label>

                <label className="field">
                  <span>Salary Range</span>
                  <input
                    name="salary_range"
                    value={form.salary_range}
                    onChange={handleChange}
                    placeholder="e.g. ₹3–5 LPA"
                    disabled={loading}
                  />
                </label>

                <label className="field full-width">
                  <span>Company Profile</span>
                  <textarea
                    name="company_profile"
                    value={form.company_profile}
                    onChange={handleChange}
                    placeholder="Describe the company..."
                    rows={3}
                    disabled={loading}
                  />
                </label>

                <label className="field full-width">
                  <span>Job Description <span className="required">*</span></span>
                  <textarea
                    name="description"
                    value={form.description}
                    onChange={handleChange}
                    placeholder="Enter the job responsibilities and description..."
                    rows={5}
                    disabled={loading}
                  />
                </label>

                <label className="field full-width">
                  <span>Requirements</span>
                  <textarea
                    name="requirements"
                    value={form.requirements}
                    onChange={handleChange}
                    placeholder="Skills, qualifications and experience..."
                    rows={3}
                    disabled={loading}
                  />
                </label>

                <label className="field full-width">
                  <span>Benefits</span>
                  <textarea
                    name="benefits"
                    value={form.benefits}
                    onChange={handleChange}
                    placeholder="Benefits offered by the employer..."
                    rows={2}
                    disabled={loading}
                  />
                </label>

                <label className="field">
                  <span>Employment Type</span>
                  <select
                    name="employment_type"
                    value={form.employment_type}
                    onChange={handleChange}
                    disabled={loading}
                  >
                    <option value="">Select employment type</option>
                    <option value="Full-time">Full-time</option>
                    <option value="Part-time">Part-time</option>
                    <option value="Contract">Contract</option>
                    <option value="Temporary">Temporary</option>
                    <option value="Other">Other</option>
                  </select>
                </label>

                <label className="field">
                  <span>Required Experience</span>
                  <select
                    name="required_experience"
                    value={form.required_experience}
                    onChange={handleChange}
                    disabled={loading}
                  >
                    <option value="">Select experience</option>
                    <option value="Internship">Internship</option>
                    <option value="Entry level">Entry level</option>
                    <option value="Associate">Associate</option>
                    <option value="Mid-Senior level">Mid-Senior level</option>
                    <option value="Director">Director</option>
                    <option value="Executive">Executive</option>
                  </select>
                </label>

                <label className="field">
                  <span>Required Education</span>
                  <select
                    name="required_education"
                    value={form.required_education}
                    onChange={handleChange}
                    disabled={loading}
                  >
                    <option value="">Select education</option>
                    <option value="High School or equivalent">
                      High School
                    </option>
                    <option value="Associate Degree">
                      Associate Degree
                    </option>
                    <option value="Bachelor's Degree">
                      Bachelor's Degree
                    </option>
                    <option value="Master's Degree">
                      Master's Degree
                    </option>
                    <option value="Doctorate">Doctorate</option>
                    <option value="Unspecified">Unspecified</option>
                  </select>
                </label>

                <label className="field">
                  <span>Industry</span>
                  <input
                    name="industry"
                    value={form.industry}
                    onChange={handleChange}
                    placeholder="e.g. Information Technology"
                    disabled={loading}
                  />
                </label>

                <label className="field">
                  <span>Job Function</span>
                  <input
                    name="function"
                    value={form.function}
                    onChange={handleChange}
                    placeholder="e.g. Engineering"
                    disabled={loading}
                  />
                </label>

                <label className="field">
                  <span>Department</span>
                  <input
                    name="department"
                    value={form.department}
                    onChange={handleChange}
                    placeholder="e.g. Development"
                    disabled={loading}
                  />
                </label>
              </div>

              <div className="checkbox-group">
                <p className="checkbox-group-title">
                  Additional posting information
                </p>

                <label className="checkbox-field">
                  <input
                    type="checkbox"
                    name="telecommuting"
                    checked={Boolean(form.telecommuting)}
                    onChange={handleChange}
                    disabled={loading}
                  />
                  Remote / telecommuting position
                </label>

                <label className="checkbox-field">
                  <input
                    type="checkbox"
                    name="has_company_logo"
                    checked={Boolean(form.has_company_logo)}
                    onChange={handleChange}
                    disabled={loading}
                  />
                  Company logo is present
                </label>

                <label className="checkbox-field">
                  <input
                    type="checkbox"
                    name="has_questions"
                    checked={Boolean(form.has_questions)}
                    onChange={handleChange}
                    disabled={loading}
                  />
                  Screening questions are included
                </label>
              </div>
            </>
          ) : (
            <>
              <div className="section-heading">
                <span className="section-number">01</span>
                <div>
                  <h2>Paste the entire job message</h2>
                  <p>
                    Include the company name, salary, requirements,
                    contact details and application instructions
                    whenever available.
                  </p>
                </div>
              </div>

              <div className="form-grid">
                <label className="field full-width">
                  <span>
                    Complete Recruitment Message{" "}
                    <span className="required">*</span>
                  </span>

                  <textarea
                    name="job_text"
                    value={jobText}
                    onChange={(event) => {
                      setJobText(event.target.value);
                      if (error) setError("");
                    }}
                    placeholder={`Paste the complete recruitment message here...

Example:
Greetings from ABC Technologies.

We are hiring fresh graduates for Software Developer roles.

Qualification: B.E./B.Tech
Experience: Freshers
Salary: As per company policy
Responsibilities: Software development and testing
How to apply: Send your resume to the official company email.

Paste your actual job message here.`}
                    rows={16}
                    maxLength={50000}
                    disabled={loading}
                  />

                  <small>
                    {jobText.length.toLocaleString()} / 50,000 characters
                  </small>
                </label>
              </div>
            </>
          )}

          <div className="form-disclaimer">
            <span aria-hidden="true">ⓘ</span>
            <p>
              FraudShield estimates recruitment fraud risk using a
              trained machine-learning model. Its prediction does not
              independently verify an employer or prove that an offer
              is genuine or fraudulent. Do not include passwords,
              bank details or other sensitive personal information.
            </p>
          </div>

          <div className="form-actions">
            <button
              type="button"
              className="secondary-button"
              onClick={() => navigate("/")}
              disabled={loading}
            >
              Cancel
            </button>

            <button
              type="submit"
              className="submit-button"
              disabled={
                loading ||
                (mode === "form" &&
                  !form.title.trim() &&
                  !form.description.trim()) ||
                (mode === "message" && !jobText.trim())
              }
            >
              {loading
                ? "Analyzing job..."
                : "Analyze Job Posting"}
              {!loading && <span aria-hidden="true"> →</span>}
            </button>
          </div>
        </form>
      </section>
    </main>
  );
}

export default AnalyzeJob;
