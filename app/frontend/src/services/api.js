
const API_URL = "http://127.0.0.1:5000";

export async function analyzeJob(input) {
  const payload =
    typeof input === "string"
      ? { job_text: input }
      : input;

  let response;

  try {
    response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
  } catch {
    throw new Error(
      "Cannot connect to Flask. Start the backend at http://127.0.0.1:5000."
    );
  }

  let data;

  try {
    data = await response.json();
  } catch {
    throw new Error("The prediction server returned an invalid response.");
  }

  if (!response.ok || data.status === "error") {
    throw new Error(
      data.message || `Analysis failed (HTTP ${response.status}).`
    );
  }

  if (!data.prediction?.final_conclusion) {
    throw new Error(
      "The prediction response is missing the final conclusion."
    );
  }

  return data;
}
