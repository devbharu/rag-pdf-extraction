// Change this line - update the API endpoint
const API = "http://127.0.0.1:8000";

export async function uploadPDF(file) {
  const fd = new FormData();
  fd.append("file", file);
  await fetch(`${API}/ingest`, { method: "POST", body: fd });
}

export async function askQuestion(q) {
  const res = await fetch(`${API}/query?question=${encodeURIComponent(q)}`, {
    method: "POST",
  });
  return res.json();
}
