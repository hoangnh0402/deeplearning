// Gọi FastAPI backend. Khi dev, Vite proxy /api → http://localhost:8000
const BASE = import.meta.env.VITE_API_URL ?? "";

async function postImage(path, file, params = {}) {
  const form = new FormData();
  form.append("file", file);
  const qs = new URLSearchParams(params).toString();
  const res = await fetch(`${BASE}${path}${qs ? `?${qs}` : ""}`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  return res.json();
}

export const detect = (file, { conf, iou, refine }) =>
  postImage("/api/detect", file, { conf, iou, refine });

export const classify = (file) => postImage("/api/classify", file);

export const health = () => fetch(`${BASE}/api/health`).then((r) => r.json());
