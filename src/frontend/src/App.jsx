import { useEffect, useState } from "react";
import { classify, detect, health } from "./api.js";
import ImageUploader from "./components/ImageUploader.jsx";
import DetectionCanvas from "./components/DetectionCanvas.jsx";
import CountPanel from "./components/CountPanel.jsx";
import ProbabilityBars from "./components/ProbabilityBars.jsx";

export default function App() {
  const [mode, setMode] = useState("detect"); // "detect" | "classify"
  const [file, setFile] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [conf, setConf] = useState(0.25);
  const [iou, setIou] = useState(0.5);
  const [refine, setRefine] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState(null);

  useEffect(() => {
    health().then(setStatus).catch(() => setStatus(null));
  }, []);

  useEffect(() => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setImageUrl(url);
    setResult(null);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  async function run() {
    if (!file) return;
    setLoading(true);
    setError("");
    try {
      const r = mode === "detect" ? await detect(file, { conf, iou, refine }) : await classify(file);
      setResult({ mode, ...r });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Phát hiện &amp; phân loại phương tiện bằng CNN</h1>
        <span className={`badge ${status ? "ok" : "off"}`}>
          API {status ? `online · ${status.device}` : "offline"}
        </span>
      </header>

      <div className="tabs" role="tablist">
        {[["detect", "Phát hiện (YOLOv8)"], ["classify", "Phân loại (CNN)"]].map(([k, label]) => (
          <button key={k} role="tab" aria-selected={mode === k}
                  className={mode === k ? "active" : ""}
                  onClick={() => { setMode(k); setResult(null); }}>
            {label}
          </button>
        ))}
      </div>

      <main>
        <section className="left">
          <ImageUploader onFile={setFile} />
          {mode === "detect" && (
            <div className="controls">
              <label>Ngưỡng confidence: <b>{conf.toFixed(2)}</b>
                <input type="range" min="0.05" max="0.95" step="0.05" value={conf}
                       onChange={(e) => setConf(+e.target.value)} />
              </label>
              <label>Ngưỡng IoU (NMS): <b>{iou.toFixed(2)}</b>
                <input type="range" min="0.1" max="0.9" step="0.05" value={iou}
                       onChange={(e) => setIou(+e.target.value)} />
              </label>
              <label className="check">
                <input type="checkbox" checked={refine} onChange={(e) => setRefine(e.target.checked)} />
                Phân loại lại từng hộp bằng CNN phân loại
              </label>
            </div>
          )}
          <button className="primary" disabled={!file || loading} onClick={run}>
            {loading ? "Đang suy luận…" : "Chạy mô hình"}
          </button>
          {error && <p className="error">{error}</p>}

          {result?.mode === "detect" && <CountPanel counts={result.counts} ms={result.inference_ms} />}
          {result?.mode === "classify" && <ProbabilityBars result={result} />}
        </section>

        <section className="right">
          {imageUrl ? (
            <DetectionCanvas src={imageUrl}
                             detections={result?.mode === "detect" ? result.detections : []} />
          ) : (
            <div className="placeholder">Chọn hoặc kéo-thả ảnh giao thông để bắt đầu</div>
          )}
        </section>
      </main>
    </div>
  );
}
