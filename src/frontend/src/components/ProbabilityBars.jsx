import { COLORS } from "./DetectionCanvas.jsx";

// Hiển thị phân phối softmax trên 5 lớp phương tiện
export default function ProbabilityBars({ result }) {
  return (
    <div className="card">
      <h3>
        Kết quả: {result.top1.label_vi} ({(result.top1.probability * 100).toFixed(1)}%)
        <small> · {result.model} · {result.inference_ms} ms</small>
      </h3>
      {result.probabilities.map((p) => (
        <div className="bar-row" key={p.label}>
          <span>{p.label_vi}</span>
          <div className="bar">
            <div style={{ width: `${p.probability * 100}%`, background: COLORS[p.label] }} />
          </div>
          <span>{(p.probability * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}
