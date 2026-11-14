import { COLORS } from "./DetectionCanvas.jsx";

const VI = { bicycle: "Xe đạp", bus: "Xe buýt", car: "Ô tô con", motorbike: "Xe máy", truck: "Xe tải" };

export default function CountPanel({ counts, ms }) {
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  return (
    <div className="card">
      <h3>Thống kê phương tiện <small>({total} đối tượng · {ms} ms)</small></h3>
      <ul className="counts">
        {Object.entries(counts).map(([k, v]) => (
          <li key={k}>
            <span className="dot" style={{ background: COLORS[k] }} />
            {VI[k] ?? k}
            <b>{v}</b>
          </li>
        ))}
      </ul>
    </div>
  );
}
