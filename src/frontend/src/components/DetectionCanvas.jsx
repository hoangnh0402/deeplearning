import { useEffect, useRef } from "react";

export const COLORS = {
  bicycle: "#2a9d8f",
  bus: "#e9c46a",
  car: "#4361ee",
  motorbike: "#e76f51",
  truck: "#8338ec",
};

// Vẽ ảnh gốc + bounding box lên <canvas>; toạ độ hộp theo pixel ảnh gốc.
export default function DetectionCanvas({ src, detections }) {
  const canvas = useRef(null);

  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      const c = canvas.current;
      if (!c) return;
      c.width = img.naturalWidth;
      c.height = img.naturalHeight;
      const ctx = c.getContext("2d");
      ctx.drawImage(img, 0, 0);
      const lw = Math.max(2, Math.round(img.naturalWidth / 400));
      ctx.font = `${12 + lw * 3}px system-ui, sans-serif`;
      for (const d of detections) {
        const { x1, y1, x2, y2 } = d.box;
        const color = COLORS[d.label] ?? "#ff006e";
        ctx.strokeStyle = color;
        ctx.lineWidth = lw;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        const text = `${d.label_vi} ${(d.confidence * 100).toFixed(0)}%${d.refined_by_classifier ? " ✓" : ""}`;
        const tw = ctx.measureText(text).width + 8;
        const th = 16 + lw * 3;
        ctx.fillStyle = color;
        ctx.fillRect(x1, Math.max(0, y1 - th), tw, th);
        ctx.fillStyle = "#fff";
        ctx.fillText(text, x1 + 4, Math.max(th - 4, y1 - 4));
      }
    };
    img.src = src;
  }, [src, detections]);

  return <canvas ref={canvas} className="canvas" aria-label="Ảnh kết quả phát hiện" />;
}
