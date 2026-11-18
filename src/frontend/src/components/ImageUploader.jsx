import { useRef, useState } from "react";

export default function ImageUploader({ onFile }) {
  const input = useRef(null);
  const [drag, setDrag] = useState(false);

  const pick = (files) => {
    const f = files?.[0];
    if (f && f.type.startsWith("image/")) onFile(f);
  };

  return (
    <div
      className={`dropzone ${drag ? "drag" : ""}`}
      onClick={() => input.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files); }}
    >
      <input ref={input} type="file" accept="image/*" hidden onChange={(e) => pick(e.target.files)} />
      <p><b>Kéo-thả ảnh</b> hoặc bấm để chọn</p>
      <small>JPG / PNG / WEBP, tối đa 10 MB</small>
    </div>
  );
}
