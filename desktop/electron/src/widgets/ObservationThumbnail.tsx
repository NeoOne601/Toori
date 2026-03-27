import { useEffect, useState } from "react";

type ObservationThumbnailProps = {
  src: string;
  alt: string;
};

export default function ObservationThumbnail({ src, alt }: ObservationThumbnailProps) {
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    setFailed(false);
  }, [src]);

  if (failed) {
    return <div className="image-fallback">Preview unavailable</div>;
  }

  return (
    <img
      src={src}
      alt=""
      title={alt}
      crossOrigin="anonymous"
      onError={() => setFailed(true)}
    />
  );
}
