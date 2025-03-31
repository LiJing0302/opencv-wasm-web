import { fetchStreamCamera } from "@/utils";
import { useEffect, useState } from "react";

export const useFetchStreamCamera = () => {
  const [loading, setLoading] = useState(true);
  const [mediaStream, setMediaStream] = useState<MediaStream>();
  useEffect(() => {
    fetchStreamCamera().then((media) => {
      setMediaStream(media);
      setLoading(false);
    });
  }, [loading]);
  return {
    loading,
    mediaStream,
  };
};
