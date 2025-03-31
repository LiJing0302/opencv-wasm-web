import { createWasmProcessor } from "@/utils/processor";
import { useEffect, useRef, useState } from "react";

export function VideoCanvas({ mediaStream }: { mediaStream?: MediaStream }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const playVideoRef = useRef<HTMLVideoElement>(null);
  const [canvasReady, setCanvasReady] = useState(false);
  const [playVideoReady, setPlayVideoReady] = useState(false);

  useEffect(() => {
    const playVideo = playVideoRef.current;
    if (playVideo) {
      playVideo.onloadeddata = () => {
        console.log("AR视频加载完成");
        // const videoWidth = playVideo.videoWidth;
        // const videoHeight = playVideo.videoHeight;
        // playVideo.width = videoWidth;
        // playVideo.height = videoHeight;
        playVideo.play().catch((err) => console.error("AR视频播放失败:", err));
        setPlayVideoReady(true);
      };
    }
  }, []);
  useEffect(() => {
    // 视频流展示
    const canvas = canvasRef.current;
    const video = videoRef.current;
    const playVideo = playVideoRef.current;
    if (canvas && video && playVideo && mediaStream) {
      video.srcObject = mediaStream;
      const videoTrack = mediaStream?.getVideoTracks()[0];
      const settings = videoTrack.getSettings();
      const videoWidth = settings.width || 0;
      const videoHeight = settings.height || 0;
      console.log("视频宽高", videoWidth, videoHeight);
      video.width = videoWidth;
      video.height = videoHeight;
      canvas.height = videoHeight;
      canvas.width = videoWidth;
      canvas.height = videoHeight;
      setCanvasReady(true);
    }
  }, [mediaStream]);

  useEffect(() => {
    if (canvasReady && playVideoReady) {
      const processor = createWasmProcessor({
        videoDOM: videoRef.current!,
        canvasDOM: canvasRef.current!,
        playVideoDOM: playVideoRef.current!,
      });
      processor.start();
    }
  }, [canvasReady, playVideoReady]);

  return (
    <div>
      <video
        ref={videoRef}
        style={{
          position: "fixed",
          opacity: 0,
        }}
        autoPlay
        playsInline
        muted
      />
      {/* 添加 AR 视频元素 */}
      <video
        src="/test.mp4" // 替换为你的 AR 视频路径
        loop
        autoPlay
        muted
        playsInline
        ref={playVideoRef}
        style={{
          display: "none", // 隐藏视频元素
        }}
      />
      <canvas
        ref={canvasRef}
        style={{
          objectFit: "cover",
          width: "100vw",
        }}
      />
    </div>
  );
}
