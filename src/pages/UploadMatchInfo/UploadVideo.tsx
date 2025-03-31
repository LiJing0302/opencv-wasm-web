import { useRef, useState } from "react";
import { Button, Message, Upload } from "@arco-design/web-react";

export default function UploadVideo({
  videoUrl,
  setVideoUrl,
}: {
  videoUrl: string | null;
  setVideoUrl: React.Dispatch<React.SetStateAction<string | null>>;
}) {
  const [loading, setLoading] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);

  const handleVideoUpload = async (file: File) => {
    try {
      setLoading(true);

      // 检查文件类型
      if (!file.type.startsWith("video/")) {
        Message.error("请上传视频文件");
        return;
      }

      // 检查文件大小（限制为 100MB）
      const maxSize = 100 * 1024 * 1024;
      if (file.size > maxSize) {
        Message.error("视频文件大小不能超过 100MB");
        return;
      }

      const dataUrl = await readFileAsDataURL(file);
      setVideoUrl(dataUrl);
      Message.success("视频上传成功");
    } catch (error) {
      Message.error("视频上传失败");
      console.error("视频上传失败:", error);
    } finally {
      setLoading(false);
    }
  };

  const readFileAsDataURL = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target?.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };

  return (
    <div>
      <h2>视频上传</h2>

      <div>
        <Upload
          accept="video/*"
          autoUpload={false}
          showUploadList={false}
          onChange={(_, uploadFile) => {
            const file = uploadFile.originFile;
            if (file) {
              handleVideoUpload(file);
            }
          }}
        >
          <Button loading={loading}>上传配对视频</Button>
        </Upload>
      </div>

      {videoUrl && (
        <div>
          <h3>视频预览</h3>
          <video ref={videoRef} src={videoUrl} controls />
        </div>
      )}
    </div>
  );
}
