import { useMemo, useState } from "react";
import { Drawer } from "@arco-design/web-react";
import { FeaturePoint } from "@/types/orbFeaturePoint";
import ImgFeatureDetector from "./FeatureDetector";
import UploadVideo from "./UploadVideo";

export type Pair = {
  imgFeaturePoints: FeaturePoint[];
  video: string;
};

export const useUploadMatchInfo = () => {
  const [visible, setVisible] = useState(false);
  const [featurePoints, setFeaturePoints] = useState<FeaturePoint[]>([]);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);

  const pair = useMemo<Pair>(() => {
    return {
      imgFeaturePoints: featurePoints,
      video: videoUrl || "",
    };
  }, [featurePoints, videoUrl]);
  const onOpen = () => {
    setVisible(true);
  };
  return {
    onOpen,
    visible,
    setVisible,
    featurePoints,
    setFeaturePoints,
    videoUrl,
    setVideoUrl,
    pair,
  };
};

export default function UploadMatchInfo(
  props: Omit<ReturnType<typeof useUploadMatchInfo>, "onOpen">
) {
  const {
    visible,
    setVisible,
    featurePoints,
    setFeaturePoints,
    videoUrl,
    setVideoUrl,
  } = props;

  return (
    <Drawer width={"80%"} visible={visible} onCancel={() => setVisible(false)}>
      <ImgFeatureDetector
        setFeaturePoints={setFeaturePoints}
        featurePoints={featurePoints}
      />
      <UploadVideo videoUrl={videoUrl} setVideoUrl={setVideoUrl} />
    </Drawer>
  );
}
