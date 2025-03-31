import { useEffect, useState } from "react";
// import { Spin } from "@arco-design/web-react";
import ORB from "./orb";
import { loadOpenCV } from "@/utils/opencvLoaded";
import { Message } from "@arco-design/web-react";

const Home = () => {
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // 加载openCV
    try {
      window.Module = {
        onRuntimeInitialized: function () {
          setLoading(false);
        },
      };
      loadOpenCV(() => {
        Message.error("加载openCV成功");
      });
    } catch (e) {
      if (e instanceof Error) {
        Message.error(`加载openCV失败: ${e.message}`);
      } else {
        Message.error("加载openCV失败");
      }
    }
  }, []);
  if (loading) {
    return <span>opencv 加载中...</span>;
  }

  return <ORB />;
};

export default Home;
