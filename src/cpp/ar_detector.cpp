#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/video.hpp>
#include <chrono>
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <emscripten/fetch.h>

using namespace cv;
using namespace std;
using namespace std::chrono;

// 添加 createMask 辅助函数，创建目标区域的掩码图像，img_size 输入图像大小，pts 四个角点的坐标
Mat createMask(cv::Size img_size, vector<Point2f> &pts)
{
    Mat mask(img_size, CV_8UC1); // 8U 表示 8 位无符号整数（取值范围 0-255）; C1表示单通道（灰度图）
    float zero = 0;
    mask = zero; // 初始化掩码全为0（黑色）
    // 定义边界方程的系数数组
    float a[4], b[4], c[4];
    // 计算四条边的直线方程系数 (ax + by + c = 0)
    // 第一组：左边和右边
    a[0] = pts[3].y - pts[0].y; // 左边
    a[1] = pts[2].y - pts[1].y; // 右边
    // 第二组：上边和下边
    a[2] = pts[1].y - pts[0].y; // 上边
    a[3] = pts[2].y - pts[3].y; // 下边
    // b 系数
    b[0] = pts[0].x - pts[3].x; // 左边
    b[1] = pts[1].x - pts[2].x; // 右边
    b[2] = pts[0].x - pts[1].x; // 上边
    b[3] = pts[3].x - pts[2].x; // 下边
    // c 系数（叉积）
    c[0] = pts[0].y * pts[3].x - pts[3].y * pts[0].x; // 左边
    c[1] = pts[1].y * pts[2].x - pts[2].y * pts[1].x; // 右边
    c[2] = pts[0].y * pts[1].x - pts[1].y * pts[0].x; // 上边
    c[3] = pts[3].y * pts[2].x - pts[2].y * pts[3].x; // 下边

    // 计算边界框，找出四个角点的最大最小x,y坐标
    float max_x = 0, min_x = img_size.width;
    float max_y = 0, min_y = img_size.height;

    for (int i = 0; i < 4; i++)
    {
        if (pts[i].x > max_x)
            max_x = pts[i].x;
        if (pts[i].x < min_x)
            min_x = pts[i].x;
        if (pts[i].y > max_y)
            max_y = pts[i].y;
        if (pts[i].y < min_y)
            min_y = pts[i].y;
    }

    // 边界检查
    max_x = min(max_x, (float)img_size.width - 1);
    max_y = min(max_y, (float)img_size.height - 1);
    min_x = max(min_x, 0.0f);
    min_y = max(min_y, 0.0f);

    // 填充掩码
    unsigned char *ptr = mask.data; // 获取掩码数据指针
    for (int y = min_y; y <= max_y; y++)
    {
        int offset = y * img_size.width; // 计算行偏移
        for (int x = min_x; x <= max_x; x++)
        {
            float val[4];
            // 计算点(x,y)相对于四条边的位置
            for (int i = 0; i < 4; i++)
            {
                val[i] = a[i] * x + b[i] * y + c[i];
            }
            // 如果点在四边形内部，设置掩码值为255（白色）
            if (val[0] * val[1] <= 0 && val[2] * val[3] <= 0)
                *(ptr + offset + x) = 255;
        }
    }
    return mask; // 返回二值掩码，目标区域像素值为 255（白色）背景区域像素值为 0（黑色）
}

extern "C"
{
    static float fps = 0.0f;
    static high_resolution_clock::time_point lastTime = high_resolution_clock::now();
    Mat template_img;         // 存储模板图像
    bool is_tracking = false; // 跟踪状态标志

    // 添加视频处理相关变量
    Mat videoFrame; // 存储视频帧

    // ORB 特征检测器
    Ptr<ORB> orb = ORB::create(300, 1.2f, 4,      // 金字塔层数（可以减少）
                               31,                // 边缘阈值
                               0,                 // 第一层级
                               2,                 // WTA_K 点数
                               ORB::HARRIS_SCORE, // 使用 HARRIS 角点评分
                               31,                // 特征点描述符大小
                               20);               // FAST 检测阈值); // ORB 特征检测器
    vector<KeyPoint>
        keyPoints_1;   // 模板图像的关键点
    Mat descriptors_1; // 模板图像的特征描述符

    // 跟踪相关变量
    Mat prev_gray, curr_gray;       // 前一帧和当前帧的灰度图
    vector<Point2f> prev_keyPoints; // 前一帧的特征点
    vector<Point2f> curr_keyPoints; // 当前帧的特征点
    vector<Point2f> prev_corners;   // 前一帧的角点
    vector<Point2f> curr_corners;   // 当前帧的角点
    // 添加光流金字塔变量
    vector<Mat> prevPyr, nextPyr; // 光流金字塔
    vector<uchar> track_status;   // 跟踪状态

    // 在图像上绘制帧率
    void drawFPS(Mat &frame, float fps)
    {
        string fpsText = "FPS: " + to_string(static_cast<int>(fps));
        putText(frame, fpsText, Point(10, 30), FONT_HERSHEY_SIMPLEX,
                1.0, Scalar(255, 0, 0), 2);
    }

    // 计算帧率
    float calculateFPS()
    {
        auto currentTime = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(currentTime - lastTime).count();
        lastTime = currentTime;
        return 1000.0f / duration;
    }

    // 添加设置参考图像的函数
    void set_reference_image(uint8_t *imageData, int width, int height)
    {
        Mat frame(height, width, CV_8UC4, imageData);   // 创建 RGBA 格式 Mat 对象
        cvtColor(frame, template_img, COLOR_RGBA2GRAY); // 灰度图转换

        // 检测特征点和计算描述子
        orb->detectAndCompute(template_img, Mat(), keyPoints_1, descriptors_1);
    }

    bool detect_objects_and_homography(uint8_t *image_data, int width, int height, uint8_t *video_data, int video_width, int video_height)
    {
        Mat img(height, width, CV_8UC4, image_data); // 创建输入RGBA格式的图像
        // 如果提供了视频数据，创建视频帧
        Mat videoFrameRGBA;
        if (video_data != nullptr && video_width > 0 && video_height > 0)
        {
            videoFrame = Mat(video_height, video_width, CV_8UC4, video_data);
        }
        // fps = calculateFPS();

        // // 添加绘制帧率
        // drawFPS(img, fps);

        if (img.empty()) // 图像为空则返回失败
            return false;

        // 图像处理逻辑
        if (!is_tracking)
        {
            static int skip_frames = 0;
            if (++skip_frames < 2)
            { // 每隔2帧才进行特征检测
                return false;
            }
            skip_frames = 0;

            // 对输入图像进行降采样 - 使用pyrDown更高效
            Mat resized;
            pyrDown(img, resized);

             // 增加预处理步骤：对图像进行直方图均衡化以增强对比度
             Mat gray_resized;
             cvtColor(resized, gray_resized, COLOR_BGRA2GRAY);
             equalizeHist(gray_resized, gray_resized);
 

            // 预先分配内存并减少特征点数量
            vector<KeyPoint> keyPoints_2;
            keyPoints_2.reserve(300); // 根据ORB创建的最大特征点数
            Mat descriptors_2;
            orb->detectAndCompute(resized, Mat(), keyPoints_2, descriptors_2);
            if (keyPoints_2.size() >= 10) // 确保检测到足够的特征点
            {
                // 将特征点坐标映射回原始分辨率
                for (auto &kp : keyPoints_2)
                {
                    kp.pt.x *= 2.0;
                    kp.pt.y *= 2.0;
                }
                // 创建特征匹配器（汉明距离）
                Ptr<BFMatcher> matcher = BFMatcher::create(NORM_HAMMING);
                // 筛选好的匹配点
                vector<Point2f> train_p, query_p;
                train_p.reserve(descriptors_1.rows);
                query_p.reserve(descriptors_1.rows);
                // 使用比率测试法筛选匹配点，更可靠的匹配策略
                const float ratio_thresh = 0.75f;
                vector<vector<DMatch>> knn_matches;
                matcher->knnMatch(descriptors_2, descriptors_1, knn_matches, 2);
                for (const auto &match : knn_matches)
                {
                    if (match.size() < 2)
                        continue;
                    if (match[0].distance < ratio_thresh * match[1].distance)
                    {
                        train_p.push_back(keyPoints_2[match[0].queryIdx].pt);
                        query_p.push_back(keyPoints_1[match[0].trainIdx].pt);
                    }
                }

                // 计算单应性矩阵至少需要4对匹配点(数学硬要求),少于4对点无法确定唯一的透视变换
                // 要求好的匹配点数量至少占原始特征点的10%
                // 这两点要求构成一个质量控制阈值，确保有足够多的可靠匹配
                if (train_p.size() >= 4 && train_p.size() >= (size_t)(0.05 * descriptors_1.rows))
                {
                    // 计算单应性矩阵
                    vector<uchar> inliers;
                    Mat H = findHomography(query_p, train_p, RANSAC, 3.0, inliers, 500, 0.995);

                    // 检查内点数量和比例
                    int inlierCount = countNonZero(inliers);
                    if (!H.empty() && inlierCount >= 8 && (float)inlierCount / train_p.size() > 0.6) // 只处理单应性矩阵不为空的情况
                    {
                        // 计算模板图像的四个角点在当前图像中的位置
                        vector<Point2f> obj_corners(4);
                        obj_corners[0] = Point2f(0, 0);
                        obj_corners[1] = Point2f(template_img.cols, 0);
                        obj_corners[2] = Point2f(template_img.cols, template_img.rows);
                        obj_corners[3] = Point2f(0, template_img.rows);

                        vector<Point2f> scene_corners(4);
                        // 定义目标区域角点，再根据单应性变换到当前摄像头画面图像中的位置
                        perspectiveTransform(obj_corners, scene_corners, H);

                        // 初始化光流跟踪
                        cvtColor(img, prev_gray, COLOR_BGRA2GRAY);
                        prev_corners = scene_corners;
                        prev_keyPoints.clear();

                        // 在目标区域内检测特征点
                        Mat mask = createMask(img.size(), scene_corners);                   // 使用 createMask 函数
                        goodFeaturesToTrack(prev_gray, prev_keyPoints, 80, 0.01, 10, mask); // prev_keyPoints是输出参数，存储检测到的特征点的位置
                        prevPyr.clear();                                                    // 清空光流金字塔
                        track_status.clear();                                               // 清空跟踪状态

                        // 绘制检测结果
                        for (int i = 0; i < 4; i++)
                        {
                            line(img, scene_corners[i], scene_corners[(i + 1) % 4],
                                 Scalar(0, 255, 0, 255), 2);
                        }
                        // 绘制特征点
                        for (const auto &kp : prev_keyPoints)
                        {
                            circle(img, kp, 3, Scalar(255, 0, 0, 255), -1);
                        }

                        is_tracking = true;
                        return true;
                    }
                }
            }
        }
        else
        {
            // 检查跟踪点是否在有效范围内
            bool valid_corners = true;
            // 检查跟踪点是否在有效范围内
            for (const auto &corner : prev_corners)
            {
                if (corner.x < 0 || corner.x >= img.cols ||
                    corner.y < 0 || corner.y >= img.rows)
                {
                    valid_corners = false;
                    break;
                }
            }

            if (!valid_corners || prev_keyPoints.size() < 10)
            {
                is_tracking = false;
                return false;
            }

            // 对输入图像进行降采样
            Mat resized;
            resize(img, resized, Size(), 0.5, 0.5);

            // 光流跟踪
            cvtColor(resized, curr_gray, COLOR_BGRA2GRAY);
            vector<float> err;

            // 将跟踪点坐标缩放到降采样分辨率
            vector<Point2f> scaled_prev_points;
            for (const auto &pt : prev_keyPoints)
            {
                scaled_prev_points.push_back(Point2f(pt.x * 0.5, pt.y * 0.5));
            }

            // 构建图像金字塔
            if (prevPyr.empty())
            {
                Mat scaled_prev_gray;
                resize(prev_gray, scaled_prev_gray, Size(), 0.5, 0.5);
                buildOpticalFlowPyramid(scaled_prev_gray, prevPyr, Size(31, 31), 4, true);
            }
            buildOpticalFlowPyramid(curr_gray, nextPyr, Size(31, 31), 4, true);

            // 使用金字塔光流跟踪calcOpticalFlowPyrLK函数
            // - 实现了Lucas-Kanade光流算法的金字塔版本
            // - 第一、二个参数：前一帧和当前帧的图像金字塔
            // - 第三个参数：前一帧中要跟踪的特征点（ prev_keyPoints ）
            // - 第四个参数：输出参数，计算得到的当前帧中特征点的新位置（ curr_keyPoints ）
            // - 第五个参数：输出参数，跟踪状态（ track_status ），1表示成功跟踪，0表示失败
            // - 第六个参数：输出参数，跟踪误差（ err ）
            // - 第七个参数： Size(21, 21) 搜索窗口大小
            // - 第八个参数： 3 表示金字塔的最大层数
            calcOpticalFlowPyrLK(prevPyr, nextPyr, scaled_prev_points, curr_keyPoints,
                                 track_status, err, Size(31, 31), 4);

            // 将跟踪结果坐标映射回原始分辨率
            for (auto &pt : curr_keyPoints)
            {
                pt.x *= 2.0;
                pt.y *= 2.0;
            }
            // 筛选有效的光流跟踪点
            vector<Point2f> tracked_prev, tracked_curr;
            for (size_t i = 0; i < track_status.size(); i++)
            {
                // prev_keyPoints.size() > i 安全检查，防止数组越界
                if (track_status[i] && prev_keyPoints.size() > i && norm(curr_keyPoints[i] - prev_keyPoints[i]) <= 25)
                {
                    tracked_prev.push_back(prev_keyPoints[i]);
                    tracked_curr.push_back(curr_keyPoints[i]);
                }
            }

            if (tracked_prev.size() >= 10)
            {
                vector<uchar> inliers;
                // 计算单应性矩阵
                Mat H = findHomography(tracked_prev, tracked_curr, RANSAC, 3.0, inliers, 300, 0.95);
                if (!H.empty())
                {
                    // 检查内点数量，确保变换的可靠性
                    int inlierCount = countNonZero(inliers);
                    if (inlierCount < 6)
                    { // 如果内点太少，认为是不可靠的变换
                        is_tracking = false;
                        return false;
                    }
                    // 更新角点位置
                    curr_corners.clear(); // 清空当前角点
                    // 根据单应性变换到当前摄像头画面图像中的位置
                    perspectiveTransform(prev_corners, curr_corners, H);

                    // 更新跟踪状态
                    swap(prev_gray, curr_gray);
                    prevPyr.swap(nextPyr);
                    prev_keyPoints = tracked_curr;
                    prev_corners = curr_corners;

                    if (!videoFrame.empty())
                    {
                        // 将视频帧转换为RGBA格式（如果需要）
                        if (videoFrame.channels() != 4)
                        {
                            cvtColor(videoFrame, videoFrameRGBA, COLOR_BGR2BGRA);
                        }
                        else
                        {
                            videoFrameRGBA = videoFrame;
                        }

                        // 创建目标区域的四个角点
                        std::vector<Point2f> videoCorners(4);
                        videoCorners[0] = Point2f(0, 0);
                        videoCorners[1] = Point2f(videoFrame.cols - 1, 0);
                        videoCorners[2] = Point2f(videoFrame.cols - 1, videoFrame.rows - 1);
                        videoCorners[3] = Point2f(0, videoFrame.rows - 1);

                        // 计算从视频帧到检测区域的透视变换矩阵
                        Mat perspectiveMatrix = getPerspectiveTransform(videoCorners, curr_corners);
                        // 创建与原图相同大小的空白图像
                        Mat warped = Mat::zeros(img.size(), CV_8UC4);

                        // 对视频帧进行透视变换，将视频投影到检测到的区域
                        warpPerspective(videoFrameRGBA, warped, perspectiveMatrix, img.size());

                        // 创建掩码，确定要替换的区域
                        Mat mask = Mat::zeros(img.size(), CV_8UC1);
                        vector<vector<Point>> contours;
                        vector<Point> contour;
                        for (const auto &corner : curr_corners)
                        {
                            contour.push_back(Point(corner.x, corner.y));
                        }
                        contours.push_back(contour);
                        fillPoly(mask, contours, Scalar(255));

                        // 将变换后的视频与原图像合成
                        warped.copyTo(img, mask);
                    }

                    // // 绘制跟踪结果
                    // for (int i = 0; i < 4; i++)
                    // {
                    //     line(img, curr_corners[i], curr_corners[(i + 1) % 4],
                    //          Scalar(0, 255, 0, 255), 2);
                    // }

                    // // 绘制跟踪点
                    // for (const auto &kp : tracked_curr)
                    // {
                    //     circle(img, kp, 3, Scalar(255, 0, 0, 255), -1);
                    // }

                    return true;
                }
            }
            else
            {
                is_tracking = false;
                prevPyr.clear();
                nextPyr.clear();
            }
        }
        return false;
    }

    uint8_t *detect_orb_features(uint8_t *imageData, int width, int height)
    {
        // 创建输入图像
        Mat img(height, width, CV_8UC4, imageData);
        Mat gray;
        cvtColor(img, gray, COLOR_RGBA2GRAY);

        // 创建 ORB 检测器
        Ptr<ORB> detector = ORB::create(1000);

        // 检测特征点
        vector<KeyPoint> keypoints;
        Mat descriptors;
        detector->detectAndCompute(gray, Mat(), keypoints, descriptors);

        // 计算需要的内存大小
        // 格式: [特征点数量(4字节) + 每个特征点(28字节) + 描述符(32字节)]
        size_t numFeatures = keypoints.size();
        size_t memSize = sizeof(int) +
                         (sizeof(float) * 5 + sizeof(int) * 2) * numFeatures +
                         descriptors.cols * numFeatures;

        // 分配内存
        uint8_t *result = (uint8_t *)malloc(memSize);

        // 写入特征点数量
        *((int *)result) = numFeatures;

        // 写入特征点信息
        size_t offset = sizeof(int);
        for (size_t i = 0; i < numFeatures; i++)
        {
            const KeyPoint &kp = keypoints[i];

            // 写入浮点数据
            *((float *)(result + offset)) = kp.pt.x;
            *((float *)(result + offset + sizeof(float))) = kp.pt.y;
            *((float *)(result + offset + sizeof(float) * 2)) = kp.size;
            *((float *)(result + offset + sizeof(float) * 3)) = kp.angle;
            *((float *)(result + offset + sizeof(float) * 4)) = kp.response;

            // 写入整数数据
            *((int *)(result + offset + sizeof(float) * 5)) = kp.octave;
            *((int *)(result + offset + sizeof(float) * 5 + sizeof(int))) = kp.class_id;

            // 写入描述符
            memcpy(result + offset + sizeof(float) * 5 + sizeof(int) * 2,
                   descriptors.ptr(i),
                   descriptors.cols);

            offset += sizeof(float) * 5 + sizeof(int) * 2 + descriptors.cols;
        }

        return result;
    }
};
