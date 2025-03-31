#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp> // 恢复这个头文件
#include <opencv2/video/tracking.hpp>
#include <chrono>
#include <emscripten/bind.h>
#include <emscripten/val.h>

using namespace cv;
using namespace std;
using namespace std::chrono;

extern "C"
{
    static high_resolution_clock::time_point lastTime = high_resolution_clock::now();
    static float fps = 0.0f;
    bool isReferenceLoaded = false;
    // 初始化特征检测器和匹配器
    Ptr<FeatureDetector> detector = ORB::create();

    // 存储参考图像和特征点
    static Ptr<ORB> orb = ORB::create(1000); // 限制特征点数量
    static Mat referenceImage;
    static vector<KeyPoint> referenceKeypoints;
    static Mat referenceDescriptors;
    static Ptr<DescriptorMatcher> matcher = BFMatcher::create(NORM_HAMMING);

    // 添加光流跟踪所需变量
    static bool isTracking = false;
    static Mat prevGray;
    static vector<Point2f> prevKeyPoints;
    static vector<Point2f> prevCorners;
    static vector<Point2f> currCorners;
    static vector<unsigned char> trackStatus;
    static vector<float> trackError;
    // 添加光流金字塔变量
    static vector<Mat> prevPyr;
    // 修改常量定义
    static const int MAX_TRACK_FAILURES = 10; // 增加容忍度
    static int trackFailureCount = 0;
    static const float MAX_TRACK_ERROR = 12.0f;      // 最大允许的跟踪误差
    static const int MIN_INLIER_COUNT = 10;          // 最小内点数量
    static float prevArea = 0;                       // 记录上一帧的四边形面积
    static const float MAX_AREA_CHANGE_RATIO = 0.3f; // 最大面积变化比例

    // 添加相机参数
    static float f_x = 640.0f;
    static float f_y = 640.0f;
    static float c_x = 320.0f;
    static float c_y = 240.0f;
    static float camera_matrix[] = {
        f_x, 0.0f, c_x,
        0.0f, f_y, c_y,
        0.0f, 0.0f, 1.0f};

    static float dist_coeff[] = {0.0f, 0.0f, 0.0f, 0.0f};
    static Mat camera_matrix_mat = Mat(3, 3, CV_32FC1, camera_matrix).clone();
    static Mat dist_coeff_mat = Mat(1, 4, CV_32FC1, dist_coeff).clone();
    static vector<Point3f> object_points;

    static Mat rotation_matrix, translation_vector;

    // 计算帧率
    float calculateFPS()
    {
        auto currentTime = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(currentTime - lastTime).count();
        lastTime = currentTime;
        return 1000.0f / duration;
    }

    // 在图像上绘制帧率
    void drawFPS(Mat &frame, float fps)
    {
        string fpsText = "FPS: " + to_string(static_cast<int>(fps));
        putText(frame, fpsText, Point(10, 30), FONT_HERSHEY_SIMPLEX,
                1.0, Scalar(255, 0, 0), 2);
    }

    // 添加设置参考图像的函数
    void set_reference_image(uint8_t *imageData, int width, int height)
    {
        Mat frame(height, width, CV_8UC4, imageData);     // 创建 RGBA 格式 Mat 对象
        cvtColor(frame, referenceImage, COLOR_RGBA2GRAY); // 灰度图转换

        // 计算宽高比
        float aspect_ratio = (float)width / (float)height;

        // 设置3D角点坐标,使用宽高比来保持比例
        Point3f corners_3d[] = {
            Point3f(-aspect_ratio, -1.0f, 0), // 左上
            Point3f(aspect_ratio, -1.0f, 0),  // 右上
            Point3f(aspect_ratio, 1.0f, 0),   // 右下
            Point3f(-aspect_ratio, 1.0f, 0)   // 左下
        };
        object_points = vector<Point3f>(corners_3d, corners_3d + 4);

        // 检测特征点和计算描述子
        orb->detectAndCompute(referenceImage, Mat(), referenceKeypoints, referenceDescriptors);
        isReferenceLoaded = true;
        isTracking = false; // 重置跟踪状态
    }

    // 修改函数声明，避免C链接警告
    static Mat createMask(Size imgSize, vector<Point2f> &pts)
    {
        Mat mask(imgSize, CV_8UC1, Scalar(0));

        // 确保有四个点
        if (pts.size() != 4)
            return mask;

        // 创建多边形
        vector<Point> poly;
        for (const auto &pt : pts)
        {
            poly.push_back(Point(pt.x, pt.y));
        }

        // 填充多边形区域
        fillConvexPoly(mask, poly, Scalar(255));
        return mask;
    }

    // 保留原有函数以满足导出需求
    void detect_orb_features(uint8_t *inputImage, int width, int height)
    {
        // 计算当前帧率
        fps = calculateFPS();

        // 将输入数据转换为 OpenCV Mat
        Mat frame(height, width, CV_8UC4, inputImage);

        // 创建 ORB 检测器
        Ptr<ORB> orb = ORB::create();
        vector<KeyPoint> keypoints;

        // 转换为灰度图
        Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_RGBA2GRAY);

        // 检测特征点
        orb->detect(gray, keypoints);

        // 在原图上绘制特征点，使用更小的圆圈
        drawKeypoints(frame, keypoints, frame,
                      Scalar(0, 255, 0),          // 保持绿色
                      DrawMatchesFlags::DEFAULT); // 使用默认绘制模式，不显示方向和大小

        // 绘制帧率
        drawFPS(frame, fps);
    }

    // 边缘检测函数
    void detect_edges(uint8_t *inputImage, uint8_t *outputImage, int width, int height)
    {
        // 计算帧率
        fps = calculateFPS();

        // 将输入数据转换为 OpenCV Mat
        Mat frame(height, width, CV_8UC4, inputImage);
        Mat output(height, width, CV_8UC4, outputImage);

        // 转换为灰度图
        Mat gray;
        cvtColor(frame, gray, COLOR_RGBA2GRAY);

        // 使用 Canny 边缘检测
        Mat edges;
        Canny(gray, edges, 100, 200);

        // 转换回 RGBA 格式
        cvtColor(edges, output, COLOR_GRAY2RGBA);

        // 绘制帧率
        drawFPS(output, fps);
    }

    // 添加 PnP 求解函数
    static void estimatePose(const vector<Point2f> &corners, Mat &rvec, Mat &tvec)
    {
        if (object_points.empty())
        {
            // 设置 3D 点（假设平面对象）
            float width = 1.0f;
            float height = 1.0f;
            object_points = {
                Point3f(-width / 2, -height / 2, 0),
                Point3f(width / 2, -height / 2, 0),
                Point3f(width / 2, height / 2, 0),
                Point3f(-width / 2, height / 2, 0)};
        }
        solvePnP(object_points, corners, camera_matrix_mat, dist_coeff_mat, rvec, tvec, false, SOLVEPNP_ITERATIVE);
    }

    void detect_and_match_features(uint8_t *inputImage, int width, int height)
    {
        // 计算帧率
        fps = calculateFPS();

        // 检查是否已设置参考图像
        if (referenceImage.empty())
        {
            return;
        }

        // 将输入数据转换为 OpenCV Mat
        Mat frame(height, width, CV_8UC4, inputImage);
        Mat gray;
        cvtColor(frame, gray, COLOR_RGBA2GRAY);

        // 检测当前帧的特征点和描述符
        vector<KeyPoint> currentKeypoints;
        Mat currentDescriptors;
        orb->detectAndCompute(gray, Mat(), currentKeypoints, currentDescriptors);

        // 特征匹配
        vector<vector<DMatch>> knnMatches;
        matcher->knnMatch(currentDescriptors, referenceDescriptors, knnMatches, 2);

        // 应用比率测试筛选好的匹配
        vector<DMatch> goodMatches;
        for (const auto &match : knnMatches)
        {
            if (match.size() >= 2 && match[0].distance < 0.75 * match[1].distance)
            {
                goodMatches.push_back(match[0]);
            }
        }

        // 如果有足够的好匹配点，计算单应性矩阵
        if (goodMatches.size() >= 4)
        {
            vector<Point2f> srcPoints, dstPoints;
            for (const auto &match : goodMatches)
            {
                srcPoints.push_back(currentKeypoints[match.queryIdx].pt);
                dstPoints.push_back(referenceKeypoints[match.trainIdx].pt);
            }

            Mat H = findHomography(srcPoints, dstPoints, RANSAC);

            if (!H.empty())
            {
                // 获取参考图像的角点
                vector<Point2f> corners(4);
                corners[0] = Point2f(0, 0);
                corners[1] = Point2f(referenceImage.cols, 0);
                corners[2] = Point2f(referenceImage.cols, referenceImage.rows);
                corners[3] = Point2f(0, referenceImage.rows);

                // 变换角点
                vector<Point2f> transformedCorners;
                perspectiveTransform(corners, transformedCorners, H.inv());

                // 绘制边框
                for (int i = 0; i < 4; i++)
                {
                    line(frame, transformedCorners[i], transformedCorners[(i + 1) % 4],
                         Scalar(0, 255, 0), 2);
                }
            }
        }

        // 绘制匹配点
        drawMatches(frame, currentKeypoints, referenceImage, referenceKeypoints,
                    goodMatches, frame, Scalar(0, 255, 0), Scalar::all(-1));

        // 绘制帧率
        drawFPS(frame, fps);
    }

    // 优化后的目标检测和单应性变换函数
    void detect_objects_and_homography(uint8_t *inputData, uint8_t *outputData, int width, int height)
    {
        // 参数检查
        if (!inputData || !outputData || width <= 0 || height <= 0)
        {
            return;
        }

        fps = calculateFPS();

        // 确保参考图像已加载
        if (!isReferenceLoaded)
        {
            return;
        }

        // 将输入数据转换为OpenCV格式
        Mat src(height, width, CV_8UC4, inputData);
        Mat dst(height, width, CV_8UC4, outputData);

        // 复制原始图像到输出
        src.copyTo(dst);

        // 转换为灰度图
        Mat grayFrame;
        cvtColor(src, grayFrame, COLOR_RGBA2GRAY);

        // 应该预分配内存
        grayFrame.create(src.size(), CV_8UC1);
        cvtColor(src, grayFrame, COLOR_RGBA2GRAY);

        // 状态机：跟踪模式和检测模式
        if (!isTracking)
        {
            // 检测模式：使用特征点匹配
            vector<KeyPoint> frameKeypoints;
            Mat frameDescriptors;
            detector->detectAndCompute(grayFrame, noArray(), frameKeypoints, frameDescriptors);

            // 如果找到特征点，进行匹配
            if (!frameKeypoints.empty() && !frameDescriptors.empty() && !referenceKeypoints.empty())
            {
                // 特征点匹配
                vector<vector<DMatch>> knnMatches;
                matcher->knnMatch(referenceDescriptors, frameDescriptors, knnMatches, 2);

                // 应用比率测试筛选好的匹配
                vector<DMatch> goodMatches;
                for (size_t i = 0; i < knnMatches.size(); i++)
                {
                    if (knnMatches[i].size() >= 2)
                    {
                        if (knnMatches[i][0].distance < 0.75f * knnMatches[i][1].distance)
                        {
                            goodMatches.push_back(knnMatches[i][0]);
                        }
                    }
                }

                // 如果有足够的好匹配，计算单应性矩阵
                if (goodMatches.size() >= 4)
                {
                    vector<Point2f> srcPoints, dstPoints;

                    for (size_t i = 0; i < goodMatches.size(); i++)
                    {
                        srcPoints.push_back(referenceKeypoints[goodMatches[i].queryIdx].pt);
                        dstPoints.push_back(frameKeypoints[goodMatches[i].trainIdx].pt);
                    }

                    // 计算单应性矩阵
                    Mat H = findHomography(srcPoints, dstPoints, RANSAC);

                    if (!H.empty())
                    {
                        // 定义参考图像的角点
                        vector<Point2f> refCorners(4);
                        refCorners[0] = Point2f(0, 0);
                        refCorners[1] = Point2f((float)referenceImage.cols, 0);
                        refCorners[2] = Point2f((float)referenceImage.cols, (float)referenceImage.rows);
                        refCorners[3] = Point2f(0, (float)referenceImage.rows);

                        // 变换角点到当前帧
                        vector<Point2f> transformedCorners(4);
                        perspectiveTransform(refCorners, transformedCorners, H);

                        // 使用 PnP 估计姿态
                        Mat rvec, tvec;
                        estimatePose(transformedCorners, rvec, tvec);

                        // 将旋转向量转换为旋转矩阵
                        Mat rotation_matrix_temp;
                        Rodrigues(rvec, rotation_matrix_temp);
                        rotation_matrix_temp.convertTo(rotation_matrix, CV_32FC1);
                        tvec.convertTo(translation_vector, CV_32FC1);

                        // 计算四元数
                        float R[9];
                        memcpy(R, rotation_matrix.data, rotation_matrix.cols * rotation_matrix.rows * sizeof(float));
                        float quaternion[4];
                        quaternion[0] = sqrtf(1.0f + R[0] + R[4] + R[8]) / 2.0f; // w
                        quaternion[1] = -(R[7] - R[5]) / (4 * quaternion[0]);    // x
                        quaternion[2] = (R[3] - R[1]) / (4 * quaternion[0]);     // y
                        quaternion[3] = -(R[2] - R[6]) / (4 * quaternion[0]);    // z

                        // 获取平移向量
                        float T[3];
                        memcpy(T, translation_vector.data, translation_vector.cols * translation_vector.rows * sizeof(float));

                        // 使用金字塔光流法准备
                        buildOpticalFlowPyramid(grayFrame, prevPyr, Size(21, 21), 3);

                        // 绘制检测到的对象边界
                        for (int i = 0; i < 4; i++)
                        {
                            line(dst, transformedCorners[i], transformedCorners[(i + 1) % 4],
                                 Scalar(0, 255, 0, 255), 4);
                        }

                        // 初始化跟踪
                        prevGray = grayFrame.clone();
                        prevCorners = transformedCorners;

                        // 在感兴趣区域内寻找更多特征点用于跟踪
                        Mat mask = createMask(grayFrame.size(), transformedCorners);
                        prevKeyPoints.clear();
                        goodFeaturesToTrack(prevGray, prevKeyPoints, 80, 0.01, 10, mask);

                        // 切换到跟踪模式
                        isTracking = true;
                        trackFailureCount = 0;

                        // 绘制匹配点（可选）
                        for (size_t i = 0; i < min(10UL, goodMatches.size()); i++)
                        {
                            Point2f pt = frameKeypoints[goodMatches[i].trainIdx].pt;
                            circle(dst, pt, 5, Scalar(255, 0, 0, 255), -1);
                        }
                    }
                }
            }
        }
        else
        {
            // 跟踪模式：使用金字塔光流法
            vector<Mat> currPyr;
            buildOpticalFlowPyramid(grayFrame, currPyr, Size(21, 21), 3);

            vector<Point2f> currKeyPoints;

            // 跟踪模式：使用光流跟踪
            if (!prevKeyPoints.empty() && !prevGray.empty())
            {
                trackStatus.clear();
                trackError.clear();

                // 使用光流法跟踪特征点
                calcOpticalFlowPyrLK(prevPyr, currPyr, prevKeyPoints, currKeyPoints,
                                     trackStatus, trackError);

                // 筛选成功跟踪的点，增加误差阈值筛选
                vector<Point2f> trackedPrevPts, trackedCurrPts;
                for (size_t i = 0; i < trackStatus.size(); i++)
                {
                    if (trackStatus[i] && trackError[i] < MAX_TRACK_ERROR)
                    {
                        trackedPrevPts.push_back(prevKeyPoints[i]);
                        trackedCurrPts.push_back(currKeyPoints[i]);
                    }
                }

                // 如果有足够的跟踪点，计算单应性矩阵
                if (trackedPrevPts.size() >= 4)
                {
                    // 使用 RANSAC 计算单应性矩阵并获取内点掩码
                    vector<uchar> inlierMask;
                    Mat H = findHomography(trackedPrevPts, trackedCurrPts, RANSAC, 3.0, inlierMask);

                    // 计算内点数量
                    int inlierCount = 0;
                    for (auto mask : inlierMask)
                    {
                        if (mask)
                            inlierCount++;
                    }

                    if (!H.empty() && inlierCount >= MIN_INLIER_COUNT)
                    {
                        // 变换上一帧的角点到当前帧
                        currCorners.clear();
                        perspectiveTransform(prevCorners, currCorners, H);

                        // 检查变换后的角点是否合理
                        bool isValidQuad = true;

                        // 检查角点是否在图像范围内（允许部分超出）
                        int outOfBoundsCount = 0;
                        for (int i = 0; i < 4; i++)
                        {
                            if (currCorners[i].x < -50 || currCorners[i].x >= dst.cols + 50 ||
                                currCorners[i].y < -50 || currCorners[i].y >= dst.rows + 50)
                            {
                                outOfBoundsCount++;
                            }
                        }

                        if (outOfBoundsCount > 1)
                        {
                            isValidQuad = false;
                        }

                        // 检查四边形面积是否合理
                        double area = 0;
                        if (isValidQuad)
                        {
                            // 计算四边形面积
                            area = contourArea(currCorners);
                            double minArea = 0.005 * dst.cols * dst.rows; // 最小面积阈值
                            double maxArea = 0.95 * dst.cols * dst.rows;  // 最大面积阈值

                            if (area < minArea || area > maxArea)
                            {
                                isValidQuad = false;
                            }

                            // 检查面积变化是否过大
                            if (prevArea > 0 && fabs(area - prevArea) / prevArea > MAX_AREA_CHANGE_RATIO)
                            {
                                isValidQuad = false;
                            }
                        }

                        if (isValidQuad)
                        {
                            // 绘制跟踪的对象边界
                            for (int i = 0; i < 4; i++)
                            {
                                line(dst, currCorners[i], currCorners[(i + 1) % 4],
                                     Scalar(0, 255, 0, 255), 4);
                            }

                            // 更新跟踪状态
                            prevGray = grayFrame.clone();
                            prevKeyPoints = currKeyPoints;
                            prevCorners = currCorners;
                            prevArea = area;

                            // 每隔一段时间在感兴趣区域内重新寻找特征点，避免点数过少
                            if (trackedPrevPts.size() < 30)
                            {
                                Mat mask = createMask(grayFrame.size(), currCorners);
                                prevKeyPoints.clear();
                                goodFeaturesToTrack(prevGray, prevKeyPoints, 100, 0.01, 10, mask);
                            }

                            // 成功跟踪，减少失败计数
                            trackFailureCount = max(0, trackFailureCount - 1);
                        }
                        else
                        {
                            trackFailureCount++; // 减小失败计数增量
                        }
                    }
                    else
                    {
                        trackFailureCount++; // 减小失败计数增量
                    }
                }
                else
                {
                    trackFailureCount++; // 减小失败计数增量
                }

                // 如果连续跟踪失败次数过多，切换回检测模式
                if (trackFailureCount > MAX_TRACK_FAILURES)
                {
                    // 应该渐进式退出跟踪
                    if (trackFailureCount > MAX_TRACK_FAILURES * 2)
                    {
                        isTracking = false;
                        prevArea = 0;
                    }
                }

                // 绘制跟踪点
                for (size_t i = 0; i < min(10UL, trackedCurrPts.size()); i++)
                {
                    circle(dst, trackedCurrPts[i], 5, Scalar(255, 0, 0, 255), -1);
                }
            }
            else
            {
                // 如果没有可跟踪的点，切换回检测模式
                isTracking = false;
                prevArea = 0; // 重置面积记录
            }
        }

        // 在图像上显示当前模式
        string modeText = isTracking ? "Tracking" : "Detection";
        putText(dst, modeText, Point(10, 60), FONT_HERSHEY_SIMPLEX,
                1.0, Scalar(0, 0, 255, 255), 2);

        // 添加绘制帧率
        drawFPS(dst, fps);
    }
}
