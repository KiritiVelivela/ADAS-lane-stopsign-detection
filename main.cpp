#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>

#include <pthread.h>

#include "channel.h"

using namespace cv;
using namespace std;

#define DEBUG 0
#define OUTPUT_VIDEO 1
#define OUTPUT_FRAMES 0
#define RESIZE 0
#define SHOW_ROI 0

// #define LINE_REJECT_SLOPE 0.25
#define LINE_REJECT_SLOPE 0.6

#define GREEN cv::Scalar(0, 255, 0)
#define RED cv::Scalar(0, 0, 255)
#define BLUE cv::Scalar(255, 0, 0)

String stop_sign_cascade_name = "./classifier/cascade.xml";
CascadeClassifier stop_sign_cascade;
Channel<cv::Mat> laneFrameChannel;
Channel<bool> frameProcessed;

// Constants for frame resizing
constexpr int width = 640;
constexpr int height = 480;

// Constants for real-world measurements
constexpr double ym_per_pix = 30.0 / 720; // meters per pixel in y dimension
constexpr double xm_per_pix = 3.7 / 350;  // meters per pixel in x dimension

Point calculateMidpoint(const Point &p1, const Point &p2)
{
    return (p1 + p2) / 2;
}

Mat region_of_interest(const Mat &img, const vector<Point> &vertices)
{
    Mat mask = Mat::zeros(img.size(), CV_8UC1);
    vector<vector<Point>> pts = {vertices};
    fillPoly(mask, pts, Scalar(255));
    Mat masked_img;
    bitwise_and(img, mask, masked_img);
    return masked_img;
}

void processStopSigns(const cv::Mat &frame, const Rect &stop_sign_roi, std::vector<Rect> &stopSigns, Mat &mask_red)
{
    Mat frame_gray, frame_hsv, frame_upper;

    // Convert frame to HSV and create a mask for red areas
    cvtColor(frame, frame_hsv, COLOR_BGR2HSV);
    Mat mask1, mask2;
    inRange(frame_hsv, Scalar(0, 100, 100), Scalar(25, 255, 255), mask1);
    inRange(frame_hsv, Scalar(160, 100, 100), Scalar(180, 255, 255), mask2);
    mask_red = mask1 | mask2;

    Mat frame_red;
    frame.copyTo(frame_red, mask_red);

    frame_upper = frame(stop_sign_roi);

    // Detect stop signs in the defined ROI
    cvtColor(frame_upper, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    stop_sign_cascade.detectMultiScale(frame_gray, stopSigns, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(30, 30), Size(60, 48));
}

void processLanes(const cv::Point &mid_roi, const vector<Point> &roi_vertices, double &right_slope,
                  double &left_slope,
                  Point &right_point,
                  Point &left_point, double &offset)
{
    try
    {
        while (true)
        {
            Mat frame = laneFrameChannel.receive();

            if (frame.empty())
            {
                break;
            }

            Mat gray, blur, edges;
            cvtColor(frame, gray, COLOR_BGR2GRAY);
            GaussianBlur(gray, blur, Size(5, 5), 0);
            Canny(blur, edges, 50, 150);

            Mat roi_edges = region_of_interest(edges, roi_vertices);

            vector<Vec4i> lines;
            HoughLinesP(roi_edges, lines, 1, CV_PI / 180, 50, 50, 100);

            // classify lines to left/right side
            std::vector<Point> left_points, right_points;
            bool leftDetected = false, rightDetected = false;

            for (int i = 0; i < lines.size(); i++)
            {
                cv::Vec4i line = lines[i];
                float dx = line[2] - line[0];
                float dy = line[3] - line[1];
                double slope = dy / dx;

                if (fabs(slope) <= LINE_REJECT_SLOPE)
                { // reject near horizontal lines
                    continue;
                }

                int midx = (line[0] + line[2]) / 2;
                if (slope < 0 && midx < frame.cols / 2)
                {
                    auto p1 = cv::Point(line[0], line[1]);
                    auto p2 = cv::Point(line[2], line[3]);
                    left_points.push_back(p1);
                    left_points.push_back(p2);
                    leftDetected = true;

#if DEBUG
                    cv::line(frame, p1, p2, RED, 2, cv::LINE_AA);
#endif
                }
                else if (slope > 0 && midx > frame.cols / 2)
                {
                    auto p1 = cv::Point(line[0], line[1]);
                    auto p2 = cv::Point(line[2], line[3]);
                    right_points.push_back(p1);
                    right_points.push_back(p2);
                    rightDetected = true;

#if DEBUG
                    cv::line(frame, p1, p2, BLUE, 2, cv::LINE_AA);
                    imshow("lane frame", frame);
#endif
                }

                Vec4d right_line, left_line;

                if (rightDetected)
                {
                    // fit a line
                    fitLine(right_points, right_line, DIST_L2, 0, 0.01, 0.01);
                    right_slope = right_line[1] / right_line[0];
                    right_point = Point(right_line[2], right_line[3]);
                }

                if (leftDetected)
                {
                    // fit a line
                    fitLine(left_points, left_line, DIST_L2, 0, 0.01, 0.01);
                    left_slope = left_line[1] / left_line[0];
                    left_point = Point(left_line[2], left_line[3]);
                }
            }

            // Compute offset from lane center
            double lane_center = (left_point.x + right_point.x) / 2.0;
            double frame_center = frame.cols / 2.0;
            offset = (frame_center - lane_center) * xm_per_pix;
            frameProcessed.send(true);
        }
    }
    catch (const ChannelClosedException &e)
    {
        return;
    }
}

void frameLoader(const std::string &videoName, BufferedChannel<cv::Mat> &frameLoaderChannel)
{
    VideoCapture cap = VideoCapture(videoName);

    if (!cap.isOpened())
    {
        cerr << "Error: Unable to open video file!" << endl;
        return;
    }

    while (true)
    {
        Mat frame;
        cap.read(frame);

        if (frame.empty())
        {
            frameLoaderChannel.close();
            break;
        }

#if RESIZE
        cv::resize(frame, frame, Size(width, height), 0, 0, cv::INTER_LINEAR);
#endif

        frameLoaderChannel.send(std::move(frame));
    }
}

inline void set_core_affinity(pthread_t tid, int core_id)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    int rc = pthread_setaffinity_np(tid,
                                    sizeof(cpu_set_t), &cpuset);

    if (rc != 0)
    {
        std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
    }
}

int main(int argc, char *argv[])
{

    if (argc != 2)
    {
        cout << "Usage: ./main <video_name>" << endl;
        return -1;
    }

    std::string videoName = argv[1];

    BufferedChannel<cv::Mat> frameLoaderChannel(1000);

    if (!stop_sign_cascade.load(stop_sign_cascade_name))
    {
        cout << "Error: loading trained cascade\n";
        return -1;
    }

#if OUTPUT_VIDEO
    VideoWriter outputVideo("output_" + videoName, VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, Size(width, height));
#endif

    int frame_count = 1;
    double start_time = (double)getTickCount();

    // trapezoidal ROI
    vector<Point> roi_vertices = {Point(175, 375), Point(280, 280), Point(350, 280), Point(450, 375)};

    // ROI MIDPOINT Trapezoidal ROI
    auto roi_mid_1 = calculateMidpoint(roi_vertices[1], roi_vertices[2]);
    auto roi_mid_2 = calculateMidpoint(roi_vertices[0], roi_vertices[3]);

    cv::Point roi_mid = calculateMidpoint(roi_mid_1, roi_mid_2);

    // intialize slopes to 1 to preven divide by zero
    double right_slope = 1;
    double left_slope = 1;
    Point right_point;
    Point left_point;

    double y1_right = 400;
    double y1_left = 400;
    double y2 = 300;
    double offset = 0;

    // Define ROI to ignore bottom 30% of the frame
    Rect stop_sign_roi(0, 0, width, height * 0.7); // 70% from the top

    // std::thread frameLoaderThread(frameLoader, "./chico-clipped.mp4", std::ref(frameLoaderChannel));
    // std::thread frameLoaderThread(frameLoader, "./GOPRO638-flipped-clipped.mp4", std::ref(frameLoaderChannel));

    std::thread frameLoaderThread(frameLoader, videoName, std::ref(frameLoaderChannel));
    set_core_affinity(frameLoaderThread.native_handle(), 2);

    std::thread laneProcessorThread;

    laneProcessorThread = std::thread(processLanes, roi_mid, roi_vertices, std::ref(right_slope), std::ref(left_slope), std::ref(right_point), std::ref(left_point), std::ref(offset));
    set_core_affinity(laneProcessorThread.native_handle(), 3);

    Mat frame;
    while (true)
    {
        try
        {
            frame = frameLoaderChannel.receive();
        }
        catch (const ChannelClosedException &e)
        {
            laneFrameChannel.close();
            break;
        }

        auto frame_copy = frame.clone();

        laneFrameChannel.send(std::move(frame_copy));

        std::vector<Rect> stopSigns;

        Mat mask_red;

        processStopSigns(frame, stop_sign_roi, stopSigns, mask_red);

        for (size_t i = 0; i < stopSigns.size(); i++)
        {
            Rect rect = stopSigns[i];

            // Check the amount of red within each detected region
            Mat detected_area = mask_red(rect);
            double redness = (double)countNonZero(detected_area) / (rect.width * rect.height);
            if (redness < 0.0015)
            {             // Threshold for redness, can be adjusted
                continue; // Skip detections that are not red enough
            }
            rectangle(frame, Point(rect.x, rect.y + stop_sign_roi.y), Point(rect.x + rect.width, rect.y + rect.height + stop_sign_roi.y), Scalar(0, 255, 0), 2);

            // add redness as text at the rectange top left vertex
            putText(frame, "redness: " + to_string(redness), Point(rect.x, rect.y + stop_sign_roi.y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);
        }

#if SHOW_ROI
        // mark roi points
        for (int i = 0; i < roi_vertices.size(); i++)
        {
            cv::circle(frame, roi_vertices[i], 2, cv::Scalar(0, 255, 0), 2);
        }
#endif

        frameProcessed.receive();

        double right_lane_x1 = ((y1_right - right_point.y) / right_slope) + right_point.x;
        double right_lane_x2 = ((y2 - right_point.y) / right_slope) + right_point.x;

        double left_lane_x1 = ((y1_left - left_point.y) / left_slope) + left_point.x;
        double left_lane_x2 = ((y2 - left_point.y) / left_slope) + left_point.x;

        auto point1 = Point(right_lane_x1, y1_right);
        auto point2 = Point(right_lane_x2, y2);
        auto point3 = Point(left_lane_x1, y1_left);
        auto point4 = Point(left_lane_x2, y2);

        vector<Point> fill_vertices = {point1, point2, point4, point3};

        line(frame, point1, point2, BLUE, 5, cv::LINE_AA);
        line(frame, point3, point4, RED, 5, cv::LINE_AA);

        fillConvexPoly(frame, fill_vertices, GREEN, cv::LINE_AA);

        auto curr_time = (double)getTickCount();
        double fps = (double(frame_count) * getTickFrequency() / double(curr_time - start_time));

        cv::putText(frame, std::to_string(fps), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);

        // Display offset from lane center
        cv::putText(frame, "Offset: " + std::to_string(offset) + "m", cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);

#if OUTPUT_VIDEO
        outputVideo.write(frame);
#endif

#if OUTPUT_FRAMES
        imwrite("frames/frame_" + to_string(frame_count) + ".jpg", frame);
#endif

        imshow("out", frame);

        frame_count++;

        waitKey(1);
    }

    frameLoaderThread.join();
    laneProcessorThread.join();

#if OUTPUT_VIDEO
    outputVideo.release();
#endif

    return 0;
}