// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <algorithm>
#include <windows.h>
#include <queue>

using namespace cv;
using namespace std;

#undef max
#undef min

wchar_t* projectPath;

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	_wchdir(projectPath);

	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testNegativeImageFast()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// The fastest approach of accessing the pixels -> using pointers
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Accessing individual pixels in a RGB 24 bits/pixel image
		// Inefficient way -> slow
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// HSV components
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// Defining pointers to each matrix (8 bits/pixels) of the individual components H, S, V 
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// Defining the pointer to the HSV image matrix (24 bits/pixel)
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;	// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	_wchdir(projectPath);

	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(100);  // waits 100ms and advances to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	_wchdir(projectPath);

	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snap the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / Image Processing)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

double clamp(double x, double low, double high) {
    if (x < low) return low;
    if (x > high) return high;
    return x;
}

int clamp(int x, int low, int high) {
    if (x < low) return low;
    if (x > high) return high;
    return x;
}

Mat gaussianFilter(const Mat& S) {
    //kernel gaussian H(u,v)
    Mat H = (Mat_<float>(5, 5) <<
        0.0005f, 0.0050f, 0.0109f, 0.0050f, 0.0005f,
        0.0050f, 0.0521f, 0.1139f, 0.0521f, 0.0050f,
        0.0109f, 0.1139f, 0.2487f, 0.1139f, 0.0109f,
        0.0050f, 0.0521f, 0.1139f, 0.0521f, 0.0050f,
        0.0005f, 0.0050f, 0.0109f, 0.0050f, 0.0005f
        );
    const int w = H.rows;
    const int k = (w - 1) / 2;

    float c = sum(H)[0];//factor de normalizare 

    std::vector<Mat> channels;
    split(S, channels);

    std::vector<Mat> out(3);
    for (int c = 0; c < 3; c++) {
        out[c] = Mat::zeros(S.rows, S.cols, CV_8UC1);
    }

    //manual convolution 
    for (int ch = 0; ch < 3; ch++) {
        for (int i = 0; i < S.rows; i++) {
            for (int j = 0; j < S.cols; j++) {
                float suma = 0.0f;

                for (int u = 0; u < w; u++) {
                    for (int v = 0; v < w; v++) {
                        int iu = clamp(i + u - k, 0, S.rows - 1);
                        int jv = clamp(j + v - k, 0, S.cols - 1);

                        suma += H.at<float>(u, v) * channels[ch].at<uchar>(iu, jv);
                    }
                }
                // normalizare 
                out[ch].at<uchar>(i, j) = saturate_cast<uchar>(suma / c);
            }
        }
    }
    Mat D;
    merge(out, D);
    return D;
}


void computeSTDdevUV(const Mat& luv, double& ch1_std, double& ch2_std) {
    vector<Mat> ch(3);
    split(luv, ch);
    Mat uF, vF;
    ch[1].convertTo(uF, CV_32F);
    ch[2].convertTo(vF, CV_32F);
    Scalar mu, su, mv, sv;
    meanStdDev(uF, mu, su);
    meanStdDev(vF, mv, sv);
    ch1_std = su[0];
    ch2_std = sv[0];
}

void regionGrowing(const Mat& luv, Mat& labels, double ch1_std, double ch2_std, double scale, int window_size) {
    int rows = luv.rows, cols = luv.cols;
    labels = Mat::zeros(rows, cols, CV_32SC1);
    vector<Mat> ch(3);
    split(luv, ch);
    Mat uCh = ch[1], vCh = ch[2];
    double T = scale * std::sqrt(ch1_std * ch1_std + ch2_std * ch2_std) ;
    printf("Threshold T=%.2f\n", T);
    int dx[8] = { -1,-1,-1, 0, 0, 1, 1, 1 };
    int dy[8] = { -1, 0, 1,-1, 1,-1, 0, 1 };
    int currentLabel = 1;
    int k2 = window_size / 2;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (labels.at<int>(i, j) != 0)
                continue;
            //seed 
            queue<Point> Q;
            Q.push(Point(j, i));
            labels.at<int>(i, j) = currentLabel;
            double sumU = 0, sumV = 0;
            int regionSize = 0;
           // media inițială doar pentru pixeli neetichetați
            for (int dy0 = -k2; dy0 <= k2; ++dy0) {
                for (int dx0 = -k2; dx0 <= k2; ++dx0) {
                    int y = clamp(i + dy0, 0, rows - 1);
                    int x = clamp(j + dx0, 0, cols - 1);
                    if (labels.at<int>(y, x) == 0) {
                        sumU += uCh.at<uchar>(y, x);
                        sumV += vCh.at<uchar>(y, x);
                        ++regionSize;
                    }
                }
            }
            double avgU = regionSize > 0 ? sumU / regionSize : uCh.at<uchar>(i, j);
            double avgV = regionSize > 0 ? sumV / regionSize : vCh.at<uchar>(i, j);
            int acceptedPixels = 1;
            // Păstrăm media fixă pe parcursul creșterii regiunii
            while (!Q.empty()) {
                Point p = Q.front(); Q.pop();
                for (int dir = 0; dir < 8; ++dir) {
                    int ny = p.y + dy[dir];
                    int nx = p.x + dx[dir];
                    if (ny < 0 || nx < 0 || ny >= rows || nx >= cols) continue;
                    if (labels.at<int>(ny, nx) != 0) continue;
                    double du = uCh.at<uchar>(ny, nx) - avgU;
                    double dv = vCh.at<uchar>(ny, nx) - avgV;
                    // euclid
                    double dist = std::sqrt(du * du + dv * dv);
                    bool accept = (dist < T);
                    if (accept) {
                        labels.at<int>(ny, nx) = currentLabel;
                        Q.push(Point(nx, ny));
                        avgU = (avgU * acceptedPixels + uCh.at<uchar>(ny, nx)) / (acceptedPixels + 1);
                        avgV = (avgV * acceptedPixels + vCh.at<uchar>(ny, nx)) / (acceptedPixels + 1);
                        acceptedPixels++;
                    }
                }
            }
            printf("Label %d: Accepted %d pixels\n", currentLabel, acceptedPixels);
            ++currentLabel;
        }
    }
    printf("Total labels: %d\n", currentLabel - 1);
}

Mat visualizeLabels(const Mat& labels) {
    int rows = labels.rows, cols = labels.cols;
    Mat labelVis(rows, cols, CV_8UC3, Scalar(0, 0, 0));
    double max_label;
    minMaxLoc(labels, nullptr, &max_label);
    vector<Vec3b> colors(static_cast<int>(max_label) + 1);
    colors[0] = Vec3b(0, 0, 0);
    for (int i = 1; i <= static_cast<int>(max_label); i++) {
        colors[i] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int label = labels.at<int>(i, j);
            labelVis.at<Vec3b>(i, j) = colors[label];
        }
    }
    return labelVis;
}

static Mat makeSegmented(const Mat& img, const Mat& labels) {
    int rows = img.rows, cols = img.cols;
    std::map<int, Vec3d> sum;
    std::map<int, int> cnt;
    // Calculăm media culorilor în spațiul L*u*v*
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            int l = labels.at<int>(y, x);
            Vec3b pix = img.at<Vec3b>(y, x);
            sum[l][0] += pix[0]; // L
            sum[l][1] += pix[1]; // u
            sum[l][2] += pix[2]; // v
            cnt[l]++;
        }
    }
    std::map<int, Vec3b> mean;
    for (auto& kv : sum) {
        int l = kv.first;
        mean[l] = Vec3b(
            uchar(kv.second[0] / cnt[l]),
            uchar(kv.second[1] / cnt[l]),
            uchar(kv.second[2] / cnt[l])
        );
    }
    double min_label, max_label;
    minMaxLoc(labels, &min_label, &max_label);
    printf("Labels range: min=%d, max=%d\n", (int)min_label, (int)max_label);
    for (int l = (int)min_label; l <= (int)max_label; ++l) {
        if (sum.find(l) == sum.end()) {
            printf("Warning: Label %d not found in sum map\n", l);
            mean[l] = Vec3b(0, 0, 0);
        }
    }
    Mat seg(rows, cols, CV_8UC3);
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            int l = labels.at<int>(y, x);
            seg.at<Vec3b>(y, x) = mean[l];
        }
    }
    return seg;
}

static Mat postProcessLuv(const Mat& segluv, Mat& labels) {
    if (segluv.empty()) {
        printf("Error: segluv is empty\n");
        return Mat();
    }
    if (labels.empty()) {
        printf("Error: labels is empty\n");
        return Mat();
    }
    if (segluv.rows != labels.rows || segluv.cols != labels.cols) {
        printf("Error: segluv and labels dimensions do not match\n");
        return Mat();
    }
    printf("Starting postProcessLuv: segluv=%dx%d, labels=%dx%d\n", segluv.cols, segluv.rows, labels.cols, labels.rows);
    Mat tmp = segluv.clone();
    Mat tmp_labels = labels.clone();
    // Element structural 3x3 (8 vecinătăți)
    Mat B = getStructuringElement(MORPH_RECT, Size(3, 3));
    // 3 op de eroziune
    for (int iter = 0; iter < 3; iter++) {
        Mat new_tmp = tmp.clone();
        Mat new_labels = tmp_labels.clone();
        for (int i = 0; i < tmp.rows; i++) {
            for (int j = 0; j < tmp.cols; j++) {
                if (tmp_labels.at<int>(i, j) != 0) {
                    int current_label = tmp_labels.at<int>(i, j);
                    int different_neighbors = 0;
                    int total_neighbors = 0;
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            if (dy == 0 && dx == 0) continue;
                            int ni = i + dy, nj = j + dx;
                            if (ni >= 0 && ni < tmp.rows && nj >= 0 && nj < tmp.cols) {
                                total_neighbors++;
                                if (tmp_labels.at<int>(ni, nj) != current_label && tmp_labels.at<int>(ni, nj) != 0) {
                                    different_neighbors++;
                                }
                            }
                        }
                    }
                    if (different_neighbors > 0) {
                        new_labels.at<int>(i, j) = 0;
                    }
                }
            }
        }
        new_tmp.copyTo(tmp);
        new_labels.copyTo(tmp_labels);
    }
    // Dilatare până când nu mai sunt pixeli neetichetați
    bool has_unlabeled;
    do {
        has_unlabeled = false;
        Mat new_tmp = tmp.clone();
        Mat new_labels = tmp_labels.clone();
        for (int i = 0; i < tmp.rows; i++) {
            for (int j = 0; j < tmp.cols; j++) {
                if (tmp_labels.at<int>(i, j) != 0) {
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            int ni = i + dy, nj = j + dx;
                            if (ni >= 0 && ni < tmp.rows && nj >= 0 && nj < tmp.cols) {
                                if (tmp_labels.at<int>(ni, nj) == 0) {
                                    new_labels.at<int>(ni, nj) = tmp_labels.at<int>(i, j);
                                    new_tmp.at<Vec3b>(ni, nj) = tmp.at<Vec3b>(i, j);
                                }
                            }
                        }
                    }
                }
                else {
                    bool all_neighbors_unlabeled = true;
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            int ni = i + dy, nj = j + dx;
                            if (ni >= 0 && ni < tmp.rows && nj >= 0 && nj < tmp.cols) {
                                if (tmp_labels.at<int>(ni, nj) != 0) {
                                    all_neighbors_unlabeled = false;
                                    break;
                                }
                            }
                        }
                        if (!all_neighbors_unlabeled) break;
                    }
                    if (all_neighbors_unlabeled) {
                        has_unlabeled = true;
                    }
                }
            }
        }
        new_tmp.copyTo(tmp);
        new_labels.copyTo(tmp_labels);
    } while (has_unlabeled);
    tmp_labels.copyTo(labels);
    return tmp;
}

void testRegionGrowing() {
    char fname[MAX_PATH];
    while (openFileDlg(fname)) {
        printf("Trying to load image: %s\n", fname);
        Mat src = imread(fname, IMREAD_COLOR);
        if (src.empty()) {
            printf("Could not load image: %s\n", fname);
            continue;
        }
        printf("Image loaded successfully: %d x %d\n", src.cols, src.rows);

        Mat blurred = gaussianFilter(src);
        printf("Image blurred successfully\n");

        Mat luv;
        cvtColor(blurred, luv, COLOR_BGR2Luv);
        printf("BGR -> Luv : DONE\n");

        // Calcul STD pentru canalele u și v
        vector<Mat> luvCh(3);
        split(luv, luvCh);
        double std_u, std_v;
        computeSTDdevUV(luv, std_u, std_v);
        printf("Computed STD: u=%.2f, v=%.2f\n", std_u, std_v);

        // Introducere factor value
        double scale;
        printf("Factor (val>0)= ");
        if (scanf("%lf", &scale) != 1 || scale <= 0) {
            printf("Invalid input for scale, using default value 1.0\n");
            scale = 1.0;
            while (getchar() != '\n');
        }
        scale = clamp(scale, 0.1, 3.0); // Limităm valoarea pentru siguranță
        printf("Scale: %.2f\n", scale);

        // Segmentare folosind region growing
        Mat labels;
        regionGrowing(luv, labels, std_u, std_v, scale, 3); // w=3
        printf("Region growing completed\n");

        // Vizualizare etichete
        Mat labelVis = visualizeLabels(labels);
        printf("Labels visualized\n");

        // Creare imagine segmentată
        Mat seg = makeSegmented(luv, labels);
        printf("Segmented image created\n");

        // Conversie înapoi în BGR pentru afișare
        Mat segBgr;
        cvtColor(seg, segBgr, COLOR_Luv2BGR);
        printf("Segmented image converted to BGR\n");

        // Postprocesare
        Mat segClean = postProcessLuv(seg, labels);
        if (segClean.empty()) {
            printf("Error: segClean is empty after postProcessLuv\n");
            continue;
        }
        printf("Post-processing completed\n");

        // Conversie înapoi în BGR pentru afișare
        Mat finalBgr;
        cvtColor(segClean, finalBgr, COLOR_Luv2BGR);
        printf("Final image converted to BGR\n");


        // Afișare imagini
        imshow("Source", src);
        imshow("Filtered", blurred);
        imshow("Labeled", segBgr);
        imshow("Final", finalBgr);
        printf("Images displayed, waiting for key press...\n");
        waitKey(0);
        printf("Key pressed, closing windows\n");
    }
}

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);
    printf("OpenCV version: %s\n", CV_VERSION);
    int op;
    do {
        system("cls");
        destroyAllWindows();
        printf("Menu:\n");
        printf(" 1 - Open image\n");
        printf(" 2 - Open BMP images from folder\n");
        printf(" 3 - Image negative\n");
        printf(" 4 - Image negative (fast)\n");
        printf(" 5 - BGR->Gray\n");
        printf(" 6 - BGR->Gray (fast, save result to disk)\n");
        printf(" 7 - BGR->HSV\n");
        printf(" 8 - Resize image\n");
        printf(" 9 - Canny edge detection\n");
        printf(" 10 - Edges in a video sequence\n");
        printf(" 11 - Snap frame from live video\n");
        printf(" 12 - Mouse callback demo\n");
        printf(" 13 - Region Growing\n");
        printf(" 0 - Exit\n\n");
        printf("Option: ");
        scanf("%d", &op);
        switch (op) {
        case 1: testOpenImage(); break;
        case 2: testOpenImagesFld(); break;
        case 3: testNegativeImage(); break;
        case 4: testNegativeImageFast(); break;
        case 5: testColor2Gray(); break;
        case 6: testImageOpenAndSave(); break;
        case 7: testBGR2HSV(); break;
        case 8: testResize(); break;
        case 9: testCanny(); break;
        case 10: testVideoSequence(); break;
        case 11: testSnap(); break;
        case 12: testMouseClick(); break;
        case 13: testRegionGrowing(); break;
        }
    } while (op != 0);
    return 0;
}