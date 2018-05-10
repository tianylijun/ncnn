#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include <net.h>

using namespace std;
using namespace cv;
using namespace ncnn;

int main(int argc, char *argv[]) {
	int i = 1, loopCnt = 1, num_threads = 1;
	char *pFname = (char *)"face.jpg";
	char *pParam = (char*)"regNet.param";
	char *pBin = (char*)"regNet.bin";
	char *pBlob = (char *)"mobilenet_v2_layer9_conv1x1_mobilenet_v2_layer9_conv1x1_scale";
	struct timeval beg, end;

	printf("e.g.:  ./demo 1.jpg 10\n");

	if (argc > 1) pFname = argv[i++];
	if (argc > 2) loopCnt = atoi(argv[i++]);
	if (argc > 3) num_threads = atoi(argv[i++]);
	if (argc > 4) pParam = argv[i++];
	if (argc > 5) pBin = argv[i++];
	if (argc > 6) pBlob = argv[i++];

	printf("img: %s loopCnt: %d num_threads: %d, param: %s bin: %s blob: %s\n", pFname, loopCnt, num_threads, pParam, pBin, pBlob);

	cv::Mat img = imread(pFname);
	if (img.empty())
	{
		printf("read img failed, %s\n", pFname);
		return -1;
	}

	printf("img w: %d h : %d step: %u\n", img.cols, img.rows, img.step[0]);

	Net regNet;
	regNet.load_param(pParam);
	regNet.load_model(pBin);

	ncnn::Mat in(img.cols, img.rows, 3, img.data);
	ncnn::Mat out;

	Extractor ex = regNet.create_extractor();
	ex.set_light_mode(true);
	ex.set_num_threads(num_threads);
	gettimeofday(&beg, NULL);

	for(int loop = 0; loop < loopCnt; loop++)
	{
		ex.input("data", in);
		int ret = ex.extract(pBlob, out);
		printf("[%d] ret: %d dims: %d c: %d h: %d w: %d\n", loop, ret, out.dims, out.c, out.h,out.w);
	}

	gettimeofday(&end, NULL);
	printf("\ntime: %ld ms, avg time : %ld ms, loop: %d\n\n", (end.tv_sec*1000000 + end.tv_usec - beg.tv_sec*1000000 - beg.tv_usec)/1000, (end.tv_sec*1000000 + end.tv_usec - beg.tv_sec*1000000 - beg.tv_usec)/(1000*loopCnt), loopCnt);
	return 0;
}
