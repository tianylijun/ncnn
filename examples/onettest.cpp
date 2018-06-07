#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include <net.h>
#include <cpu.h>

using namespace std;
using namespace cv;
using namespace ncnn;

int main(int argc, char *argv[])
{
    int ret = 0, i = 1, loopCnt = 1, num_threads = 1, bigcore = 0;
    char *pFname = (char *)"face.jpg";
    char *pParam = (char*)"48net.param";
    char *pBin = (char*)"48net.bin";
    char *pBlob = (char *)"prob1";
    struct timeval beg, end;

    printf("e.g.:  ./demo 1.jpg 48net.param 48net.bin prob1 10 4\n");

    if (argc > 1) pFname = argv[i++];
    if (argc > 2) pParam = argv[i++];
    if (argc > 3) pBin = argv[i++];
    if (argc > 4) pBlob = argv[i++];
    if (argc > 5) loopCnt = atoi(argv[i++]);
    if (argc > 6) num_threads = atoi(argv[i++]);
    if (argc > 7) bigcore = atoi(argv[i++]);
    printf("img: %s param: %s bin: %s blob: %s loopCnt: %d num_threads: %d bigcore: %d\n", pFname, pParam, pBin, pBlob, loopCnt, num_threads, bigcore);

    cv::Mat img = imread(pFname);
    if (img.empty())
    {
        printf("read img failed, %s\n", pFname);
        return -1;
    }
    img.convertTo(img, CV_32F, 1.0 / 128, -127.5/128);
    printf("img c: %d w: %d h : %d step: %lu\n", img.channels(), img.cols, img.rows, img.step[0]);

    Net cnnNet;
    ret = cnnNet.load_param(pParam);
    printf("Load param %d\n", ret);
    ret = cnnNet.load_model(pBin);
    printf("Load model %d\n", ret);

    ncnn::Mat in(img.cols, img.rows, img.channels(), img.data);
    ncnn::Mat out;

    if (bigcore) ncnn::set_cpu_powersave(2);

    Extractor ex = cnnNet.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(num_threads);

    gettimeofday(&beg, NULL);

    for(int loop = 0; loop < loopCnt; loop++)
    {
        ex.input("data", in);
        int ret = ex.extract(pBlob, out);
        printf("[%d] ret: %d dims: %d c: %d h: %d w: %d\n", loop, ret, out.dims, out.c, out.h, out.w);
    }

    gettimeofday(&end, NULL);
    printf("\ntime: %ld ms, avg time : %.3f ms, loop: %d thread:%d bigcore:%d\n\n", (end.tv_sec*1000000 + end.tv_usec - beg.tv_sec*1000000 - beg.tv_usec)/1000, (end.tv_sec*1000000 + end.tv_usec - beg.tv_sec*1000000 - beg.tv_usec)/(1000.0*loopCnt), loopCnt, num_threads, bigcore);
    for (unsigned i = 0; i < out.total(); i++)
    {
        if ((0 != i)&& (0 == i % 16))
            printf("\n");
        printf("%9.6f ", out[i]);
    }
    printf("\n");
    return 0;
}
