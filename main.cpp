#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <sys/time.h>
#include <dirent.h>

using namespace std;
using namespace cv;

int folder_num;
HOGDescriptor Hog;
char filePath[3][200];

Size winSize(112,24), blockSize(16,8),blockStride(16,16),cellSize(8,8),computeSize(8,8);
//Size winSize(112,24), blockSize(24,24),blockStride(22,24),cellSize(12,12),computeSize(12,12);
//function of initializing
void config(){
    int nbin=6;//orientation bins
    folder_num=2;//number of folders containing samples 86_0504pm/300_0428_800_600/240_112*24_800_600
    snprintf(filePath[0],150,"/Users/lan/Desktop/Papers/FirstConf/experiments/crop_samples/pos_samples/totestsamples"); //path of folder containing positive samples
    //snprintf(filePath[0],150,"/Users/lan/Desktop/TarReg/svm/crop_samples/tobecroped/300_0428_800_600/");
    snprintf(filePath[1],150,"/Users/lan/Desktop/Papers/FirstConf/experiments/crop_samples/neg_samples/totestsamples"); //path of folder containing negative samples
    //snprintf(filePath[2],150,"/Users/lan/Desktop/TarReg/svm/svmrobot/hard_samples/"); // path of folder containing hard and negative samples

    HOGDescriptor hog(winSize,blockSize,blockStride,cellSize,nbin);//define(initialize) the parameters of HOG descriptor
    Hog=hog;
}

//继承自CvSVM的类，因为生成setSVMDetector()中用到的检测子参数时，需要用到训练好的SVM的decision_func参数，
//但通过查看CvSVM源码可知decision_func参数是protected类型变量，无法直接访问到，只能继承之后通过函数访问

//get all of the file(path) names, and push them into a vector
void getFiles( string path, vector<string>& files )
{
    DIR  *dir;
    struct dirent  *ptr;
    dir = opendir(path.c_str());
    string pathName;
    while((ptr = readdir(dir)) != NULL){// scan all of the file names
        if(ptr->d_name[0]!='.'){
            files.push_back(pathName.assign(path).append("/").append(string(ptr->d_name)));
        }
    }
}

int main()
{
    config();
    //dimensions of Hog descriptor depends on winSize(112,24), blockSize(16,16),blockStride(16,8),cellSize(8,8)
    int DescriptorDim = Hog.getDescriptorSize();
    //define three vector and push all of files names to the the storage of vectors.
    vector<string> file_name_array[3];
    vector<string> logs;//logs
    logs.push_back(string("reading positive testing,file num:")); //constructing the contents of logs.
    logs.push_back(string("reading negative testing,file num:"));
    logs.push_back(string("reading hard negative samples,file num:"));

    for(int i=0;i<folder_num;++i){
        vector<string> file_names;//the temperate container for preserving file names.
        getFiles(filePath[i], file_names);
        file_name_array[i]=file_names;
        //all the three lines of codes are constructing contents
        char*file_names_size_str=(char*)malloc(20*sizeof(char));
        snprintf(file_names_size_str,19,"%d",file_names.size()); //每个文件夹下的图片数目，把整型转成字符串
        logs[i].append(file_names_size_str); //把上面生成的字符串拼接进对应的日志
    }

    CvSVM svm;//SVM classifier
    cout<<"------loading SVM classifier------"<<endl;
    svm.load("/Users/lan/Desktop/Papers/FirstConf/experiments/training/SVM_HOG_boot.xml");//从XML文件读取训练好的SVM模型
    //Mat类是opencv中表示一个n维的稠密数值型的单通道或多通道数组的类。用于存储实数或复数值的向量和矩阵、灰度或彩色图像、体素、向量场、点云、张量、直方图
    Mat single_sampleFeatureMat= Mat::zeros(1,DescriptorDim, CV_32FC1);;//单个样本的特征向量组成的矩阵，行数等于1，列数等于HOG描述子维数

    //分别依次是正样本判断错误图片数量，判断正确图片数量 successively, the numbers of wrong and correct predications for positive samples.
    int positive_error=0, positive_correct=0;
    //分别依次是负样本判断错误图片数量，判断正确图片数量 successively, the numbers of error and correct predications for negative samples.
    int negative_error=0, negative_correct=0;

    //分别是正样本和负样本预测正确率 the ratio of predication for samples.
    float positive_correct_rate,negative_correct_rate;

    struct timeval tpstart,tpend; //the two struct values about time
    double timeuse; //running time of training function
    gettimeofday(&tpstart,NULL); //get the starting time

    for(int i=0;i<folder_num;++i){

        int predict_result; //记录预测结果
        cout<<logs[i]<<endl; //print logs

        for(int j=0;j<file_name_array[i].size();++j){
            Mat src = imread(file_name_array[i][j]); //load and read image 读取图片
            vector<float> descriptors; //define hog descriptor vector
            Hog.compute(src,descriptors,computeSize); //compute the hog descriptors计算HOG描述子,src=images,computsSize=winStride

            for(int k=0; k<DescriptorDim; k++)
            {
                single_sampleFeatureMat.at<float>(0,k) = descriptors[k]; //the kth element in the descriptor, 样本的特征向量中的第k个元素
            }

            predict_result=svm.predict(single_sampleFeatureMat); //judgement for current sample 判断当前样本

            if(i==0){

                if(predict_result==1) //正确识别正目标
                    positive_correct++;
                else
                {
                    positive_error++; //未识别正目标
                    //cout << "hard_positive_sample : " << file_name_array[0][j] << endl;
                }


            }else{
                if(predict_result==-1){
                    negative_correct++; //正确识别负目标
                }else{
                    negative_error++; //未识别负目标
                    //cout << "hard_negative_sample : " << file_name_array[1][j] << endl;
                }
            }

            src.release(); //释放内存
        }
    }
    gettimeofday(&tpend,NULL); //get the time of end
    timeuse=1000000*(tpend.tv_sec-tpstart.tv_sec)+tpend.tv_usec-tpstart.tv_usec; //calculations for running time.
    timeuse/=1000;
    cout<<"finish detecting samples，time cost："<<timeuse<<"ms"<<endl;

    //依次计算识别正确率和错误率
    cout<<"Number of predicted robot:"<<(float)positive_correct <<endl;
    cout<<"The correct rate of prediction:"<<(float)positive_correct/(positive_correct+positive_error)<<endl;
    cout<<"Number of predicted No-robot:"<<(float)negative_correct <<endl;
    cout<<"the correct rate for of No-prediction:"<<(float)negative_correct/(negative_correct+negative_error)<<endl;

    return 0;
}