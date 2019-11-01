# 人脸识别（验证登录）

使用dlib进行人脸识别,然后使用resnet预训练模型进行特征提取，最后对所提取到的特征进行比较，当两者距离较近的时候检测成功。

Dlib was used for face recognition, and VGG pre-training model was used for feature extraction. Finally, the extracted features were compared, and the detection was successful when the two were close to each other.

### 文件结构

desc_file.npy 人脸描述信息

candidate_file.npy 名字信息

dlib_face_recognition_resnet_model_v1.dat 预训练模型

main.py 主函数

shape_predictor_68_face_landmarks.dat 提取人脸特征模型

