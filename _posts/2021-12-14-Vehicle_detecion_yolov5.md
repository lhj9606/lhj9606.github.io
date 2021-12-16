---

title: YOLOv5를 이용한 Vehicle Detection & Validation with Video Game Graphics (AI-X:Deep-Learning)
author: 
  name: Hong Jun Lee
  link: https://github.com/lhj9606
date: 2021-12-03 20:55:00 +0900
categories: [Blogging, AI+X]
tags: [AI, Deep Learning, Yolov5, Vehicle Detection]
pin: true
---

---

<center><b> Hanyang University 2021 Fall / AI+X:Deep Learning / Final Project </b></center>

---
<br>
## Member

|  이름  |    학과    |    학번    |        E-mail         | 역할 |
| :----: | :--------: | :--------: | :-------------------: | :----:|
| 이홍준 | 기계공학부 | 2015011203 | lhj9606@hanyang.ac.kr | Everything |


---
<br>

## 0. Video Summary



YOUTUBE EMBEDDED / LINK INSERT



<br>


---
## 1. Proposal (Option A)

 4차 산업혁명 시대가 대두되면서, 여러 기술들이 발전하고 주목받고 있지만 그 중에서도 우리 생활에 밀접하게 관련있고, 접근성이 좋은 기술 중장 주목받는 기술 중 하나를 꼽자면 바로 '**자율주행**' 기술일 것이다. 이러한 자율주행 기술이 급격하게 발전하게 된 이유 중 하나는 Deep Learning(Deep Neural Network)의 발전, 그리고 CNN(Convolutional Neural Network) 등의 등장으로 인한 컴퓨터 비전(CV, Computer Vision) 분야의 많은 기술적 도약이 있었기에 가능해졌다.



<center><iframe title="vimeo-player" src="https://player.vimeo.com/video/192179726?h=849b6a5ec9" width="640" height="360" frameborder="0" allowfullscreen></iframe></center>

<center><b>Tesla의 자율주행 예시 영상</b></center>

---
### 1.1. Need

자율주행 자동차는 '인지', '판단', '제어'의 3가지 Step으로 스스로 주행을 관장하는데, '인지' 단계에서 ***카메라(Camera), 레이더(Rader), 라이다(LiDAR)*** 등의 여러 센서로 부터 취합한 정보를 바탕으로 차량 주변의 환경을 인식하며, 이를 토대로 '판단'과 '제어'를 수행한다. 즉 **'인지'** 단계가 자율주행 알고리즘에 있어 가장 첫번째 단계이므로, 자율주행에서 가장 중요하면서도 차량의 주변 환경을 명확하게 인식해야만 안전한 주행이 가능하기 때문에 매우 중요한 단계라고 할 수 있다.

이러한 **'인지'** 수행 과정에서 카메라, 레이더, 라이다 등의 센서 정보들 모두 중요하며, 각각의 장단점으로 인해 상호보완적으로 사용되지만, 물체의 종류를 식별하는 데에 가장 중요한 정보는 바로 ***'카메라 데이터'*** 이다. 

위의 [Tesla의 자율주행 예시 영상] 내의 우측의 카메라 영상 처리 결과를 보면 알 수 있다시피 카메라 데이터를 차량의 자율주행의 정보로 활용하는 모습을 볼 수 있다.

본 프로젝트에서는 '인지' 단계에서의 자율주행 차량이 어떻게 주변 사물을 인식하는지 알아보기 위해 '카메라 데이터'를 입력으로 받고 이를 검출해내는 모델(YOLOv5 사용)을 만들어 볼 것이다.



!["CNN을 활용한 Vehicle Detection의 사례 (a) Faster R-CNN, (b) DP-SSD300, (c) SSD300, (d) DP-SSD300"](https://www.mdpi.com/sensors/sensors-19-00594/article_deploy/html/images/sensors-19-00594-g001.png)

<center><b>CNN을 활용한 Vehicle Detection의 사례 (a) Faster R-CNN, (b) DP-SSD300, (c) SSD300, (d) DP-SSD300</b></center>


------
### 1.2. Objective

 본 프로젝트는 YOLOv5를 기반으로 하여, 자율주행 환경에 필요한 이미지들을 모델에 학습하고, 자율주행 환경에 필요한 이미지 검출 속도에 맞게 모델을 튜닝할 예정이다. 

 학습된 모델을 토대로 실제 차량에 탑재하여, 실제 차량과 그 이외의 주행 상황에 등장 가능한 물체에 대한 인식을 검출하는 시도를 하면 좋겠지만, 실제 차량에 컴퓨터와 카메라를 탑재하여 이를 검증해보는 것이 어렵다. 따라서, 본 프로젝트에서는 자율주행 환경을 모사 가능한 일부 게임 (ex. *GTA5, Forza Horizon* 등)의 화면 데이터를 카메라 데이터로 대응시켜 모델의 객체 검출이 잘 이뤄지는지 확인해볼 예정이다.

<br>


---
## 2. Datasets

자율주행을 위한 딥러닝 학습용 Open Dataset은 다양하게 공개가 되어있다. 



### 2.1. KITTI Dataset

 [http://www.cvlibs.net/datasets/kitti/](http://www.cvlibs.net/datasets/kitti/)

<center><iframe width="644" height="364" src="https://www.youtube.com/embed/KXpZ6B1YB_k" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></center>

<br>

 KITTI Dataset은 현존하는 자율주행 Dataset 중에서 가장 많이 사용되는 Dataset으로 자율주행의 다양한 인지 Task를 위한 라벨링(Annotation)을 제공한다. 2D/3D 물체 검출(Object Detection)은 물론 물체 추적(Tracking), 거리 추정, Odometry, 스테레오 비전, 영역 분할 등의 Task에 활용될 수 있다. 또한 알고리즘 벤치마크를 위한 리더보드도 제공한다.



<img src="http://www.cvlibs.net/datasets/kitti/images/setup_top_view.png" style="zoom: 67%;" />

<img src="http://www.cvlibs.net/datasets/kitti/images/passat_sensors_920.png" style="zoom:67%;" />

<center><b>KITTI Dataset 구축을 위해 사용되었던 Vehicle과 Sensor Setup</b></center>

 <br>

KITTI Dataset은 개조된 Volkswagen Passat B6을 이용하여 데이터를 수집하였으며, 데이터의 처리 및 저장은 Intel i7 CPU와 Linux OS를 이용하였다.

데이터 수집을 위해 개조된 차량의 센서 구성은 다음과 같다.

- 1 Inertial Navigation System (GPS/IMU): [OXTS RT 3003](http://www.oxts.com/default.asp?pageRef=21)
- 1 Laserscanner: [Velodyne HDL-64E](http://velodynelidar.com/lidar/hdlproducts/hdl64e.aspx)
- 2 Grayscale cameras, 1.4 Megapixels: [Point Grey Flea 2 (FL2-14S3M-C)](http://www.ptgrey.com/products/flea2/flea2_firewire_camera.asp)
- 2 Color cameras, 1.4 Megapixels: [Point Grey Flea 2 (FL2-14S3C-C)](http://www.ptgrey.com/products/flea2/flea2_firewire_camera.asp)
- 4 Varifocal lenses, 4-8 mm: [Edmund Optics NT59-917](http://www.edmundoptics.com/imaging/imaging-lenses/zoom-lenses/varifocal-imaging-lenses/1620)



센서 구성에서 알 수 있다시피, KITTI Dataset은 단순히 카메라 데이터만을 포함하고 있지는 않다. GPS, IMU 정보와 LiDAR Pointcloud 정보도 내장하고 있다. 그렇지만 본 프로젝트에서는 카메라 데이터만 이용할 예정이다.

다만 신호등에 대한 라벨링 정보가 없어 신호등에 대한 학습이 제한된다는 단점이 있다.

<center><u><span style="color:blue">본 프로젝트에서는 프로젝트 진행 환경(HW, SW)을 고려하여 자율주행 인지에 가장 자주 쓰이는 KITTI Dataset을 이용하여 학습을 진행한다.</span></u></center>

<center><img src="\images\KITTI_instruction.PNG" style="zoom:80%;" /></center>

<br>

KITTI 홈페이지에 간단한 가입을 마친 후에서 'Object' 탭 내의 '*left color images of object data set*'과 '*training labels of object data set*' 을 다운로드 받는다.

기본적인 Dataset 구성은 다음과 같다.

* **Training Image** : 7,481 images (Resolution : 1242 * 375) / 7,481 labels
* **Testing image** : 7,518 images -> label이 없으므로 사용하지 않을 예정
* **Classification** : ['Cyclist', 'DontCare', 'Misc', 'Person_sitting', 'Tram', 'Truck', 'Van', 'car', 'person'] - 9개 분류

<br>

---

### 2.2. nuScenes Dataset
[https://www.nuscenes.org/](https://www.nuscenes.org/)

 현대자동차그룹과 앱티브(Aptiv)의 합작사인 '모셔널(Motional)'이 구축한 자율주행 개발을 위한 Dataset으로 360도 감지를 위한 6대의 카메라와 5대의 레이더, 1대의 LiDAR, IMU, GPS를 통해 수집된 데이터이다. 즉 자율주행 차량에 필요한 모든 센서를 부착 후 수집하여, 완전한 자율주행 차량 개발에 응용이 가능하며, 1,400,000의 카메라 이미지와 390,000 라이다 포인트를 갖고 있으며, 데이터 수집을 위한 운행은 보스턴과 싱가포르에서 수행되었다. 23개의 객체 분류를 갖고있다.

<br>

---

### 2.3. Waymo Dataset
[https://waymo.com/open](https://waymo.com/open)


 Waymo Dataset은 CVPR 2019에서 공개된 연구 목적 비상업용 Dataset으로 Motion Dataset과 Perception Dataset으로 나누어 제공되며, 자율주행 자동차의 인지분야와 관련된 데이터는 Perception Dataset이다. 

 이 Dataset은 Waymo의 자율주행 차량이 1950개 주행 구간에서 수집한 지역별, 시간별, 날씨별 데이터를 포함하고 있으며 각 구간은 10Hz, 20초의 연속주행 데이터를 포함하고 있다. 즉 20,000,000개 이상의 프레임 데이터와 시간 환산 시 500시간이 넘는 데이터이다. 또한 전방 카메라 데이터 외에도 5개의 라이다(LiDAR, 중거리 LiDAR 1개, 단거리 LiDAR 4개) 데이터 등도 포함하고 있으며 4개의 Class 정보(Vehicles, Pedestrians, Cyclists, Signs)와 1000개의 카메라 Segments 데이터를 포함하고 있다.

<br>

---
### 2.4. BDD100K
[https://www.bdd100k.com/](https://www.bdd100k.com/)


 UC 버클리 인공지능 연구실(BAIR)에서 공개한 Dataset으로 40초의 비디오 시퀀스, 720px 해상도, 30 fps의 동영상으로 취득된 100,000개의 비디오 시퀀스로 구성된다. 해당 Dataset에는 다양한 날씨 조건은 물론, GPS 정보, IMU 정보, 시간 정보도 포함되어 있다. 또한 차선 및 주행 가능 영역에 대한 라벨링이 되어있다. 그리고 버스, 신호등, 교통 표지판, 사람, 자전거, 트럭, 자동차 등의 정보가 담긴 100,000개의 이미지에 라벨링이 완료된 2D Bounding Box가 포함되어 있다.

 BDD100K는 이와 같은 정보를 통해 물체 검출, 세그먼테이션, 운전 가능 지역, 차선 검출 등의 Task 수행이 가능하다.

<br>


---
## 3. Methodology



### [Computing Environment]

* *OS : Windows 10 Pro 21H2 / WSL2 Ubuntu*
* *CPU : Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz*
* *RAM : 32GB*
* *GPU : NVIDIA GeForce GTX970 4GB (Driver version - 497.09, CUDA version - 11.5)* 
* *Anaconda Command line client (version 1.9.0)*



*F 드라이브 내에 project 폴더를 만들어서 해당 프로젝트를 진행하였으며, Anaconda를 이용하여 가상환경을 설정하여 진행하였음*



패키지 호환성을 위하여 Python 버전은 3.8버전을 설치할 것을 권장

```powershell
conda create -n 'env_name' python=3.8
```




---
### 3.1. YOLOv5(You Only Look Once)

[https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)

<br>

<center><iframe width="717" height="269" src="https://www.youtube.com/embed/NM6lrxy0bxs" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></center>

<br>

YOLO는 'You Only Look Once'의 약자로 2016년 Joseph Redmon이 CVPR에서 발표한 'You Only Look Once: Unified, Real-Time Object Detection' 객체 탐지(Object Detection) 딥러닝 모델이다. 

해당 논문 발표 당시에는 대표적인 객체 탐지 기법으로 R-CNN(Region with Convolutional Neural Network) 계열(R-CNN, Fast R-CNN, Faster R-CNN 등)과 DPM(Deoformable Part Models) 등이 존재하고 있었다. 그러나 해당 기법들의 정확도는 좋았으나, 객체 탐지를 실시간으로 하기에는 너무 느렸다. Faster R-CNN의 경우에도 7fps가 최대 속도였다. 



<center><img src="\images\RCNN.png" style="zoom: 67%;" /></center>



이러한 기법들은 2단계로 이미지를 처리하기 때문이었다. 즉 이미지 입력이 들어오면 객체가 있을 것으로 추정되는 영역 ROI(Region Of Interest)를 추출한 이후, 다음 단계에서 ROI들은 분류기(Classifier)에 의해 객체별로 분류가 이루어지는 구조였다.



<center><img src="\images\yolo_model.png" style="zoom: 67%;" /></center>



그러나 YOLO는 다르다. YOLO는 단 한 단계의 구조로 네트워크가 작동하는데, 즉 이미지의 특징(Feature) 추출, 객체(Object)의 위치 파악, 객체 분류(Classification)이 한 번에 수행되므로 앞서 언급된 구조들에 비해서 정확도는 조금 떨어지지만 처리 속도가 매우 빠르고 간단하다.



즉, YOLO의 주요한 특징은 크게 다음과 같이 3가지로 설명될 수 있다.



* **You Only Look Once**

  YOLO는 ROI등의 기법 사용 없이 이미지 전체를 단 한번만 본다.

* **Unified Detection**
  
  다른 객체 탐지 모델 들은 다양한 전처리 모델과 인공 신경망을 결합해서 사용하지만, YOLO는 단 하나의 인공신경망에서 이를 전부 처리한다. 이런 특징 때문에 YOLO가 다른 모델보다 간단해 보인다.
  
* **Real-Time Object Detection**

  2단계의 객체 탐지 모델들에 비해서 매우 빠르게 객체 탐지가 가능하기 때문에 실시간으로 사용이 가능하다.



가장 초창기 YOLOv1의 구조는 다음과 같다.

<center><img src="\images\yolo_arch.png" style="zoom: 67%;" /></center>

전체적인 네트워크 신경만 구조는 총 24개의 합성곱 레이어(Convolutional Layer)와 2개의 Fully-Connected Layer를 사용하고 있다. 

* **Pre-trained Network**

   위의 이미지 중 좌측부터 6개의 Layer Box는 Pre-trained Network로 GooLeNet을 사용해 ImageNet 1000-class Dataset을 사전에 학습하고 이를 Fine-tuning한 네트워크이다. 이 사전학습 네트워크는 총  20개의 합성곱 레이어(Convolutional Layer)로 이루어져 있다. Fine-tuning 과정에서의 특이한 점은 ImageNet의 이미지는 224*224의 크기를 갖지만 이미지를 의도적으로 2배 키워 해당 데이터를 학습에 사용하였다.

* **Reduction Layer**

   YOLO는 1x1의 합성곱 레이어(Convolutional Layer)를 교대로 적용하여 이를 이용하여 Feature Space를 축소시켜(reduce), 계산 속도를 향상시켰다.

* **Training Layer**

   2개의 Fully-Connected Layer로 이루어져 있으며, 앞선 네트워크에서 학습한 Feature를 토대로 Class probability와 Bounding Box의 위치를 학습하고 예측한다. YOLO의 최종 출력은 7x7x30의 예측 텐서(prediction tensors)로 나온다.



여기까지가 YOLO에 대한 간략하면서도 전체적인 설명이었다.

이번 프로젝트에서 사용할 YOLOv5는 이러한 YOLO의 5번째 버전이라는 뜻이다. 다만, 기존의 버전들과의 차이점이 있다면 YOLOv5는 YOLO를 처음 고안한 Joseph Redmon에 의해 탄생된 것이 아니라는 점이다. YOLOv5는 Glenn Jocher가 만든 것으로 속도 측면에서는 기존의 YOLO 이외에도 다른 객체 탐지 모델에 비해 탐지 속도가 매우 빠르다는 점이다.



<center><img src ="https://user-images.githubusercontent.com/26833433/136763877-b174052b-c12f-48d2-8bc4-545e3853398e.png"></center>



YOLOv5가 객체 탐지(Object Detection) 모델 중에 매우 빠른 편에 속하므로, 실시간 요소에 적용하기 적합하며, 이러한 이유로 본 프로젝트, 즉 자율주행의 객체 탐지 모델에 사용하기 유리할 것으로 판단되어 YOLOv5를 본 프로젝트에서 사용하기로 하였다.

<br>

---

**[YOLOv5 Setting]**

<br>

 1 . 우선 F 드라이브 내에 본 프로젝트 파일을 정리하고 저장할 폴더 'project'를 생성한 후, Windows PowerShell을 이용하여 해당 폴더로 이동한다.

```powershell
cd F:\project
pwd
```

```plaintext
Path
----
F:\project
```

 2 . 이후 git 명령어를 이용하여 Github의 Yolov5 Repository를 우리가 사용할 폴더에 clone 해주도록 한다.

```powershell
git clone https://github.com/ultralytics/yolov5
```

```plaintext
Cloning into 'yolov5'...
remote: Enumerating objects: 10222, done.
remote: Total 10222 (delta 0), reused 0 (delta 0), pack-reused 10222
Receiving objects: 100% (10222/10222), 10.49 MiB | 9.56 MiB/s, done.
Resolving deltas: 100% (7066/7066), done.
```

3 . 이후 project 폴더를 확인해보면 'yolov5' 폴더가 생성된 것을 확인할 수 있다. 그리고 해당 폴더 내에 어떠한 파일들이 있는지 확인해보자.

```powershell
ls
```

```plaintext
    디렉터리: F:\project


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----      2021-12-16   오후 2:09                yolov5
```

```powershell
cd yolov5
ls
```

```plaintext
    디렉터리: F:\project\yolov5


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----      2021-12-16   오후 2:09                .github
d-----      2021-12-16   오후 2:09                data
d-----      2021-12-16   오후 2:09                models
d-----      2021-12-16   오후 2:09                utils
-a----      2021-12-16   오후 2:09           3905 .dockerignore
-a----      2021-12-16   오후 2:09             77 .gitattributes
-a----      2021-12-16   오후 2:09           4219 .gitignore
-a----      2021-12-16   오후 2:09           1619 .pre-commit-config.yaml
-a----      2021-12-16   오후 2:09           5085 CONTRIBUTING.md
-a----      2021-12-16   오후 2:09          12603 detect.py
-a----      2021-12-16   오후 2:09           2227 Dockerfile
-a----      2021-12-16   오후 2:09          21073 export.py
-a----      2021-12-16   오후 2:09           6524 hubconf.py
-a----      2021-12-16   오후 2:09          35801 LICENSE
-a----      2021-12-16   오후 2:09          15038 README.md
-a----      2021-12-16   오후 2:09            928 requirements.txt
-a----      2021-12-16   오후 2:09            974 setup.cfg
-a----      2021-12-16   오후 2:09          33294 train.py
-a----      2021-12-16   오후 2:09          58110 tutorial.ipynb
-a----      2021-12-16   오후 2:09          18365 val.py
```

4 . 이제 Yolov5를 원활히 구동하기 위해서는 YOLOv5 작동에 필요한 라이브러리를 설치해야 한다. 이러한 의존성 패키지 관련 정보는 requirments.txt에 정리되어 있으므로 해당 파일을 열어서 확인해보자.


```powershell
cat requirements.txt
```

```plaintext
# pip install -r requirements.txt

# Base ----------------------------------------
matplotlib>=3.2.2
numpy>=1.18.5
opencv-python>=4.1.2
Pillow>=7.1.2
PyYAML>=5.3.1
requests>=2.23.0
scipy>=1.4.1
torch>=1.7.0
torchvision>=0.8.1
tqdm>=4.41.0

# Logging -------------------------------------
tensorboard>=2.4.1
# wandb

# Plotting ------------------------------------
pandas>=1.1.4
seaborn>=0.11.0

# Export --------------------------------------
# coremltools>=4.1  # CoreML export
# onnx>=1.9.0  # ONNX export
# onnx-simplifier>=0.3.6  # ONNX simplifier
# scikit-learn==0.19.2  # CoreML quantization
# tensorflow>=2.4.1  # TFLite export
# tensorflowjs>=3.9.0  # TF.js export

# Extras --------------------------------------
# albumentations>=1.0.3
# Cython  # for pycocotools https://github.com/cocodataset/cocoapi/issues/172
# pycocotools>=2.0  # COCO mAP
# roboflow
thop  # FLOPs computation
```

5 . 해당 requirements.txt 파일에 의존성 패키지에 관한 정보가 정리되어 있고, 해당 파일 내에 서술된 것처럼 'pip install' 명령어 실행을 통하여 YOLOv5 작동에 필요한 라이브러리를 받을 수 있다.

```powershell
pip install -r requirements.txt
```



여기까지 문제없이 잘 이루어졌다면, 본격적으로 YOLOv5의 사용 준비는 모두 끝난 것이다.

---
###  3.2. Dataset Pre-Processing



자율주행 자동차의 이미지 검출 모델 학습에 사용되는 Dataset들은 모두 통일된 Format으로 존재하는 것이 아니라, 객체가 존재하는 곳을 표현하는 Bounding Box의 좌표값에 대한 Labeling 방법, 그리고 객체의 종류를 정리해놓은 Annotation의 방법이 서로 다르기 때문에 사용할 Dataset의 Format을 확인하여 Yolov5의 학습에 적합한 형태로 가공해주어야 한다.



YOLO의 경우에는 Yolov5 PyTorch라는 Dataset Format을 사용하고 있다.





---
### 3.3. Dataset Training



이제 본격적으로 YOLOv5을 Preprocessing이 완료된 Dataset을 이용하여 학습을 진행해주면 된다.





학습에 앞서 YOLOv5에 맞는 torch/PyTorch, torchvision, torchaudio, cudatoolkit을 설치해주어야 한다.

몇 번의 시행착오 결과, cudatoolkit은 11버전에선 잘 작동하지 않았으며, 가장 잘 맞는 세팅은 Python 3.8.12 기준 Pytorch(torch) = 1.10.0, torchvision = 0.11.1, torchaudio = 0.10.0, cudatoolkit = 10.2 버전이다.

```powershell
# conda를 이용한 설치
conda install pytorch=1.10.0 torchvision=0.11.1 torchaudio=0.10.0 cudatoolkit=10.2 -c pytorch

# pypi를 이용한 설치
pip3 install torch==1.10.0+cu102 torchvision==0.11.1+cu102 torchaudio===0.10.0+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html
```



<img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/model_comparison.png" style="zoom: 25%;" />



 우리는 자율주행 자동차의 상황을 가정하여, 정확도도 중요하지만 더 중요한 것은 객체 검출의 빠른 응답성이며, 컴퓨팅 환경의 특성상 큰 가중치 모델(ex. YOLOv5l, YOLOv5x) 등의 학습에 어려움이 있으므로, **YOLOv5s** pre-trained model을 학습 가중치로 사용한다.

```powershell
python train.py --img 416 --batch 16 --epochs 100 --data 'Dataset/KITTI_YOLOv5/data.yaml' --weights yolov5s.pt --cache
```

* **--img** : Dataset image의 size (1:1 비율 기준)

* **--batch** : 학습에 사용할 batch의 사이즈 (Iteration 횟수 = Dataset Image 개수 / Batch Size)

* **--epochs** : 학습에 사용할 Epochs의 횟수

* **--data** : Dataset의 정보(YOLO Format)가 저장된 YAML 파일의 위치

* **--cfg** : 모델의 YAML 파일의 위치

* **--weights** : 학습에 사용할 가중치 모델 (custom model 혹은 pretained model; yolov5s.pt, yolov5l.pt 등)

* **--device** : 학습에 사용할 하드웨어 선정 CPU(cpu) / GPU(0, 1, 2, 3) 

  Nvidia CUDA 가속 가능 하드웨어 사용시 GPU 사용 추천

* **--name** : 학습 정보를 'runs' 폴더에 저장할 때 사용할 이름 (default = exp, exp2, exp3, ...)

* **--resume** : 학습을 재개를 명령하는 변수, 학습 과정 중 중간에 종료된 경우 해당 명령어 사용시 마지막으로 시행했던 epoch부터 다시 시작하며 weights는 'last.pt' 사용

```plaintext
train: weights=yolov5s.pt, cfg=, data=Dataset/KITTI_YOLOv5/data.yaml, hyp=data\hyps\hyp.scratch.yaml, epochs=100, batch_size=16, imgsz=416, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, evolve=None, bucket=, cache=ram, image_weights=False, device=, multi_scale=False, single_cls=False, adam=False, sync_bn=False, workers=8, project=runs\train, name=exp, exist_ok=False, quad=False, linear_lr=False, label_smoothing=0.0, patience=100, freeze=0, save_period=-1, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest
github: up to date with https://github.com/ultralytics/yolov5
YOLOv5  v6.0-147-g628817d torch 1.10.0+cu102 CUDA:0 (NVIDIA GeForce GTX 970, 4096MiB)

hyperparameters: lr0=0.01, lrf=0.1, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
Weights & Biases: run 'pip install wandb' to automatically track and visualize YOLOv5  runs (RECOMMENDED)
TensorBoard: Start with 'tensorboard --logdir runs\train', view at http://localhost:6006/
Overriding model.yaml nc=80 with nc=9

                 from  n    params  module                                  arguments
  0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2]
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]
  2                -1  1     18816  models.common.C3                        [64, 64, 1]
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]
  4                -1  2    115712  models.common.C3                        [128, 128, 2]
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]
  6                -1  3    625152  models.common.C3                        [256, 256, 3]
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]
  8                -1  1   1182720  models.common.C3                        [512, 512, 1]
  9                -1  1    656896  models.common.SPPF                      [512, 512, 5]
 10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']
 12           [-1, 6]  1         0  models.common.Concat                    [1]
 13                -1  1    361984  models.common.C3                        [512, 256, 1, False]
 14                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']
 16           [-1, 4]  1         0  models.common.Concat                    [1]
 17                -1  1     90880  models.common.C3                        [256, 128, 1, False]
 18                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]
 19          [-1, 14]  1         0  models.common.Concat                    [1]
 20                -1  1    296448  models.common.C3                        [256, 256, 1, False]
 21                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]
 22          [-1, 10]  1         0  models.common.Concat                    [1]
 23                -1  1   1182720  models.common.C3                        [512, 512, 1, False]
 24      [17, 20, 23]  1     37758  models.yolo.Detect                      [9, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
Model Summary: 270 layers, 7043902 parameters, 7043902 gradients, 15.9 GFLOPs

Transferred 343/349 items from yolov5s.pt
Scaled weight_decay = 0.0005
optimizer: SGD with parameter groups 57 weight, 60 weight (no decay), 60 bias
train: Scanning 'F:\project\yolov5\Dataset\KITTI_YOLOv5\train\labels' images and labels...5237 found, 0 missing, 0 empt
train: WARNING: F:\project\yolov5\Dataset\KITTI_YOLOv5\train\images\000005_png.rf.281f637fc5adac2f716ffd1c902f8cf2.jpg: 1 duplicate labels removed
train: WARNING: Cache directory F:\project\yolov5\Dataset\KITTI_YOLOv5\train is not writeable: [WinError 183] 파일이 이
미 있으므로 만들 수 없습니다: 'F:\\project\\yolov5\\Dataset\\KITTI_YOLOv5\\train\\labels.cache.npy' -> 'F:\\project\\yolov5\\Dataset\\KITTI_YOLOv5\\train\\labels.cache'
train: Caching images (2.7GB ram): 100%|█████████████████████████████████████████| 5237/5237 [00:01<00:00, 3773.06it/s]
val: Scanning 'F:\project\yolov5\Dataset\KITTI_YOLOv5\valid\labels.cache' images and labels... 1496 found, 0 missing, 0
val: Caching images (0.8GB ram): 100%|███████████████████████████████████████████| 1496/1496 [00:00<00:00, 2950.69it/s]
module 'signal' has no attribute 'SIGALRM'

AutoAnchor: 4.44 anchors/target, 0.994 Best Possible Recall (BPR). Current anchors are a good fit to dataset
Image sizes 416 train, 416 val
Using 8 dataloader workers
Logging results to runs\train\exp
Starting training for 100 epochs...

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      0/99     1.45G   0.08324    0.0397   0.03698        58       416: 100%|██████████| 328/328 [01:29<00:00,  3.67it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:12<0
                 all       1496      10764       0.42      0.157     0.0933     0.0316

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      1/99     1.65G   0.06682   0.03277   0.02447        50       416: 100%|██████████| 328/328 [01:23<00:00,  3.92it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:12<0
                 all       1496      10764      0.281       0.34        0.2     0.0752

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      2/99     1.66G   0.06149   0.03192   0.02134        32       416: 100%|██████████| 328/328 [01:22<00:00,  3.99it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:12<0
                 all       1496      10764      0.531      0.301      0.269      0.108

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      3/99     1.66G   0.05673    0.0311   0.01919        43       416: 100%|██████████| 328/328 [01:19<00:00,  4.12it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:11<0
                 all       1496      10764      0.612      0.361      0.358      0.163

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      4/99     1.66G    0.0535   0.03167   0.01767        54       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:11<0
                 all       1496      10764       0.64      0.373      0.384      0.181

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      5/99     1.66G   0.05106   0.03029   0.01582        49       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:11<0
                 all       1496      10764      0.517      0.481      0.478      0.219

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      6/99     1.66G    0.0498   0.03004   0.01489        39       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:11<0
                 all       1496      10764      0.647      0.481      0.504      0.245

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      7/99     1.66G   0.04903   0.02983    0.0142        31       416: 100%|██████████| 328/328 [01:18<00:00,  4.16it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:11<0
                 all       1496      10764      0.537      0.544      0.515      0.247

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      8/99     1.66G   0.04816   0.02901   0.01342        83       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:11<0
                 all       1496      10764      0.577       0.54      0.551      0.271

     Epoch   gpu_mem       box       obj       cls    labels  img_size
      9/99     1.66G   0.04753   0.02913   0.01329        55       416: 100%|██████████| 328/328 [01:18<00:00,  4.17it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:11<0
                 all       1496      10764       0.64      0.513       0.57      0.286

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     10/99     1.66G   0.04718   0.02891   0.01237        27       416: 100%|██████████| 328/328 [01:19<00:00,  4.14it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:11<0
                 all       1496      10764        0.6      0.582      0.592      0.303

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     11/99     1.66G   0.04662   0.02938   0.01197        63       416: 100%|██████████| 328/328 [01:18<00:00,  4.17it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.635      0.572      0.599      0.308

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     12/99     1.66G   0.04599   0.02877   0.01177        74       416: 100%|██████████| 328/328 [01:20<00:00,  4.09it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:11<0
                 all       1496      10764      0.638      0.612      0.624      0.314

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     13/99     1.66G   0.04606   0.02858   0.01163        71       416: 100%|██████████| 328/328 [01:18<00:00,  4.16it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:11<0
                 all       1496      10764      0.681      0.607      0.648      0.349

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     14/99     1.66G   0.04511   0.02833   0.01097        56       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.664      0.601       0.63      0.333

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     15/99     1.66G   0.04522   0.02847   0.01059        38       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.661      0.598      0.637      0.334

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     16/99     1.66G   0.04479   0.02773    0.0108        22       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.672      0.621      0.649      0.352

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     17/99     1.66G   0.04438   0.02813   0.01054        45       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.722      0.621      0.669      0.364

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     18/99     1.66G   0.04442    0.0282   0.01006        46       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764       0.72      0.595       0.66      0.358

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     19/99     1.66G   0.04419   0.02773  0.009892        78       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.766      0.612      0.677      0.368

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     20/99     1.66G   0.04367   0.02741  0.009715        45       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764       0.72      0.634      0.685      0.374

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     21/99     1.66G   0.04352   0.02762  0.009474        35       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.716      0.644      0.684      0.384

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     22/99     1.66G   0.04315    0.0272  0.009536        27       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.677      0.672      0.682      0.381

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     23/99     1.66G   0.04328   0.02743  0.009294        39       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.766      0.651      0.701      0.393

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     24/99     1.66G   0.04283    0.0272  0.009115        52       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.816      0.596      0.694      0.382

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     25/99     1.66G   0.04275   0.02707  0.009128        63       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764       0.77      0.614      0.696      0.391

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     26/99     1.66G   0.04249   0.02701  0.008743        43       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.788      0.634      0.711      0.403

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     27/99     1.66G   0.04224   0.02714  0.008738        46       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.777      0.637      0.703      0.404

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     28/99     1.66G   0.04188   0.02663  0.008663        44       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.766      0.658      0.716      0.399

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     29/99     1.66G   0.04205   0.02694  0.008606        73       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764       0.73      0.693      0.725      0.419

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     30/99     1.66G    0.0421   0.02664  0.008372        54       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.703      0.706       0.72      0.409

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     31/99     1.66G   0.04135   0.02688  0.008196        31       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.733       0.69      0.722      0.423

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     32/99     1.66G   0.04142   0.02656  0.008112        33       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.833      0.661      0.745      0.429

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     33/99     1.66G   0.04118    0.0264  0.007884        24       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.733      0.686      0.727      0.423

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     34/99     1.66G   0.04095   0.02626  0.007735        26       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.758      0.676      0.723      0.428

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     35/99     1.66G   0.04068   0.02624  0.007821        19       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.765      0.641       0.72      0.422

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     36/99     1.66G   0.04096   0.02651  0.007695        41       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764       0.78      0.683      0.737      0.428

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     37/99     1.66G   0.04088   0.02619  0.007663        67       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.748      0.699      0.737      0.428

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     38/99     1.66G   0.04053   0.02593  0.007452        29       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.781      0.686      0.743      0.431

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     39/99     1.66G   0.04048   0.02596  0.007258        35       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.789      0.675      0.746      0.434

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     40/99     1.66G   0.04013   0.02603  0.007301        59       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.718      0.723      0.744      0.447

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     41/99     1.66G   0.03998   0.02567  0.007279        32       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.807      0.669      0.745       0.44

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     42/99     1.66G   0.04002   0.02608  0.007095        53       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.749      0.701       0.75      0.446

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     43/99     1.66G   0.03937   0.02536  0.007091        47       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.786      0.676      0.741      0.438

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     44/99     1.66G   0.03981   0.02585  0.007216        71       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.796      0.704      0.753      0.442

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     45/99     1.66G   0.03899   0.02543  0.006904        65       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.784      0.703      0.759      0.455

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     46/99     1.66G    0.0393   0.02584  0.006807        54       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.749       0.74      0.763      0.453

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     47/99     1.66G   0.03895   0.02553  0.006826        24       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.799      0.717       0.77      0.458

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     48/99     1.66G    0.0387   0.02501  0.006643        55       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.849      0.683       0.76      0.458

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     49/99     1.66G   0.03906   0.02553  0.006735        55       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.793      0.718      0.765      0.473

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     50/99     1.66G   0.03876   0.02528  0.006478        50       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.798      0.726      0.772      0.471

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     51/99     1.66G   0.03874   0.02514  0.006694        36       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764       0.82      0.696      0.758      0.463

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     52/99     1.66G   0.03859   0.02509  0.006541        33       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.814      0.716      0.777      0.469

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     53/99     1.66G   0.03821   0.02467  0.006532        29       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.808      0.716      0.773      0.473

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     54/99     1.66G    0.0381   0.02479  0.006236        73       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.825      0.698      0.773       0.47

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     55/99     1.66G   0.03802   0.02448  0.006511        29       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.801       0.73      0.773      0.477

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     56/99     1.66G   0.03789   0.02453  0.006197        41       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.769      0.753      0.777      0.469

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     57/99     1.66G   0.03797   0.02507  0.006109        49       416: 100%|██████████| 328/328 [01:18<00:00,  4.17it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.779      0.741      0.778      0.478

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     58/99     1.66G   0.03758   0.02446  0.006092        55       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.823      0.703      0.777      0.484

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     59/99     1.66G    0.0374   0.02442  0.005976        31       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.834      0.715       0.78      0.483

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     60/99     1.66G   0.03704   0.02417  0.005903        74       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.801      0.738      0.783      0.493

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     61/99     1.66G   0.03727   0.02402  0.006025        56       416: 100%|██████████| 328/328 [01:18<00:00,  4.17it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.832      0.706      0.775      0.482

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     62/99     1.66G   0.03706   0.02452  0.005827        40       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.814      0.738      0.781      0.486

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     63/99     1.66G   0.03685   0.02427  0.005797        66       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.813      0.736      0.783      0.484

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     64/99     1.66G   0.03683   0.02414  0.005559        53       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.809      0.741      0.782      0.487

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     65/99     1.66G   0.03664   0.02415  0.005662        56       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764       0.84      0.706      0.775      0.492

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     66/99     1.66G    0.0367   0.02426  0.005662        59       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.808      0.724      0.778      0.488

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     67/99     1.66G   0.03639     0.024  0.005471        82       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.809      0.753      0.784      0.498

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     68/99     1.66G   0.03648   0.02399  0.005519        82       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.801       0.73      0.783      0.498

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     69/99     1.66G   0.03638    0.0238  0.005442        58       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.847      0.719      0.795      0.497

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     70/99     1.66G   0.03635   0.02372  0.005305        19       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.823      0.734      0.786        0.5

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     71/99     1.66G   0.03619   0.02342  0.005361        31       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.849      0.713       0.79      0.505

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     72/99     1.66G   0.03617   0.02366  0.005392        71       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.786      0.761       0.79      0.504

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     73/99     1.66G   0.03591   0.02368   0.00531        52       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.767       0.77       0.79      0.506

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     74/99     1.66G   0.03597   0.02353  0.005158        32       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.846      0.733      0.796      0.507

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     75/99     1.66G   0.03544   0.02336   0.00504        35       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.842      0.731       0.79      0.512

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     76/99     1.66G   0.03577   0.02373  0.005084        75       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.825      0.744      0.791      0.512

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     77/99     1.66G   0.03521    0.0232  0.004905        60       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.821       0.73      0.791      0.509

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     78/99     1.66G   0.03544   0.02354  0.004891        26       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.799      0.754      0.795      0.513

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     79/99     1.66G   0.03535   0.02299  0.004931        51       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.814      0.751      0.798      0.514

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     80/99     1.66G   0.03539   0.02329  0.005002        28       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764       0.86      0.725      0.795      0.511

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     81/99     1.66G   0.03505   0.02343  0.004824        45       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.832      0.731      0.795      0.513

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     82/99     1.66G   0.03531   0.02341  0.004916        48       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.819      0.721      0.789      0.512

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     83/99     1.66G   0.03493   0.02294  0.004854        90       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.844      0.718      0.791       0.51

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     84/99     1.66G   0.03494   0.02304  0.004698        58       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.803      0.743      0.796      0.519

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     85/99     1.66G   0.03505   0.02308  0.004799        56       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.835      0.737      0.797      0.516

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     86/99     1.66G   0.03479   0.02298  0.004711        49       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.826      0.741      0.793      0.513

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     87/99     1.66G   0.03483   0.02292   0.00472        54       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.812      0.742      0.794      0.517

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     88/99     1.66G    0.0349   0.02316  0.004642        64       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.811      0.737      0.799       0.52

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     89/99     1.66G   0.03447   0.02259  0.004679        44       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.849      0.721      0.798      0.523

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     90/99     1.66G    0.0343   0.02289  0.004766        37       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.837      0.735      0.796      0.521

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     91/99     1.66G   0.03451   0.02277  0.004571        49       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.796      0.761      0.798      0.522

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     92/99     1.66G   0.03446   0.02254  0.004675        33       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.852      0.738      0.799      0.519

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     93/99     1.66G    0.0346   0.02284  0.004604        59       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.785      0.771      0.799       0.52

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     94/99     1.66G   0.03437    0.0225  0.004508        73       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.851      0.728        0.8      0.522

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     95/99     1.66G   0.03406   0.02211  0.004517        22       416: 100%|██████████| 328/328 [01:18<00:00,  4.18it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.806      0.753        0.8       0.52

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     96/99     1.66G   0.03426   0.02259  0.004472        61       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.843      0.732      0.804      0.525

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     97/99     1.66G   0.03438   0.02287  0.004456        81       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.824      0.751      0.807      0.524

     98/99     1.66G   0.03443   0.02249  0.004442        78       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.781      0.776      0.806      0.528

     Epoch   gpu_mem       box       obj       cls    labels  img_size
     99/99     1.66G   0.03405   0.02236  0.004483        70       416: 100%|██████████| 328/328 [01:18<00:00,  4.19it/
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:10<0
                 all       1496      10764      0.786      0.772      0.804      0.524
100 epochs completed in 2.498 hours.
Optimizer stripped from runs\train\exp\weights\last.pt, 14.4MB
Optimizer stripped from runs\train\exp\weights\best.pt, 14.4MB

Validating runs\train\exp\weights\best.pt...
Fusing layers...
Model Summary: 213 layers, 7034398 parameters, 0 gradients, 15.9 GFLOPs
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 47/47 [00:16<0
                 all       1496      10764      0.781      0.776      0.806      0.528
             Cyclist       1496        355      0.777      0.746      0.784      0.453
            DontCare       1496       2331      0.478      0.363      0.341      0.104
                Misc       1496        186      0.755      0.871      0.907      0.591
      Person_sitting       1496         22       0.65      0.591       0.64      0.298
                Tram       1496        130      0.912      0.953      0.971      0.694
               Truck       1496        228      0.941      0.956      0.979      0.782
                 Van       1496        590      0.895      0.939      0.961      0.733
                 car       1496       5948      0.878      0.937      0.964      0.757
              person       1496        974      0.742      0.629      0.708      0.335
Results saved to runs\train\exp
```




---
### 3.4. YOLOv5 (You Only Look Once)


---
### 3.5. OpenCV





---
## 4. Evaluation & Analysis

### 4. 1. yolov5s

### 4. 2. Epoch 25

### 4. 3. Epoch 50

### 4. 4. Epoch 100




---
## 5. Related Work


#### Autonomous Driving 

## 6. Conclusion & Discussion



## Reference

* Vehicle Detection in Urban Traffic Surveillance Images Based on Convolutional Neural Networks with Feature Concatenation / [https://doi.org/10.3390/s19030594](https://doi.org/10.3390/s19030594)
* Are we ready for autonomous driving? The KITTI vision benchmark suite / [https://ieeexplore.ieee.org/document/6248074](https://ieeexplore.ieee.org/document/6248074)
* Top 5 Autonomous Driving Dataset Open-Sourced At CVPR 2020 / [https://analyticsindiamag.com/top-5-autonomous-driving-dataset-open-sourced-at-cvpr-2020/](https://analyticsindiamag.com/top-5-autonomous-driving-dataset-open-sourced-at-cvpr-2020/)
* You Only Look Once: Unified, Real-Time Object Detection / [https://arxiv.org/abs/1506.02640](https://arxiv.org/abs/1506.02640)
* Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks / [https://arxiv.org/abs/1506.01497](https://arxiv.org/abs/1506.01497)
* You Only Look Once — 다.. 단지 한 번만 보았을 뿐이라구! / [https://medium.com/curg/you-only-look-once-%EB%8B%A4-%EB%8B%A8%EC%A7%80-%ED%95%9C-%EB%B2%88%EB%A7%8C-%EB%B3%B4%EC%95%98%EC%9D%84-%EB%BF%90%EC%9D%B4%EB%9D%BC%EA%B5%AC-bddc8e6238e2](https://medium.com/curg/you-only-look-once-%EB%8B%A4-%EB%8B%A8%EC%A7%80-%ED%95%9C-%EB%B2%88%EB%A7%8C-%EB%B3%B4%EC%95%98%EC%9D%84-%EB%BF%90%EC%9D%B4%EB%9D%BC%EA%B5%AC-bddc8e6238e2)

