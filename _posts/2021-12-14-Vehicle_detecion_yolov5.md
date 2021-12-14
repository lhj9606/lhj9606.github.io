---
title: YOLOv5를 이용한 Vehicle Detection & Validation with Video Game Graphics (AI-X:Deep-Learning)
author: 
  name: Hong Jun Lee
  link: https://github.com/cotes2020
date: 2021-12-03 20:55:00 +0900
categories: [Blogging, AI+X]
tags: [AI, Deep Learning, Yolov5, Vehicle Detection]
pin: true
---

---

<center><b> Hanyang University 2021 Fall / AI+X:Deep Learning / Final Project </b></center>

---

## Member

|  이름  |    학과    |    학번    |        E-mail         | 역할 |
| :----: | :--------: | :--------: | :-------------------: | :----:|
| 이홍준 | 기계공학부 | 2015011203 | lhj9606@hanyang.ac.kr | Everything |

------

## ■ Video Summary










---
## 1. Proposal (Option A)

 4차 산업혁명 시대가 대두되면서, 여러 기술들이 발전하고 주목받고 있지만 그 중에서도 우리 생활에 밀접하게 관련있고, 접근성이 좋은 기술 중장 주목받는 기술 중 하나를 꼽자면 바로 '**자율주행**' 기술일 것이다. 이러한 자율주행 기술이 급격하게 발전하게 된 이유 중 하나는 Deep Learning(Deep Neural Network)의 발전, 그리고 CNN(Convolutional Neural Network) 등의 등장으로 인한 컴퓨터 비전(CV, Computer Vision) 분야의 많은 기술적 도약이 있었기에 가능해졌다.



<center><iframe title="vimeo-player" src="https://player.vimeo.com/video/192179726?h=849b6a5ec9" width="640" height="360" frameborder="0" allowfullscreen></iframe></center>

<center><b>Tesla의 자율주행 예시 영상</b></center>

---

### 1.1. Need

자율주행 자동차는 '인지', '판단', '제어'의 3가지 Step으로 스스로 주행을 관장하는데, '인지' 단계에서 ***카메라(Camera), 레이더(Rader), 라이다(LiDAR)*** 등의 여러 센서로 부터 취합한 정보를 바탕으로 차량 주변의 환경을 인식하며, 이를 토대로 '판단'과 '제어'를 수행한다. 즉 **'인지'** 단계가 자율주행 알고리즘에 있어 가장 첫번째 단계이므로, 자율주행에서 가장 중요하면서도 차량의 주변 환경을 명확하게 인식해야만 안전한 주행이 가능하기 때문에 매우 중요한 단계라고 할 수 있다.

이러한 **'인지'** 수행 과정에서 카메라, 레이더, 라이다 등의 센서 정보들 모두 중요하며, 각각의 장단점으로 인해 상호보완적으로 사용되지만, 물체의 종류를 식별하는 데에 가장 중요한 정보는 바로 ***'카메라 데이터'*** 이다. 본 프로젝트에서는 '인지' 단계에서의 자율주행 차량이 어떻게 주변 사물을 인식하는지 알아보기 위해 '카메라 데이터'를 입력으로 받는 모델(YOLOv5 사용)을 만들어 볼 것이다.



!["CNN을 활용한 Vehicle Detection의 사례 (a) Faster R-CNN, (b) DP-SSD300, (c) SSD300, (d) DP-SSD300"](https://www.mdpi.com/sensors/sensors-19-00594/article_deploy/html/images/sensors-19-00594-g001.png)

<center><b>CNN을 활용한 Vehicle Detection의 사례 (a) Faster R-CNN, (b) DP-SSD300, (c) SSD300, (d) DP-SSD300</b></center>

------

### 1.2. Objective

본 프로젝트는 YOLOv5를 기반으로 하여, 자율주행 환경에 필요한 이미지들을 모델에 학습하고, 자율주행 환경에 필요한 이미지 검출 속도에 맞게 모델을 튜닝할 예정이다. 

학습된 모델을 토대로 실제 차량에 탑재하여, 실제 차량과 그 이외의 주행 상황에 등장 가능한 물체에 대한 인식을 검출하는 시도를 하면 좋겠지만, 실제 차량에 컴퓨터와 카메라를 탑재하여 이를 검증해보는 것이 어렵다. 따라서, 본 프로젝트에서는 자율주행 환경을 모사 가능한 일부 게임 (ex. *GTA5, Euro Truck Simulator, Forza Horizon* 등)의 화면 데이터를 카메라 데이터로 대응시켜 모델의 객체 검출이 잘 이뤄지는지 확인해볼 예정이다.



## 2. Datasets

자율주행을 위한 딥러닝 학습용 Open Dataset은 다양하게 공개가 되어있습니다. 

* **COCO Dataset**


COCO Dataset은 객체 검출(Object Detection), 세그먼테이션(Segmentation), 키포인트 탐지(Keypoint Detection) 등과 같은 컴퓨터 비전(CV, Computer Vision) 분야의 Task를 목적으로 만들어진 Dataset이다. 

COCO Dataset의 구성은 다음과 같다. (COCO 2017 Dataset 기준)

- 학습(Training) : 118,000 images

- 검증(Validation) : 5,000 images

- 시험(Test) : 41,000 images

  

* **Waymo Dataset**


Waymo Dataset은 CVPR 2019에서 공개된 연구 목적 비상업용 Dataset으로 Motion Dataset과 Perception Dataset으로 나누어 제공되며, 자율주행 자동차의 인지분야와 관련된 데이터는 Perception Dataset이다. 

이 Dataset은 Waymo의 자율주행 차량이 1950개 주행 구간에서 수집한 지역별, 시간별, 날씨별 데이터를 포함하고 있으며 각 구간은 10Hz, 20초의 연속주행 데이터를 포함하고 있다. 또한 전방 카메라 데이터 외에도 5개의 라이다(LiDAR) 데이터 등도 포함하고 있으며 4개의 Class 정보(Vehicles, Pedestrians, Cyclists, Signs)와 1000개의 카메라 Segments 데이터를 포함하고 있다.



* **KITTI Dataset**


KITTI Dataset은 현존하는 자율주행 Dataset 중에서 가장 많이 사용되는 Dataset으로 자율주행의 다양한 인지 Task를 위한 라벨링(Annotation)을 제공한다. 2D/3D 물체 검출(Object Detection)은 물론 물체 추적(Tracking), 거리 추정, Odometry, 스테레오 비전, 영역 분할 등의 Task에 활용될 수 있다. 또한 알고리즘 벤치마크를 위한 리더보드도 제공한다.

다만 신호등에 대한 라벨링 정보가 없어 신호등에 대한 학습이 제한된다는 단점이 있다.



* **BDD100K**


UC 버클리 인공지능 연구실(BAIR)에서 공개한 Dataset으로 40초의 비디오 시퀀스, 720px 해상도, 30 fps의 동영상으로 취득된 100,000개의 비디오 시퀀스로 구성된다. 해당 Dataset에는 다양한 날씨 조건은 물론, GPS 정보, IMU 정보, 시간 정보도 포함되어 있다. 또한 차선 및 주행 가능 영역에 대한 라벨링이 되어있다. 그리고 버스, 신호등, 교통 표지판, 사람, 자전거, 트럭, 자동차 등의 정보가 담긴 100,000개의 이미지에 라벨링이 완료된 2D Bounding Box가 포함되어 있다.

BDD100K는 이와 같은 정보를 통해 물체 검출, 세그먼테이션, 운전 가능 지역, 차선 검출 등의 Task 수행이 가능하다.



* **PASCAL VOC Dataset**



 위의 데이터셋 중 자율주행 인지에 가장 자주 쓰이는 KITTI Dataset을 이용하여 우선 학습을 진행하고, 학습의 성능을 향상시키거나 변경시키기 위해 다른 Dataset도 추가적으로 사용해볼 계획이다.

***KITTI Dataset Link : http://www.cvlibs.net/datasets/kitti/***









## 3. Methodology

### 3.1. YOLOv5(You Only Look Once)



YOLO는 



https://github.com/ultralytics/yolov5



####  3.2. Dataset Pre-Processing

  ```bash
  $ git clone https://github.com/ultralytics/yolov5
  $ cd yolov5
  $ pip install -r requirements.txt
  ```

​	

####  3.2. Dataset Training

#### 3.3. YOLOv5 (You Only Look Once)

#### 3.4. OpenCV






## 4. Evaluation & Analysis


## 5. Related Work


#### Autonomous Driving 

## 6. Conclusion & Discussion



## Reference

* https://doi.org/10.3390/s19030594
