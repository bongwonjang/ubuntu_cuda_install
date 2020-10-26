# ubuntu 18.04 환경에 CUDA, cuDNN 설치 매뉴얼

## ubuntu 18.04 설치(그래픽 카드는 RTX 2080ti 사용)

* ubuntu 18.04 설치 방법

1. usb를 통해 설치하는 방법(참고용)
<br>[Link1](https://nov19.tistory.com/53)
<br>[Link2](https://neoprogrammer.tistory.com/6)

2. 연구실 아무나 붙잡고 ubuntu 설치 usb를 부탁드린다.

## CUDA 설치 및 cuDNN 설치

* 본인이 있는 연구실에서 tensorflow는 보통 1.13~1.15를 사용하므로, 
  <br>CUDA는 10.0 버전 설치. 
  <br>cuDNN은 7.6.5 버전으로 설치해도 tensorflow 동작에 문제 없음

[tensorflow table 참고](https://www.tensorflow.org/install/source#gpu)

* 다른 블로그는 전부 실패.
  <br>유일하게 ubuntu 18.04(RTX 2080ti)에 무리없이 설치됨
  <br>5번 테스트 결과, 실패 없었음

[CUDA 설치 링크](https://medium.com/@exesse/cuda-10-1-installation-on-ubuntu-18-04-lts-d04f89287130)

* 링크의 설명은 10.1을 설치하지만, 10-0, 10.0으로 숫자를 바꾸면 해당 CUDA 버전
  설치

* cuDNN은 libcudnn7을 사용하면 CUDA 10.x에 맞는 cuDNN 최신 버전 설치

1. 먼저, 이미 깔려진 NVIDIA랑 관련된 것들을 전부 제거.  
   그냥 우분투를 깨끗하게 재설치하는 것이 편하다.

```
sudo rm /etc/apt/sources.list.d/cuda*

sudo apt remove --autoremove nvidia-cuda-toolkit

sudo apt remove --autoremove nvidia-*
```

2. CUDA PPA를 시스템에 설치

```
sudo apt update

sudo add-apt-repository ppa:graphics-drivers

sudo apt-key adv --fetch-keys  http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'

sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'
```

3. CUDA 10.0 설치. 만약 다른 버전을 설치하고 싶다면, cuda-10-0을 수정하면 됨.

```
sudo apt update

sudo apt install cuda-10-0

sudo apt install libcudnn7
```

4. profile 열기

```
sudo vi ~/.profile
```

5. profile에 CUDA PATH 추가. 이때, 저장은 `ESC` 1번 누르고, `Shift + z`를 
   2번 누른다. <br>**여기서도 CUDA 버전 주의!!!**

```
# set PATH for cuda 10.0 installation
if [ -d "/usr/local/cuda-10.0/bin/" ]; then
    export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
fi
```

6. reboot 후 잘 돌아가는지 확인

```
sudo reboot
```

* CUDA

```
nvcc --version
```

* NVIDIA Driver

```
nvidia-smi
```

* libcudnn

```
/sbin/ldconfig -N -v $(sed ‘s/:/ /’ <<< $LD_LIBRARY_PATH) 2>/dev/null | grep libcudnn
```

## Pytorch 및 Tensorflow 설치

* 먼저, pip3가 설치되어 있는지 확인. 본인은 Python3을 쓰므로, 다음과 같이 pip3를 설치한다.

```
sudo apt-get install python3-pip
```

### Pytorch 설치 방법
[옛날 Pytorch 링크](https://pytorch.org/get-started/previous-versions/)

* 위 링크에서 CUDA 10.0에 맞는 pip3 설치 양식을 찾는다. 본인은 아래를 선택.
  <br>근데, 이게 나중에 wavenet_vocoder를 설치하면서 의미가 없어지는 거 같다.
  <br>**일단 돌아가니 다행인 것으로 넘기겠다.**

```
# CUDA 10.0
pip3 install torch==1.2.0 torchvision==0.4.0
```

### Tensorflow 설치 방법
* 둘 다 설치해야 한다.

```
pip3 install tensorflow==1.14
pip3 install tensorflow-gpu==1.14
```

### 이후 다시 리부트 

```
sudo reboot
```

여기서 다음과 같은 python 코드로 gpu 사용 여부를 확인하면 된다.

```
import torch
import tensorflow as tf

print(torch.cuda.is_available())
print(tf.test.is_gpu_available())
```

또는 터미널에서

```
python3 -c "import torch; print(torch.cuda.is_available())"

python3 -c "import tensorflow as tf; print(tf.test.is_gpu_available())"
```

## 추가: 간혹 librosa 패키지를 설치할 때 생기는 building wheel for llvmlite <br>또는 building wheel for numba 문제해결

* python3 환경을 가정
  <br>[참고한 링크](https://acver.tistory.com/6)

```
sudo apt-get install llvm-7
# (librosa에서 사용될 llvmlite를 지원하는 llvm의 최소 버젼이다. 지원 버젼은 7.0 7.1 8.0으로 현재까지 알고 있다.)
```

```
sudo LLVM_CONFG=/usr/bin/llvm-config-7 pip3 install llvmlite==0.32.0

sudo LLVM_CONFG=/usr/bin/llvm-config-7 pip3 install numba==0.43.0
```

```
sudo LLVM_CONFG=/usr/bin/llvm-config-7 pip3 install librosa
```

```
sudo reboot
```

그리고

```
pip3 install librosa
```

## 추가: 설치한 라이브러리 버전 충돌 문제

**ex)**
<span style="color:red">
ERROR ... keras requires tensorflow 2.2 or higher. install tensorflow via pip install tensorflow ...
</span>

위의 문제는 keras 버전이 요구하는 tensorflow 버전이 2.2로 최신 버전을 요구
<br>따라서 설치된 keras의 버전을 낮추는 방식으로 해결할 수 있다.

[참고한 링크1](https://stackoverflow.com/questions/62465620/error-keras-requires-tensorflow-2-2-or-higher)

[참고한 링크2](https://forums.developer.nvidia.com/t/import-keras-requires-tensorflow-2-2-or-higher/140572)

```
pip3 install keras==2.3.1
```
