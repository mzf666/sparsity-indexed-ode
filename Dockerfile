FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel

RUN apt update -y && DEBIAN_FRONTEND=noninteractive apt install -y --allow-unauthenticated --no-install-recommends \
    build-essential apt-utils cmake git curl vim ca-certificates \
    libjpeg-dev libpng-dev \
    libgtk3.0 libsm6 cmake ffmpeg pkg-config \
    qtbase5-dev libqt5opengl5-dev libassimp-dev \
    libboost-python-dev libtinyxml-dev bash \
    wget unzip libosmesa6-dev software-properties-common \
    libopenmpi-dev libglew-dev openssh-server \
    libosmesa6-dev libgl1-mesa-glx libgl1-mesa-dev patchelf libglfw3 zlib1g-dev unrar \
    libglib2.0-dev libsm6 libxext6 libxrender-dev freeglut3-dev ffmpeg

RUN pip install opencv-python==4.6.0.66 future==0.18.2 pathlib==1.0.1
RUN pip install pandas==1.3.5 easydict==1.10
RUN pip install seaborn==0.12.1 einops==0.6.0
RUN pip install scikit-image==0.19.3
RUN pip install PyPDF2==2.11.2
RUN pip install openpyxl==3.0.10
WORKDIR /HappyResearch

ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

#RUN python -c "from torchvision.models import resnet50,vgg16_bn;m=resnet50(weights='IMAGENET1K_V1');m=vgg16_bn(weights='IMAGENET1K_V1')"
#RUN pip install transformers
#
#RUN python -c "from transformers import (    AutoConfig,AutoModelForImageClassification,);model = AutoModelForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',    from_tf=False,    config=AutoConfig.from_pretrained(    'google/vit-base-patch16-224-in21k',    num_labels=100,    finetuning_task='image-classification',),ignore_mismatched_sizes=False,)"
#RUN pip install datasets
#RUN python -c "from transformers import (AutoConfig,AutoModelForQuestionAnswering);model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased',from_tf=False,config=AutoConfig.from_pretrained('bert-base-uncased',),)"
#RUN pip install evaluate
#RUN pip install scikit-learn
#RUN python -c "from transformers import (AutoConfig,AutoModelForSequenceClassification);model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased',from_tf=False,config=AutoConfig.from_pretrained('bert-base-cased',),)"
#
#RUN python -c "from torchvision.models import resnet50,vgg16_bn,mobilenet_v2;m=resnet50(weights='IMAGENET1K_V1');m=vgg16_bn(weights='IMAGENET1K_V1');m=mobilenet_v2(weights='IMAGENET1K_V2');"

COPY code code_static