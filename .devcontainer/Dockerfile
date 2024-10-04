##########################################################################################################################################################################################
# Commands to execute the training script within a detached docker container:

########################################################### First choose all the arguments used to build this docker container ###########################################################

# At first we use the pytorch prebuilt image from NVIDIA 
ARG FROM_IMAGE=nvcr.io/nvidia/pytorch:24.09-py3 

# Select the username, userID and groupID for the user which will be created inside the container
ARG USERNAME=EnvisionUser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# The working directory at which we will start inside the container 
ARG WORKING_DIR=/workspace/SAM2 




##########################################################################################################################################################################################
######################################## Create the first layer of the image  => Here all ubuntu packages and python packages should be installed ########################################
FROM $FROM_IMAGE AS ubuntu_cacher

# Assure the timezone is european => CPH 
ENV TZ=Europe/Copenhagen 
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && date 

# Install ubuntu packages 
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update \
    && apt-get -y --no-install-recommends install software-properties-common \
    && apt-get -y update \
    && apt-get -y upgrade \
    && apt-get -y --no-install-recommends install \
        unzip \
        curl \
        git \
        wget \
        sudo \
        ssh \
        tree

### Install aws client 
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
        unzip awscliv2.zip && \
        sudo ./aws/install 




##########################################################################################################################################################################################
######################################################## Create the second layer of the image => here a new user will be created #########################################################
FROM ubuntu_cacher AS user_cacher
ARG USERNAME
ARG USER_UID
ARG USER_GID

# Natively the image runs as root. That is not best practise and will give some problems. Thus we need to create and add another user (that we here also make root). 
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && usermod -aG root $USERNAME \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update -y \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
RUN usermod -s /bin/bash $USERNAME
RUN usermod -aG plugdev $USERNAME

# Install the bash-auto complete 
RUN sudo apt-get install bash-completion && sudo apt-get install --reinstall bash-completion




##########################################################################################################################################################################################
################################################### Create the third layer of the image => here python requirements will be installed ####################################################
FROM user_cacher AS python_package_installer

### Copy all requirements.txt files into the container 
COPY .devcontainer/requirements_sam2.txt .devcontainer/requirements_sam2.txt

### Install python requirements 
RUN python -m pip install pip -U
RUN python -m pip install -r .devcontainer/requirements_sam2.txt 


# Install GIC shared library - which includes all standard functions 
ADD "http://envision:Envision1234@192.168.19.36:8666/packages/" cache
RUN python -m pip install --extra-index-url http://envision:Envision1234@192.168.19.36:8666 CameraAISharedLibrary InfluxClient AWSSecretsHelper AITrainingSharedLibrary --trusted-host 192.168.19.36




##########################################################################################################################################################################################
####################################################### Create the final layer of the image  => Here the container is ready to run #######################################################
FROM python_package_installer AS runner 
ARG WORKING_DIR 
ARG USERNAME

# Use the newly created user
USER ${USERNAME}

# Set the current working directory 
WORKDIR ${WORKING_DIR} 

# Copy the current working directory into the working directory here 
COPY . ${WORKING_DIR} 

### Create the environmental variables needed for the Docker container 
ENV WANDB_API_KEY e7220eb15a89a255ac255d759c7089d1d1b88577
ENV IMAGE_DATASETS /Image_Datasets
ENV PYTHONPATH=${WORKING_DIR}:${PYTHONPATH}
ENV AWS_CONFIG_FILE /workspace/.aws/config
ENV AWS_SHARED_CREDENTIALS_FILE /workspace/.aws/credentials
