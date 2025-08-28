# In pyroki$
# Build
# docker build -f .docker/Dockerfile -t humble-pyroki:0.0.1 .
# RUN
# docker run -it --rm --gpus all --privileged --network host -v .:/root/pyroki/ --name humble_pyroki humble-pyroki:0.0.1 

FROM osrf/ros:humble-desktop-full

SHELL ["/bin/bash", "-c"]

# Install apt packages 
RUN apt update && apt install -y  \
    vim \
    net-tools \
    python3-pip \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-rosidl-generator-dds-idl

# Install pip packages
RUN python3 -m pip install -U pip gdown

# Clone unitree_ros2
WORKDIR /root
RUN git clone https://github.com/unitreerobotics/unitree_ros2

# Build unitree_ros2
WORKDIR /root/unitree_ros2
RUN source /opt/ros/humble/setup.bash && \
    colcon build
COPY setup.sh /root/unitree_ros2/setup.sh

# Copy repository to /root/pyroki
COPY . /root/pyroki

# Install pyroki
WORKDIR /root/pyroki
RUN python3 -m pip install --ignore-installed .

# Download sit terminal states and trajectories
# RUN cd examples/eetrack && \
#     gdown --id 1u53kk0U8BHesAoaDC6YB4Bgpl_mNkmgN

RUN echo "source /root/unitree_ros2/setup.sh" >> ${HOME}/.bashrc

WORKDIR /root
ENTRYPOINT ["/bin/bash"]
