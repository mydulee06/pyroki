# In pyroki$
# Build
# docker build -f .docker/Dockerfile -t humble-pyroki:0.0.1 .
# RUN
# docker run -it --gpus all --privileged --network host -v .:/root/pyroki/ --name humble_pyroki humble-pyroki:0.0.1 

FROM osrf/ros:humble-desktop-full

SHELL ["/bin/bash", "-c"]

# --- Fix ROS2 expired GPG key ---
RUN rm -f /etc/apt/sources.list.d/ros2-latest.list \
 && rm -f /etc/apt/sources.list.d/ros2.list \
 && apt-get update && apt-get install -y curl gnupg2 lsb-release \
 && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    | tee /usr/share/keyrings/ros-archive-keyring.gpg > /dev/null \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    > /etc/apt/sources.list.d/ros2.list \
 && apt-get update


# --- Install apt packages ---
RUN apt-get install -y \
    vim \
    net-tools \
    python3-pip \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-rosidl-generator-dds-idl \
 && rm -rf /var/lib/apt/lists/*

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
RUN python3 -m pip install --no-cache-dir --ignore-installed -e .


# Download sit terminal states and trajectories
# RUN cd examples/eetrack && \
#     gdown --id 1u53kk0U8BHesAoaDC6YB4Bgpl_mNkmgN

# Build pyroki ros
WORKDIR /root/pyroki_ws/src
RUN source /root/unitree_ros2/setup.sh && \
    cp -r /root/pyroki/rospkg ./pyroki_ros && \
    cd .. && \
    colcon build

RUN echo "source /root/unitree_ros2/setup.sh" >> ${HOME}/.bashrc

WORKDIR /root
ENTRYPOINT ["/bin/bash"]
