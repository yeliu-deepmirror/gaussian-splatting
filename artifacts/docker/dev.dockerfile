FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

SHELL [ "/bin/bash", "--login", "-c" ]

RUN apt-get update --fix-missing
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install -y \
      ffmpeg \
      libsm6 \
      libxext6 \
      libusb-1.0-0 \
      sudo \
      make \
      vim \
      wget \
      bzip2 \
      curl \
      flake8 \
      python-pytest \
      ninja-build \
      git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user
ARG usr=plenoxel
ARG uid=1000
ARG grp=plenoxel
ARG gid=1000
ENV USER $usr
ENV UID $uid
ENV GRP $grp
ENV GID $gid
ENV HOME /home/$USER

RUN groupadd -g $GID $GRP
RUN adduser --disabled-password \
    --gecos "Non-root user" \
    --uid $UID \
    --gid $GID \
    --home $HOME \
    --disabled-password --force-badname \
    $USER
# Copy the default configuration files for the user.
# This step is skipped in `adduser` since we already have the home directory
# for the cache path mounted when we created the container.
RUN cp -r /etc/skel/. /home/$USER
# Allow members of group sudo to execute any command.
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >>/etc/sudoers
# Add the user to group sudo.
RUN usermod -aG sudo "$USER"

RUN chown $USER:$GRP /home/$USER
RUN ls -ad /home/$USER/.??* | xargs chown -R $USER:$GRP

# COPY environment.yml /tmp/
# RUN chown $UID:$GID /tmp/environment.yml

USER $USER

# install miniconda
ENV MINICONDA_VERSION latest
ENV CONDA_DIR $HOME/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

# make non-activate conda commands available
ENV PATH=$CONDA_DIR/bin:$PATH

# make conda activate command available from /bin/bash --login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile

# make conda activate command available from /bin/bash --interative shells
RUN conda init bash

# build the conda environment
# RUN conda update --name base --channel defaults conda && \
#     conda env create --prefix /home/$USER/env --file /tmp/environment.yml --force && \
#     conda clean --all --yes

# activate in bash by default
# RUN echo "conda activate /home/$USER/env" >> /home/$USER/.bashrc
# SHELL ["conda", "run", "-p", "/home/$USER/env", "/bin/bash", "-c"]

# link to jupyter kernal
# RUN python -m ipykernel install --user --name=nerf_pl
