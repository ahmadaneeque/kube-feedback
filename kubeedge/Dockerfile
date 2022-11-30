FROM ubuntu:bionic
USER root

ENV container docker
ENV LC_ALL C
ENV DOCKERVERSION=18.03.1-ce
ENV KEADMVERSION=v1.10.0
ENV DEBIAN_FRONTEND noninteractive

RUN sed -i 's/# deb/deb/g' /etc/apt/sources.list

RUN apt-get update \
    && apt-get install -y curl systemctl systemd systemd-sysv nano openssh-server sudo \
    && apt-get clean\
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN apt install  nano openssh-server sudo -y

RUN cd /lib/systemd/system/sysinit.target.wants/ \
    && ls | grep -v systemd-tmpfiles-setup | xargs rm -f $1


# RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 test 
# RUN  echo 'test:test' | chpasswd
RUN  echo 'root:root' | chpasswd


#RUN curl -fsSLO https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKERVERSION}.tgz \
#  && tar xzvf docker-${DOCKERVERSION}.tgz --strip 1 \
#                 -C /usr/local/bin docker/docker \
#  && rm docker-${DOCKERVERSION}.tgz

RUN rm -f /lib/systemd/system/multi-user.target.wants/* \
    /etc/systemd/system/*.wants/* \
    /lib/systemd/system/local-fs.target.wants/* \
    /lib/systemd/system/sockets.target.wants/*udev* \
    /lib/systemd/system/sockets.target.wants/*initctl* \
    /lib/systemd/system/basic.target.wants/* \
    /lib/systemd/system/anaconda.target.wants/* \
    /lib/systemd/system/plymouth* \
    /lib/systemd/system/systemd-update-utmp*

RUN curl -fsSL -O https://github.com/kubeedge/kubeedge/releases/download/${KEADMVERSION}/keadm-${KEADMVERSION}-linux-amd64.tar.gz \
  && tar xvzf keadm-${KEADMVERSION}-linux-amd64.tar.gz --strip 1 \
  && mv  ./keadm/keadm /usr/local/bin/keadm \ 
  && rm keadm-${KEADMVERSION}-linux-amd64.tar.gz

#COPY . /resources

RUN service ssh start

#RUN echo "#!/bin/bash\n\$@" > /usr/bin/sudo
#RUN chmod +x /usr/bin/sudo

EXPOSE 22

VOLUME [ "/sys/fs/cgroup" ]

#CMD ["/usr/sbin/sshd","-D"]
CMD ["/lib/systemd/systemd"]
