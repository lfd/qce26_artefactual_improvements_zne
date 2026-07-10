FROM python:3.12-slim

ARG USER="repro"

# System deps: LaTeX + R
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        latexmk \
        libcurl4-openssl-dev \
        libfontconfig1-dev \
        libfribidi-dev \
        libharfbuzz-dev \
        libssl-dev \
        libtiff5-dev \
        libuv1-dev \
        libxml2-dev \
        r-base \
        r-base-dev \
        sudo \
        texlive-base \
        texlive-fonts-recommended \
        texlive-latex-extra \
        texlive-luatex \
        texlive-pictures \
        texlive-publishers \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir ~/.R && \
    echo "MAKEFLAGS = -j$(($(nproc)/4))" > ~/.R/Makevars

# R packages (same set as companion paper)
RUN Rscript -e "install.packages(c('tidyverse', 'scales', 'patchwork', 'tikzDevice', 'ggnewscale'), \
    Ncpus=parallel::detectCores(), repos='https://cloud.r-project.org', clean=TRUE)"

# Python dependencies
WORKDIR /root
COPY ./reproduction/requirements.txt .
RUN pip install --no-cache-dir -r ./requirements.txt && \
    rm requirements.txt

# Unprivileged user
RUN useradd -m ${USER} && \
    echo "${USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
USER ${USER}
WORKDIR /home/${USER}

CMD [ "bash" ]
