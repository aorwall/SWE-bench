FROM ubuntu:jammy

ARG TARGETARCH

# Install third party tools
RUN apt-get update && \
    apt-get install -y bash gcc git jq wget g++ make libffi-dev build-essential python3.11 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Initialize git
RUN git config --global user.email "swebench@pnlp.org"
RUN git config --global user.name "swebench"

# Create new user
RUN useradd -ms /bin/bash swe-bench
USER swe-bench
WORKDIR /home/swe-bench
RUN chown -R swe-bench:swe-bench /home/swe-bench

# Setup Conda
ENV PATH="/home/swe-bench/miniconda3/bin:${PATH}"
ARG PATH="/home/swe-bench/miniconda3/bin:${PATH}"

COPY docker/getconda.sh .
RUN bash getconda.sh ${TARGETARCH} \
    && rm getconda.sh \
    && mkdir ~/.conda \
    && bash miniconda.sh -b \
    && rm -f miniconda.sh
RUN conda --version \
    && conda init bash \
    && conda config --append channels conda-forge

# Install python packages
RUN pip install datasets python-dotenv gitpython unidiff rich importlib

# Install SWE-Bench
COPY . .
RUN pip install -e .

COPY scripts/*.sh .

CMD ["/bin/bash"]
