# Build stage
FROM public.ecr.aws/lambda/python:3.9 AS builder

# Install system dependencies for sentencepiece
RUN yum install -y gcc g++ make curl && \
    yum clean all

# Download PyTorch wheel first
RUN mkdir -p /tmp/wheels && \
    curl -L -o /tmp/wheels/torch-2.1.0-cp39-cp39-linux_x86_64.whl \
    https://download.pytorch.org/whl/cpu/torch-2.1.0%2Bcpu-cp39-cp39-linux_x86_64.whl

# Install dependencies
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /tmp/wheels/torch-2.1.0-cp39-cp39-linux_x86_64.whl && \
    pip install --no-cache-dir \
        "tqdm>=4.27" \
        "regex!=2019.12.17" \
        "requests>=2.31.0" \
        "packaging>=20.0" \
        "filelock>=3.0.0" \
        "pyyaml>=5.1" \
        "huggingface-hub>=0.16.4" \
        "tokenizers>=0.14,<0.19" \
        "safetensors>=0.3.1" \
        "numpy>=1.17,<2.0" \
        "accelerate>=0.27.0" \
        "sentencepiece==0.1.99" \
        "uvicorn" && \
    pip install --no-cache-dir -r requirements.txt --target . && \
    find . -type d -name "tests" -exec rm -rf {} + && \
    find . -type d -name "examples" -exec rm -rf {} + && \
    rm -rf *.dist-info

# Copy only the necessary model files and source code
COPY saved_model/ ./saved_model/
COPY jam.py ./

# Final stage
FROM public.ecr.aws/lambda/python:3.9

# Install system dependencies for sentencepiece
RUN yum install -y gcc g++ make curl && \
    yum clean all

# Copy PyTorch wheel from builder
COPY --from=builder /tmp/wheels /tmp/wheels

# Install required packages in the final stage
RUN pip install --no-cache-dir /tmp/wheels/torch-2.1.0-cp39-cp39-linux_x86_64.whl && \
    pip install --no-cache-dir \
        "tqdm>=4.27" \
        "regex!=2019.12.17" \
        "requests>=2.31.0" \
        "packaging>=20.0" \
        "filelock>=3.0.0" \
        "pyyaml>=5.1" \
        "huggingface-hub>=0.16.4" \
        "tokenizers>=0.14,<0.19" \
        "safetensors>=0.3.1" \
        "numpy>=1.17,<2.0" \
        "accelerate>=0.27.0" \
        "sentencepiece==0.1.99" \
        "uvicorn"

# Copy dependencies and application files from builder
COPY --from=builder /var/task/. .

# Set the CMD to your handler
CMD [ "jam.lambda_handler" ]
