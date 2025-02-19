# Build stage
FROM public.ecr.aws/lambda/python:3.9 AS builder

# Install dependencies
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --target . && \
    find . -type d -name "tests" -exec rm -rf {} + && \
    find . -type d -name "examples" -exec rm -rf {} + && \
    rm -rf *.dist-info

# Copy only the necessary model files and source code
COPY saved_model/ ./saved_model/
COPY jam.py ./

# Final stage
FROM public.ecr.aws/lambda/python:3.9

# Copy dependencies and application files from builder
COPY --from=builder /var/task/. .

# Set the CMD to your handler
CMD [ "jam.lambda_handler" ]
