FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Clone llama.cpp
WORKDIR /app
RUN git clone https://github.com/ggerganov/llama.cpp.git

# Build
WORKDIR /app/llama.cpp
RUN make

# Default command
CMD ["./main", "-m", "/models/mistral.gguf", "-p", "Hello"]
