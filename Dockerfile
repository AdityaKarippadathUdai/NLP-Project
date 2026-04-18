FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp.git

# Build using CMake
WORKDIR /app/llama.cpp
RUN cmake -B build
RUN cmake --build build --config Release

# Copy model INTO image (⚠️ heavy)
WORKDIR /models
COPY models/mistral-7b-instruct-v0.2.Q4_K_M.gguf /models/mistral.gguf

# Run model
WORKDIR /app/llama.cpp/build/bin
CMD ["./llama-cli", "-m", "/models/mistral.gguf", "-p", "Explain NLP pipeline"]
