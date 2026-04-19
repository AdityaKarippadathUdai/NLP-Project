FROM ubuntu:22.04

# ============================
# System dependencies
# ============================
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ============================
# Set working directory
# ============================
WORKDIR /app

# ============================
# Copy project files
# ============================
COPY . /app

# ============================
# Install Python dependencies
# ============================
RUN pip install --no-cache-dir \
    gradio \
    torch \
    transformers \
    sentencepiece \
    accelerate \
    nltk \
    spacy \
    wikipedia-api \
    requests \
    beautifulsoup4 \
    lxml \
    ddgs \
    numpy \
    scikit-learn \
    llama-cpp-python \
    google-generativeai

# ============================
# Download NLP resources
# ============================

# NLTK data
RUN python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# spaCy model
RUN python3 -m spacy download en_core_web_sm

# ============================
# Clone & build llama.cpp
# ============================
RUN git clone https://github.com/ggerganov/llama.cpp.git

WORKDIR /app/llama.cpp
RUN cmake -B build
RUN cmake --build build --config Release

# ============================
# Copy GGUF model
# ============================
WORKDIR /models
COPY models/mistral-7b-instruct-v0.2.Q4_K_M.gguf /models/mistral.gguf

# ============================
# Expose Gradio port
# ============================
EXPOSE 7860

# ============================
# Run Gradio interface
# ============================
WORKDIR /app
CMD ["python3", "interface.py"]