FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    HOST=0.0.0.0 \
    PORT=7860 \
    YOLO_CONFIG_DIR=/tmp \
    MPLCONFIGDIR=/tmp
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user ecorebook_app.py best.pt type.pt ./

EXPOSE 7860
CMD ["python", "ecorebook_app.py"]
