FROM python:3.13
WORKDIR /simple_gospels_agent
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
COPY src/ .
CMD ["python", "gospels_agent.py"]