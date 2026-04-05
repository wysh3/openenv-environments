FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install package
RUN pip install --no-cache-dir -e .

ENV PYTHONPATH=/app
ENV PORT=7860

# Expose port
EXPOSE 7860

# Run server
CMD ["python", "-m", "constraint_solver.server.app"]
