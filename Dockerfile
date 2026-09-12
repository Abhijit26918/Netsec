FROM python:3.10-slim

WORKDIR /app

# Copy only what pip needs first (manifest + the package "-e ." installs) so that
# editing app.py/templates/etc. later doesn't invalidate this layer and force a
# full dependency reinstall on every rebuild.
COPY requirements.txt setup.py ./
COPY networksecurity ./networksecurity

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Now bring in everything else (app.py, templates/, final_model/, etc.)
COPY . .

# Don't run the app as root inside the container. The app writes logs/,
# Artifacts/, etc. at runtime via os.makedirs(), so appuser needs ownership
# of /app, not just read access.
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["python", "app.py"]
