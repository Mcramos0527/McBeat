# McBeat AI Pipeline

Celery workers for audio analysis, clip scoring, beat matching, caption generation, and rendering.

## Key Dependencies

```
librosa
essentia
openai          # Whisper
anthropic       # Claude
opencv-python
ffmpeg-python
celery
redis
```

## Running Workers

```bash
pip install -r requirements.txt
celery -A worker worker --loglevel=info
```
