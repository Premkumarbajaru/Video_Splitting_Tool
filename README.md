# Video Splitting Tool 🎬

A professional AI-powered video processing tool with a beautiful web interface for transcription, segmentation, and clip generation.

## Features ✨

- 🎥 Download videos from YouTube and other platforms
- 🎵 Extract and separate audio/vocals using Demucs
- 📝 AI-powered transcription with Whisper
- ✂️ Automatic video segmentation
- 🎬 Generate clips based on segments
- 🌐 Beautiful modern web interface
- 📊 Real-time progress tracking

## Installation 🚀

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install FFmpeg (required):
   - Windows: Download from https://ffmpeg.org/download.html
   - Linux: `sudo apt install ffmpeg`
   - macOS: `brew install ffmpeg`

## Usage 💻

### Web Interface (Recommended)

1. Start the web server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Enter a YouTube URL or local video path and click "Start Processing"

### Command Line

```bash
python main.py --url "https://youtube.com/watch?v=..."
# OR
python main.py --input_video "path/to/video.mp4"
```

## Configuration ⚙️

Edit `config.yaml` to customize:

- Transcription model size (tiny, small, medium, large)
- Silence removal parameters
- Maximum segment duration
- Output directories

## Output 📁

Processed files are saved in:
- `processed/audio/` - Extracted audio files
- `processed/vocals/` - Separated vocals
- `processed/transcripts/` - JSON transcripts
- `processed/clips/` - Generated video clips

## Tech Stack 🛠️

- **Backend**: Flask, Python
- **Frontend**: HTML5, CSS3, JavaScript
- **AI Models**: Whisper (transcription), Demucs (audio separation)
- **Video Processing**: FFmpeg, yt-dlp

## License 📄

MIT License
