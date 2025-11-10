from flask import Flask, render_template, request, jsonify, send_file
from pathlib import Path
import json
import yaml
from src.utils import ensure_dir
from src.downloader import download
from src.extractor import extract_audio
from src.splitter import separate_vocals
from src.transcriber import transcribe
from src.cleaner import remove_silence
from src.segmenter import segment_transcript
from src.clipper import create_clips
from src.merger import find_and_merge_streams
import threading

app = Flask(__name__)
processing_status = {"status": "idle", "progress": 0, "message": ""}

def load_config():
    cfg_path = Path("config.yaml")
    default = {
        "paths": {
            "downloads": "downloads",
            "processed": "processed",
            "audio": "processed/audio",
            "vocals": "processed/vocals",
            "clips": "processed/clips",
            "transcripts": "processed/transcripts",
        },
        "transcription": {"model_size": "small", "compute_type": "int8"},
        "silence_removal": {"stop_periods": -1, "stop_threshold": "-50dB", "detection": "peak"},
        "segmentation": {"max_segment_seconds": 60},
    }
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
            for k, v in loaded.items():
                if isinstance(v, dict) and k in default:
                    default[k].update(v)
                else:
                    default[k] = v
    return default

def process_video(url, input_video):
    global processing_status
    try:
        print(f"Starting processing: url={url}, input_video={input_video}")
        processing_status = {"status": "processing", "progress": 10, "message": "Initializing..."}
        cfg = load_config()
        root = Path(__file__).parent
        paths = cfg["paths"]
        
        downloads = ensure_dir(str(root / paths["downloads"]))
        processed = ensure_dir(str(root / paths["processed"]))
        audio_dir = ensure_dir(str(root / paths["audio"]))
        vocals_dir = ensure_dir(str(root / paths["vocals"]))
        clips_dir = ensure_dir(str(root / paths["clips"]))
        transcripts_dir = ensure_dir(str(root / paths["transcripts"]))

        processing_status["progress"] = 20
        processing_status["message"] = "Downloading video..."
        print("Step: Downloading video...")
        
        if input_video:
            video_path = Path(input_video)
            if not video_path.is_absolute():
                video_path = (root / video_path).resolve()
            print(f"Using local video: {video_path}")
        elif url:
            print(f"Downloading from URL: {url}")
            video_path = download(url, downloads)
            print(f"Downloaded to: {video_path}")
        else:
            raise ValueError("No input provided")

        processing_status["progress"] = 40
        processing_status["message"] = "Extracting audio..."
        print("Step: Extracting audio...")
        audio_path = extract_audio(video_path, audio_dir)
        print(f"Audio extracted to: {audio_path}")

        processing_status["progress"] = 50
        processing_status["message"] = "Preparing audio..."
        print("Step: Skipping vocal separation (using original audio)")
        vocals_path = audio_path

        processing_status["progress"] = 60
        processing_status["message"] = "Transcribing..."
        print("Step: Transcribing...")
        tr_cfg = cfg["transcription"]
        transcript_segments = transcribe(vocals_path, tr_cfg.get("model_size", "small"), tr_cfg.get("compute_type", "int8"))
        print(f"Transcription complete: {len(transcript_segments)} segments")

        transcript_path = Path(transcripts_dir) / f"{video_path.stem}_transcript.json"
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump({"segments": transcript_segments}, f, ensure_ascii=False, indent=2)

        processing_status["progress"] = 70
        processing_status["message"] = "Removing silence..."
        print("Step: Removing silence...")
        s_cfg = cfg["silence_removal"]
        clean_audio = remove_silence(vocals_path, audio_dir, s_cfg["stop_periods"], s_cfg["stop_threshold"], s_cfg["detection"])
        print(f"Silence removed: {clean_audio}")

        processing_status["progress"] = 80
        processing_status["message"] = "Segmenting transcript..."
        print("Step: Segmenting transcript...")
        seg_cfg = cfg["segmentation"]
        segments = segment_transcript(transcript_segments, seg_cfg.get("max_segment_seconds", 60))
        print(f"Created {len(segments)} segments")

        segments_path = Path(transcripts_dir) / f"{video_path.stem}_segments.json"
        with open(segments_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        processing_status["progress"] = 90
        processing_status["message"] = "Creating clips..."
        clip_paths = create_clips(video_path, segments, clips_dir)

        processing_status = {
            "status": "completed",
            "progress": 100,
            "message": "Processing completed!",
            "clips": [str(p) for p in clip_paths],
            "transcript": str(transcript_path),
            "segments": str(segments_path)
        }
        print("Processing completed successfully")
    except Exception as e:
        print(f"ERROR in process_video: {e}")
        import traceback
        traceback.print_exc()
        processing_status = {"status": "error", "progress": 0, "message": str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    url = data.get('url')
    input_video = data.get('input_video')
    
    thread = threading.Thread(target=process_video, args=(url, input_video))
    thread.start()
    
    return jsonify({"status": "started"})

@app.route('/status')
def status():
    return jsonify(processing_status)

@app.route('/results')
def results():
    clips_dir = Path("processed/clips")
    clips = [{"name": f.name, "path": str(f)} for f in clips_dir.glob("*.mp4")] if clips_dir.exists() else []
    return jsonify({"clips": clips})

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/download/<path:filename>')
def download_file(filename):
    try:
        file_path = Path(filename)
        if file_path.exists():
            return send_file(file_path, as_attachment=True)
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download/txt/<path:filename>')
def download_txt(filename):
    try:
        file_path = Path(filename)
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            txt_content = ""
            if 'segments' in data:
                for seg in data['segments']:
                    txt_content += f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}\n\n"
            elif isinstance(data, list):
                for seg in data:
                    txt_content += f"Segment {seg['segment_id']}:\n{seg['text']}\n\n"
            
            from io import BytesIO
            buffer = BytesIO(txt_content.encode('utf-8'))
            buffer.seek(0)
            return send_file(buffer, as_attachment=True, download_name=file_path.stem + '.txt', mimetype='text/plain')
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download/pdf/<path:filename>')
def download_pdf(filename):
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import simpleSplit
        from io import BytesIO
        
        file_path = Path(filename)
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            buffer = BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            width, height = letter
            y = height - 50
            
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, y, "Transcript")
            y -= 30
            
            c.setFont("Helvetica", 10)
            
            if 'segments' in data:
                for seg in data['segments']:
                    text = f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}"
                    lines = simpleSplit(text, "Helvetica", 10, width - 100)
                    for line in lines:
                        if y < 50:
                            c.showPage()
                            y = height - 50
                            c.setFont("Helvetica", 10)
                        c.drawString(50, y, line)
                        y -= 15
                    y -= 10
            elif isinstance(data, list):
                for seg in data:
                    text = f"Segment {seg['segment_id']}: {seg['text']}"
                    lines = simpleSplit(text, "Helvetica", 10, width - 100)
                    for line in lines:
                        if y < 50:
                            c.showPage()
                            y = height - 50
                            c.setFont("Helvetica", 10)
                        c.drawString(50, y, line)
                        y -= 15
                    y -= 10
            
            c.save()
            buffer.seek(0)
            return send_file(buffer, as_attachment=True, download_name=file_path.stem + '.pdf', mimetype='application/pdf')
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/view/<path:filename>')
def view_file(filename):
    try:
        file_path = Path(filename)
        if file_path.exists():
            return send_file(file_path)
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
