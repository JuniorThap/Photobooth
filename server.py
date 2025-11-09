from flask import Flask, send_from_directory, jsonify, render_template_string
from pathlib import Path

app = Flask(__name__)

PHOTO_DIR = Path("resource/output")
PHOTO_DIR.mkdir(parents=True, exist_ok=True)

@app.route("/<folder>/<filename>")
def serve_file(folder, filename):
    folder_path = PHOTO_DIR / folder
    file_path = folder_path / filename
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404
    return send_from_directory(folder_path, filename)

@app.route("/<folder>/")
def serve_folder(folder):
    folder_path = PHOTO_DIR / folder
    if not folder_path.exists() or not folder_path.is_dir():
        return jsonify({"error": f"Folder not found: {folder_path}"}), 404

    files = sorted([f.name for f in folder_path.iterdir() if f.is_file()])

    html = """
    <html>
    <head>
        <title>{{ folder }}</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                background:#fafafa;
                font-family:sans-serif;
                text-align:center;
                margin:0;
                padding:0;
                display:flex;
                flex-direction:column;
                align-items:center;
            }

            h2 {
                margin:25px 0 10px;
                color:#222;
                text-align:center;
            }

            .media-container {
                width:100%%;
                max-width:480px;
                display:flex;
                flex-direction:column;
                align-items:center;
                gap:25px;
                margin-bottom:40px;
            }

            img, video {
                width: 100%;
                max-width: 480px;
                height: auto;
                display: block;
                margin: 0 auto;
                background: #ddd;
                border:none;
                border-radius:0;
                object-fit: contain;
                aspect-ratio: 4 / 3;
            }

            a.file-link {
                display:block;
                text-align:center;
                margin-top:8px;
                color:#007bff;
                font-weight:600;
                text-decoration:none;
            }

            .download-all {
                width:100%%;
                max-width:480px;
                background:#007bff;
                color:white;
                font-size:22px;
                font-weight:bold;
                padding:18px;
                border:none;
                border-radius:12px;
                text-decoration:none;
                margin-bottom:80px;
                box-shadow:0 4px 10px rgba(0,0,0,0.2);
            }

            .download-all:active {
                background:#0056b3;
                transform:scale(0.98);
            }
        </style>

        <script>
            function downloadAll() {
                const files = {{ files|tojson }};
                const folder = "{{ folder }}";
                for (const f of files) {
                    const a = document.createElement('a');
                    a.href = `/${folder}/${f}`;
                    a.download = f;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                }
            }
        </script>
    </head>
    <body>
        <h2>📸 Session: {{ folder }}</h2>

        {% if not files %}
            <p>No files found in this session.</p>
        {% else %}
            <div class="media-container">
                {% for f in files %}
                    {% if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg') %}
                        <img src="/{{ folder }}/{{ f }}">
                    {% elif f.endswith('.mp4') %}
                        <video controls src="/{{ folder }}/{{ f }}"></video>
                    {% endif %}
                    <a class="file-link" href="/{{ folder }}/{{ f }}" download>Download {{ f }}</a>
                {% endfor %}
            </div>
        {% endif %}

        <button class="download-all" onclick="downloadAll()">⬇️ Download All</button>
    </body>
    </html>
    """
    return render_template_string(html, folder=folder, files=files)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
