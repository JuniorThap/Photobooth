@echo off
echo === Starting backend server ===
start cmd /k "python server.py"

echo === Starting ngrok ===
start cmd /k "ngrok http 5000"

echo Waiting for ngrok to generate URL...
timeout /t 5 >nul

echo === Fetching ngrok URL ===
FOR /F "tokens=4" %%i IN ('curl -s http://127.0.0.1:4040/api/tunnels ^| findstr /i "public_url"') DO (
    set URL=%%i
)

echo Detected ngrok URL: %URL%

echo === Launching Photobooth ===
python main.py --server_url %URL% --camera_index 0
