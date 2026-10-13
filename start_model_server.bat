@echo off
REM YuE2 model sunucusu: http://127.0.0.1:7861 (yalnizca bu bilgisayardan erisilebilir)
REM   start_model_server.bat             yerel agirliklarla, tamamen cevrimdisi
REM   start_model_server.bat --mock      modelsiz test modu
REM   start_model_server.bat --preload   modeli baslangicta yukle
REM Bellek ayari ekran kartinin bellegine gore otomatik secilir (--budget auto):
REM   20 GB altinda --offload-ar. 8-12 GB kartlarda --phase-offload: her asamada modelin yalnizca o asamanin
REM   kullandigi yarisi kartta, diger yarisi RAM'de bekler (kalite degismez). Yedek olarak --shared-gpu-memory
REM   de acilir: yine sigmayan kisim (uzun sarki + CFG) paylasimli GPU bellegine (sistem RAM'i) konur; yavaslar
REM   ama hata vermez. NVIDIA surucusu 536.40 ya da yenisi olmali ve NVIDIA Denetim Masasi'ndaki
REM   "CUDA - Sysmem Fallback Policy" Driver Default kalmali.
REM Elle vermek icin: start_model_server.bat --budget 16 --offload-ar
REM Ilk kurulum: setup_windows.bat
setlocal EnableExtensions
cd /d "%~dp0"
if not defined YUE2_ENV set "YUE2_ENV=..\yue_ortam"
if not defined YUE2_MODELS set "YUE2_MODELS=..\models"
set "PY=%YUE2_ENV%\Scripts\python.exe"
REM Python dosyalara ve loglara UTF-8 yazsin; yoksa Turkce karakterler Windows'un kod sayfasina sigmaz.
set "PYTHONUTF8=1"
if not exist "%PY%" (
    echo Python ortami bulunamadi: %PY%
    echo Once kurulumu calistirin: setup_windows.bat
    exit /b 1
)
set "MOCK=0"
for %%a in (%*) do if /i "%%~a"=="--mock" set "MOCK=1"
if "%MOCK%"=="1" goto :mock
if not exist "%YUE2_MODELS%\YuE2-3B\model.safetensors" goto :no_weights
if not exist "%YUE2_MODELS%\YuE2-Vae\model.safetensors" goto :no_weights
REM Hugging Face'e hicbir istek gitmesin.
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "HF_HUB_DISABLE_TELEMETRY=1"
REM --budget auto once gelir; komut satirinda verilen --budget onu ezer.
"%PY%" server\yue2_server.py --model "%YUE2_MODELS%\YuE2-3B" --vae "%YUE2_MODELS%\YuE2-Vae" --offline --budget auto %*
exit /b

:mock
"%PY%" server\yue2_server.py %*
exit /b

:no_weights
echo Model agirliklari bulunamadi: %YUE2_MODELS%\YuE2-3B ve %YUE2_MODELS%\YuE2-Vae
echo setup_windows.bat ile indirin ya da Mac'teki models klasorunu yueModel\models olarak kopyalayin.
exit /b 1
