@echo off
REM YuE2 - Windows kurulumu (bir kez). Gerekenler: NVIDIA ekran karti + guncel surucu, internet, ~30 GB bos yer.
REM   setup_windows.bat                  Python ortamlarini kurar, eksik model agirliklarini indirir
REM   setup_windows.bat --no-sheetsage   SheetSage2 (Cover nota cikarici) kurulmaz; Cover'da "Basit analiz" secilir
REM Tekrar calistirmak guvenlidir: kurulu olanlar atlanir, eksikler tamamlanir.
REM Mac'ten kopyalanan ..\models klasoru varsa agirliklar yeniden inmez. Ortam klasorlerini (yue_ortam,
REM sheetsage_ortam, yue-ui\venv) Mac'ten kopyalamayin; Windows'ta calismazlar, burada yeniden kurulur.
REM PyTorch CUDA surumu varsayilan cu128 (RTX 20/30/40/50 serisi). Degistirmek icin once: set TORCH_CUDA=cu126
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
if not defined YUE2_ENV set "YUE2_ENV=..\yue_ortam"
if not defined YUE2_MODELS set "YUE2_MODELS=..\models"
if not defined SHEETSAGE_ENV set "SHEETSAGE_ENV=..\sheetsage_ortam"
if not defined TORCH_CUDA set "TORCH_CUDA=cu128"
set "TORCH_INDEX=https://download.pytorch.org/whl/%TORCH_CUDA%"
set "SHEETSAGE=1"
if /i "%~1"=="--no-sheetsage" set "SHEETSAGE=0"

echo === 1/5 Ekran karti
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo Uyari: nvidia-smi bulunamadi. NVIDIA ekran karti ve surucusu yoksa model CPU'da cok yavas calisir.
) else (
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
)

echo === 2/5 uv - Python ortamlarini kurar, gerekirse Python'u da kendisi indirir
where uv >nul 2>&1
if errorlevel 1 (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)
where uv >nul 2>&1
if errorlevel 1 (
    echo uv kurulamadi. Elle kurun: https://docs.astral.sh/uv/getting-started/installation/
    goto :fail
)

echo === 3/5 YuE2 ortami: %YUE2_ENV%
set "PY=%YUE2_ENV%\Scripts\python.exe"
call :check_copied "%YUE2_ENV%" || goto :fail
if not exist "%PY%" (
    uv venv --python 3.12 "%YUE2_ENV%" || goto :fail
)
uv pip install --python "%PY%" torch==2.10.0 torchaudio==2.10.0 --index-url %TORCH_INDEX% || goto :fail
REM Kurulu CUDA'li torch yue2'nin torch==2.10.0 sartini karsilar; PyPI'daki CPU surumuyle degistirilmez.
uv pip install --python "%PY%" "wheels\yue2_infer-0.1.5-py3-none-any.whl" fastapi==0.136.1 uvicorn==0.46.0 librosa "julius>=0.2.3" dora-search einops pyyaml omegaconf "gradio>=5.20,<6" requests || goto :fail
uv pip install --python "%PY%" --no-deps demucs==4.0.1 openunmix || goto :fail
"%PY%" -c "import torch, yue2, demucs.pretrained, librosa, gradio; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '| yue2', yue2.__version__)" || goto :fail

echo === 4/5 Model agirliklari: %YUE2_MODELS%
set "SHEETSAGE_FLAG="
if "%SHEETSAGE%"=="1" set "SHEETSAGE_FLAG=--sheetsage"
"%PY%" download_models.py "%YUE2_MODELS%" %SHEETSAGE_FLAG% || goto :fail

if "%SHEETSAGE%"=="0" goto :done
echo === 5/5 SheetSage2 ortami: %SHEETSAGE_ENV% - torch 2.8 istedigi icin ayri, Python 3.11
set "SPY=%SHEETSAGE_ENV%\Scripts\python.exe"
call :check_copied "%SHEETSAGE_ENV%" || goto :fail
if not exist "%SPY%" (
    uv venv --python 3.11 "%SHEETSAGE_ENV%" || goto :sheetsage_fail
)
uv pip install --python "%SPY%" torch==2.8.0 torchaudio==2.8.0 --index-url %TORCH_INDEX% || goto :sheetsage_fail
uv pip install --python "%SPY%" -r "%YUE2_MODELS%\SheetSage2\requirements.txt" || goto :sheetsage_fail
"%SPY%" -c "import torch, transformers; print('SheetSage2 hazir | torch', torch.__version__, '| CUDA', torch.cuda.is_available())" || goto :sheetsage_fail
goto :done

:sheetsage_fail
echo.
echo Uyari: SheetSage2 kurulamadi. Text to Song ve Cover yine calisir; Cover'da Nota cikarici: Basit analiz secin.

:done
echo.
echo Kurulum tamam. Iki ayri pencerede baslatin:
echo   start_model_server.bat --preload
echo   start_ui_windows.bat       Gradio arayuzu, http://127.0.0.1:7860
echo   ya da frontend\start.bat   web arayuzu, http://127.0.0.1:7870
exit /b 0

:fail
echo.
echo Kurulum yarida kaldi; yukaridaki hatayi kontrol edin. Sorun giderildikten sonra tekrar calistirabilirsiniz.
exit /b 1

REM Mac/Linux'tan kopyalanmis bir ortam (bin\ var, Scripts\python.exe yok) Windows'ta calismaz.
:check_copied
if exist "%~1\bin\" if not exist "%~1\Scripts\python.exe" (
    echo %~1 Mac/Linux'tan kopyalanmis bir ortam; Windows'ta calismaz.
    echo Silip tekrar calistirin: rmdir /s /q "%~1"
    exit /b 1
)
exit /b 0
