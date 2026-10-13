@echo off
REM YuE2 web arayuzu: http://127.0.0.1:7870
REM   frontend\start.bat                     model sunucusuna (127.0.0.1:7861) baglanir
REM   frontend\start.bat --mock              modelsiz test sunucusunu da baslatir (kisa bir deneme tonu uretir)
REM   frontend\start.bat --api http://ADRES:7861 --port 8000
REM   frontend\start.bat --host 0.0.0.0 --password PAROLA      aga/disariya acar, parola ister
REM Parola ozel karakter iceriyorsa komut satiri yerine YUE2_WEB_PASSWORD ortam degiskeniyle verin.
REM Tum secenekler: frontend\start.bat --help
setlocal EnableExtensions
REM YUE2_ENV, diger script'lerdeki gibi yue-ui klasorune gore cozulur.
pushd "%~dp0.."
if not defined YUE2_ENV set "YUE2_ENV=..\yue_ortam"
for %%p in ("%YUE2_ENV%") do set "ENV_PY=%%~fp\Scripts\python.exe"
popd
cd /d "%~dp0"
set "PYTHONUTF8=1"
REM Site yalnizca Python'un standart kutuphanesini kullanir; mock sunucusu ise yue_ortam'daki fastapi + uvicorn'u.
if not defined PYTHON (
    if exist "%ENV_PY%" (set "PYTHON=%ENV_PY%") else set "PYTHON=python"
)

set "MOCK=0"
set "ARGS="
:parse
if "%~1"=="" goto :parsed
if /i "%~1"=="--mock" set "MOCK=1" & shift & goto :parse
set ARGS=%ARGS% %1
shift
goto :parse
:parsed

if "%MOCK%"=="0" goto :web
REM Ayri port: ayni anda calisan gercek model sunucusuyla (7861) cakismaz.
if not defined YUE2_MOCK_PORT set "YUE2_MOCK_PORT=7862"
if not exist "%ENV_PY%" (
    echo Mock sunucusu yue_ortam ortaminda calisir: %ENV_PY%
    echo Once kurulumu calistirin: setup_windows.bat
    exit /b 1
)
if not exist ..\logs mkdir ..\logs
start "YuE2 mock" /b "%ENV_PY%" ..\server\yue2_server.py --mock --port %YUE2_MOCK_PORT% --outputs ..\outputs\mock > ..\logs\mock_server.log 2>&1
set /a TRIES=0
:wait_mock
"%ENV_PY%" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:%YUE2_MOCK_PORT%/health', timeout=1)" >nul 2>&1 && goto :mock_ready
set /a TRIES+=1
if %TRIES% geq 60 (
    echo Mock sunucusu baslamadi; port %YUE2_MOCK_PORT% dolu olabilir, YUE2_MOCK_PORT ile degistirin. Log:
    type ..\logs\mock_server.log
    exit /b 1
)
ping -n 2 127.0.0.1 >nul
goto :wait_mock
:mock_ready
echo Mock model sunucusu: http://127.0.0.1:%YUE2_MOCK_PORT% - log: logs\mock_server.log, ciktilar: outputs\mock
set ARGS=--api http://127.0.0.1:%YUE2_MOCK_PORT% %ARGS%

:web
"%PYTHON%" server.py %ARGS%
set "CODE=%ERRORLEVEL%"
if "%MOCK%"=="1" call :stop_mock
exit /b %CODE%

REM Ctrl+C mock sunucusunu da kapatir; site baska bir nedenle kapanirsa portu dinleyen sureci kapatir.
:stop_mock
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /r /c:":%YUE2_MOCK_PORT% .*LISTENING"') do taskkill /pid %%p /f >nul 2>&1
exit /b 0
