@echo off
chcp 65001 >nul
title Insan Tespit Demo Uygulamasi

echo ==================================================
echo Insan Tespit Demo Uygulamasi Baslatiliyor
echo ==================================================

REM Proje klasorune gec
cd /d "%~dp0"

REM Sanal ortam kontrolu
if not exist "venv\Scripts\activate.bat" (
    echo Sanal ortam bulunamadi. Yeni sanal ortam olusturuluyor...
    python -m venv venv
)

REM Sanal ortami aktif et
call venv\Scripts\activate.bat

REM Gerekli paketleri yukle
echo Gerekli kutuphaneler kontrol ediliyor/yukleniyor...
pip install -r requirements.txt

REM Uygulamayi baslat
echo ==================================================
echo Uygulama baslatiliyor...
echo Tarayicidan su adresi acabilirsiniz:
echo http://127.0.0.1:5000
echo ==================================================

python app.py

pause