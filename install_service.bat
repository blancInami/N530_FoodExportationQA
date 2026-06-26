@echo off
REM your service name
set service_name=N530_FoodExportationQA

REM your nssm path
set nssm=WindowsInstallRequirements\nssm\win64\nssm.exe
set nssm_path=%~dp0%nssm%

REM your venv name and venv python path
set VENV_NAME=venv
set venv_py=%~dp0%VENV_NAME%\Scripts\python.exe

REM your app file path
set app=main.py
set app_path=%~dp0%app%

%nssm_path% install "%service_name%" "%venv_py%" "%app_path%"

pause