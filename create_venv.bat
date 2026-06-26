@echo off

REM Set the desired environment name
set VENV_NAME=venv

REM Set the folder path based on the current batch file's directory
set "FOLDER_PATH=%~dp0%VENV_NAME%"

REM Check if the folder exists
if exist "%FOLDER_PATH%" (
    REM Delete the folder
    rmdir /s /q "%FOLDER_PATH%"
    echo Folder deleted.
) else (
    echo Folder does not exist.
)

REM Create the virtual environment specifically with Python 3.12
py -3.12 -m venv %VENV_NAME%

REM Activate the virtual environment
call %VENV_NAME%\Scripts\activate

REM Install packages from requirements.txt using local repository
pip install --no-index --find-links=WindowsInstallRequirements/downloaded_packages/ -r requirements.txt

REM Deactivate the virtual environment
deactivate

echo Virtual environment setup complete.
pause