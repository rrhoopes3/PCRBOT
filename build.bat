@echo off
echo === Multi PCR Dashboard Build ===
echo.
echo Installing dependencies...
pip install -r requirements_desktop.txt
echo.
echo Building executable...
pyinstaller desktop_app.spec --noconfirm
echo.
echo Build complete!
echo Output: dist\MultiPCRDashboard\MultiPCRDashboard.exe
pause
