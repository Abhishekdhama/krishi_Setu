@echo off
REM Set Ollama to use D: drive for models
setx OLLAMA_MODELS "D:\ollama_models"

echo Environment variable set successfully!
echo Please RESTART your computer for changes to take effect.
echo.
echo After restart, Ollama will use D:\ollama_models
pause
