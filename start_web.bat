@echo off
setlocal
chcp 65001 >nul
call "%~dp0TCN\adding_problem\启动用户侧储能排查服务.bat" %*
exit /b %ERRORLEVEL%
