cls
author:StickCui
echo off
title XDUthesis
:STARTS
cls
echo.         XDUthesis Windows CMD ����
echo.
echo. ==============================================
echo. 
echo. 1. ����ģ�������ļ��Լ�ģ��˵���ĵ�
echo.
echo. 2. ���� Demo ʾ���ĵ�
echo.
echo. 3. �����м��ļ����˳�
echo.
echo. ==============================================
:CHO
set /p choice=��ѡ��Ҫ���еĲ�����Enterȷ����
if /i "%choice%"=="1" goto THEMEMANUAL
if /i "%choice%"=="2" goto DEMO
if /i "%choice%"=="3" goto DELETE
echo ѡ����Ч������������
echo.
goto CHO
:THEMEMANUAL
xelatex XDUthesis.dtx
makeindex -s gind.ist -o XDUthesis.ind XDUthesis.idx
makeindex -s gglo.ist -o XDUthesis.gls XDUthesis.glo
xelatex XDUthesis.dtx
xelatex XDUthesis.dtx
goto STARTS
:DEMO
xelatex -synctex=1 -interaction=nonstopmode Demo
bibtex Demo
xelatex -synctex=1 -interaction=nonstopmode Demo
xelatex -synctex=1 -interaction=nonstopmode Demo
goto STARTS
:DELETE
del  *.toc /s
del  *.bbl /s
del  *.blg /s
del  *.out  /s
del  *.aux  /s
del  *.log  /s
del  *.bak  /s
del  *.thm  /s
del  *.synctex.gz /s
del  *.fdb_latexmk /s
del  *.fls /s
del  *.glo /s
del  *.gls /s
del  *.idx /s
del  *.ilg /s
del  *.ind /s
del  *.nav /s
del  *.snm /s
del  *.ins /s
del  *.xdv /s