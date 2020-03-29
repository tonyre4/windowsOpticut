If exist obf (rd obf /s /q)
copy src\a.py input\
python Intensio-Obfuscator\intensio\intensio_obfuscator.py --input input --output obf -rth
copy src\main.py obf\
cd obf
pyinstaller --icon=..\src\dmm.ico main.py
cd dist\main
REM mkdir Cbc-2.7.5-win32-cl15icl11.1
REM cd Cbc-2.7.5-win32-cl15icl11.1
REM xcopy ..\..\..\..\loadout\Cbc-2.7.5-win32-cl15icl11.1 . /s/h/e/k/f/c
copy ..\..\..\src\cbc.exe .
REM cd..
main.exe ..\..\..\csv\chico.csv
cd ..\..\..
