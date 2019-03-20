echo HELP: takes the following parameters: 1) path to script (relative to torchsim) and 2) results folder, 3) the following arguments are passed to the python experiment sctipt and parsed by the parse_test_args().

if [%~1]==[] (
	SET SCRIPT_NAME=torchsim\research\research_topics\rt_3_7_1_task0_analysis\experiments\task0_analysis_experiment.py
) else (
	SET SCRIPT_NAME=%1
)

if [%~2]==[] (
	SET RESULTS_FOLDER=D:\torchsim\experiments-results\3_7_1-task0-analysis
) else (
	SET RESULTS_FOLDER=%2
)

echo /////////////////////////////////////////////////////////////////////////// preparing the environment
set root=%AppData%\..\Local\Continuum\anaconda3
call %root%/Scripts/activate.bat %root%
call activate torchsim
call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat"
setlocal
set PYTHONPATH=%~dp0


:repeat
echo /////////////////////////////////////////////////////////////////////////// calling the script
call python %SCRIPT_NAME% --save --load --alternative-results-folder=%RESULTS_FOLDER% %3 %4 %5 %6

IF %ERRORLEVEL% NEQ 0 ( 
	GOTO :repeat
)

echo /////////////////////////////////////////////////////////////////////////// done
endlocal 
pause 
