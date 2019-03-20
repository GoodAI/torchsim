# TorchSim

TorchSim is a simulation platform developed for the purpose of AGI research, but is generic enough to be used in other scenarios. The models which the simulator executes are graph-based: Nodes execute in the order defined by connections between outputs and inputs and data is passed along these connections. PyTorch is used as the backend for tensor storage and processing.

The simulation can be optionally controlled and observed via an included web-based UI. Internal and output tensors of nodes can be observed with basic tensor/matrix observers, and custom observers are written for other use-cases.

A part of this repository is an implementation of ToyArchitecture - a simple interpretable AGI model. A paper will be published and linked here later.

The original code was developed internally at GoodAI by (in alphabetical order):

Simon Andersson, Joe Davidson, Petr Dluhos, Jan Feyereisl, Petr Hlubucek, Martin Hyben, Matej Nikl, Premysl Paska, Martin Poliak, Jan Sinkora, Martin Stransky, Josef Strunc, Jaroslav Vitku

## How to install TorchSim
### Windows

#### Installation steps 
1. If you don't have Visual Studio 2017, install [Visual Studio 2017 Community](https://visualstudio.microsoft.com/downloads/). You will use its tools for building cuda kernels.
	* In `Workloads` tab check `Desktop development with C++`.
	* For VS2017: if there is error in `host_config.h`, in `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2\include\crt\host_config.h` change line 131 `#if _MSC_VER < 1600 || _MSC_VER > 1913 ` to  `#if _MSC_VER < 1600 || _MSC_VER > 1916`
2. (Recommended) reinstall [Anaconda](https://www.anaconda.com/download/) if your version is <5.2.
3. Install [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) (version 10.0)
4. Create conda environment with Python 3.7 
    1. `conda create -n torchsim python==3.7`
    2. `activate torchsim`
5. Install PyTorch 1.0+ with cuda 10.0
    * `conda install pytorch torchvision cuda100 -c pytorch`
    * `conda install cython`
6. Install libraries from project requirements
    * Navigate to directory of torchsim project `cd D:\torchsim`.
    * `pip install -r requirements.txt`
7. Install PyCharm. 
    * To be able to use python console with all required libraries from PyCharm, you should launch PyCharm from the anaconda prompt. See the next step for details.
8. For you to be able to compile CUDA from PyCharm, it has to know where your compiler is. Start PyCharm via a bat file which looks like this 
(Note that the paths might be different for you depending on your version of PyCharm, Visual Studio and Anaconda):
```
set root=%AppData%\..\Local\Continuum\anaconda3
REM newer versions of anaconda are in: root=C:\Users\<user>\Anaconda3
call %root%/Scripts/activate.bat %root%
call activate torchsim
call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64\vcvars64.bat"
REM or using VS 2017 community: 
REM call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat"
start "" /B "C:\Program Files\JetBrains\PyCharm Community Edition 2018.2.3\bin\pycharm64.exe"
REM ! check that everything executed successfully - the paths may be different on each PC !
pause 
```

### Linux
#### Installation steps 
0. Make sure you have gcc/g++ 7 and are pointing to them by default (use `update-alternatives`) to configure your system if need be.
    * `sudo apt install g++ make swig`
    * (`swig` is required by Box2D installation later)
1. Install the CUDA toolkit (version 10+)
    * Download the latest installer [here](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64)
    * Remove earlier versions of the toolkit and drivers before installation: `sudo apt purge nvidia-*`
    * Follow the instructions [here](https://askubuntu.com/questions/799184/how-can-i-install-cuda-on-ubuntu-16-04)
2. Set up the relevant paths. Add to your .bashrc:
    * `export CUDA_HOME="/usr/local/cuda"`
    * `export LD_LIBRARY_PATH="${CUDA_HOME}/lib64"`
    * `export CUDA_ROOT="/usr/local/cuda/"`
3. Download and install [Anaconda](https://www.anaconda.com/download/#linux)
    * Create new environment, install PyTorch and the packages from requirements.txt, all in the same way as on Windows.
4. For compiling custom kernels, set the default run/test location to the top level of the repository (e.g. ~/path/to/repo/torchsim)
5. Install PyCharm using snap: `sudo snap install pycharm-community --classic`
6. After PyCharm setup, try to run the unit tests.

## PyCharm setup

Because of how the cuda compilation works, you need to run tests from the root directory. In PyCharm, do the following:

0. Open project TorchSim in PyCharm, locating the root folder of your local TorchSim git repo.
1. Go to File -> Settings -> Tools -> Python Integrated Tools and set the default test runner to pytest.
2. Go to File -> Settings -> Project:TorchSim -> Project Interpreter, click the settings button on the far right -> Add... -> Conda Environment -> Existing environment -> Interpreter -> path to your conda interpreter (`C:\Users\user.name\AppData\Local\Continuum\anaconda3\envs\torchsim\python.exe` or `C:\Users\<user>\Anaconda3\envs\torchsim\python.exe` on Windows), then set it as the project interpreter.
3. Go to Run -> Edit Configurations, select Templates -> Python Tests -> pytest and choose the root directory of the TorchSim project (where this README is) as the Working Directory.
4. Add new Run Configuration: Edit Configurations -> '+' -> Python -> 
    * Name: "Run main"
    * Script path: locate file torchsim/main.py
    * Environment variables: add `PYDEVD_USE_FRAME_EVAL=NO`
    * You can choose a different than default model: add parameter e.g. `--model expert`
5. Add Run Configuration for tests: Edit Configurations -> '+' -> Python tests -> pytest ->
    * Name: "Run tests"
    * Script path: locate folder torchsim/tests
    * Set environment variable `USE_SMALL_DATASETS=1` to speed up tests significantly
6. By default, Pytorch doesn't have autocomplete hinting with Pytorch. This is a WIP by devs, but their latest efforts with the majority of
functions can be set up:
    * Get the `__init__.pyi` file from `torchsim/` and put it in your `site-packages/torch/` directory in your python installation (on Windows it is located in `C:\Users\<user>\AppData\Local\Continuum\anaconda3\envs\torchsim\Lib\site-packages\torch` or `C:\Users\<user>\Anaconda3\envs\torchsim\Lib\site-packages\torch`)

## How to run with UI

1. Run UI server by `ui_server.py`
2. Start TorchSim from PyCharm (run `main.py`)
3. Open UI in a browser window: `localhost:5000`

## UI development

UI sources are located in `/js`

Development steps:
1. Run `npm install` - Must be run whenever project dependencies (`package.json`) changes and also the first time the UI is being built. 
2. Run development build by `npm run dev` - this command never stops, watches for the file changes and incrementally builds UI.
3. When committing, build production by `npm run build` - this builds sources in production mode.

After build (2. or 3.) `Ctrl + shift + R` reload of webpage is needed.

## Troubleshooting

* If your compilation of kernels hangs (usually, the last line on the output will be `Using C:\Users\<user>\AppData\Local\Temp\torch_extensions as PyTorch extensions root...`), then go to the indicated torch_extensions folder and delete its contents.

* During communication with UI in debug mode, there are memory leaks (and errors are printed into the console), this can be fixed by adding `PYDEVD_USE_FRAME_EVAL=NO` into the environment variables (in the pycharm run configuration).

* If you are unable to import torch or numpy from PyCharm, try to import them in command line `>python` `>>>import numpy`. In case this works properly, add to your PATH environment variable `C:\Users\<user>\AppData\Local\Continuum\Anaconda3\Library\bin` or `C:\Users\<user>\Anaconda3\Library\bin` (for standard Anaconda single user installation).

## Acknowledgements

We used some code from [reactwm](https://github.com/stayradiated/reactwm) to bootstrap our UI. The license is [here](3rd-party-licenses/reactwm-license.txt).
