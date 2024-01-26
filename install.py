import pip

_all_ = [
    "av==11.0.0",
    "filelock==3.9.0",
    "fsspec==2023.4.0",
    "imageio==2.32.0",
    "imageio-ffmpeg==0.4.9",
    "mpmath==1.3.0",
    "networkx==3.0",
    "psutil==5.9.6",
    "PyQt5==5.15.10",
    "sympy==1.12",
    "gymnasium==0.29.1",
    "scikit-learn==1.3.2",
    "casadi==3.6.4",
    "pandas==2.1.3",
    "sphinxcontrib-napoleon==0.7",
    "sphinx==7.2.6",
    "myst-parser==2.0.0",
    "sphinx-rtd-theme==1.3.0",
    "linkify-it-py==2.0.2",
    "nbsphinx==0.9.3",
    "tqdm==4.66.1",
    "control==0.9.4",
    "future==0.18.3",
    "timeout-decorator==0.5.0",
    "nltk==3.8.1",
    "path==16.7.1",
    "gym-electric-motor==2.0.0",
    "chardet==5.2.0",
    "influxdb-client==1.39.0",
    "asyncua==1.0.6"
    ]

windows = []

linux = [
    "torch==2.1.0+cpu",
    "torchaudio==2.1.0+cpu",
    "torchvision==0.16.0+cpu",
    ]

# MacOS
darwin = [
    "torch==2.1.1",
    "torchaudio==2.1.1",
    "torchvision==0.16.1",
    ]

def install(packages):
    for package in packages:
        if "torch" in package:
            pip.main(['install', package, "-f", "https://download.pytorch.org/whl/torch_stable.html"])
        else:
            pip.main(['install', package])

if __name__ == '__main__':

    from sys import platform

    install(_all_) 
    if platform == 'windows':
        install(windows)
    if platform.startswith('linux'):
        install(linux)
    if platform == 'darwin': # MacOS
        install(darwin)