# Piano Transcription - Project in AI SS2023 - IPC-JKU
====================================================


This is a 2-part interactive jupyter-notebook which applies state-of-the-art music information retrieval tools to audio (music) files, and helps exploring the abilities of these tools for future applications in the jazz-music domain. 

How to use: 

* Download repository

* Create conda environment with the provided requirement files with the following steps:

install conda or activate any conda environment already installed
```
pip install conda / conda activate some-env
```
install mamba (much faster in installing pkgs than conda)
```
conda install mamba
```
create new env with mamba
```
mamba create --name <env-name> --file requirements_mamba.txt -c conda-forge
```
activate env
```
conda activate <env-name>
```
install the rest of the packages not available with conda via pip
```
pip install -r requirements_pip.txt
```
run jupyter (inside project directory)
```
jupyter lab
```
done
```

`piano_transcription.ipynb`
-------------

Step-by-step piano transcription from raw audio to source separated audio to MIDI.

`evaluation.ipynb`
-------------

Evaluation notebook of the madmom downbeat tracker applied to the annotated Filosax data.

`piano_transcription_utils.py`
-------------

Utils file with all functions needed.

`readme.md`
-------------

Infos and Instructions


`report.pdf`
-------------

Project Report with more infos.




## Thanks and credits:

DEMUCS: https://github.com/facebookresearch/demucs
Spleeter(Deezer): https://github.com/deezer/spleeter
PTI: https://github.com/qiuqiangkong/piano_transcription_inference
MT-3: https://github.com/magenta/mt3
Filosax: https://dave-foster.github.io/filosax/
madmom: https://github.com/CPJKU/madmom
Francesco Foscaring @ ICP - JKU

