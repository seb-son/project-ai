# Piano Transcription - Project in AI SS2023 - IPC-JKU
====================================================


This is a 2-part interactive jupyter-notebook, where state-of-the-art music information retrieval tools are applied to audio (music) files from the jazz-music domain, with the aim of testing and evaluating these tools for future development of applications and tools, specialized on jazz-music. 

## How to use: 

* Download repository

* Create conda environment with the provided requirement files with the following steps:

- Install conda and/or activate any conda environment already installed
```
pip install conda / conda activate some-env
```
- Install mamba (much faster in installing pkgs than conda)
```
conda install mamba
```
- Create new env with mamba
```
mamba create --name <env-name> --file requirements_mamba.txt -c conda-forge
```
- Activate env
```
conda activate <env-name>
```
- Install the rest of the packages not available with conda via pip
```
pip install -r requirements_pip.txt
```
- Run jupyter (inside project directory)
```
jupyter lab
```
done.

`piano_transcription.ipynb`
-------------

Step-by-step piano transcription from raw audio, to source (piano) separated audio, to MIDI.

`evaluation.ipynb`
-------------

Evaluation notebook of the madmom downbeat tracker applied to the annotated Filosax data, as well as visual comparison of midi-piano-transcriptions from PTI and MT-3.

`piano_transcription_utils.py`
-------------

Utils file with all functions needed.


`report/Project_Report_AI.pdf`
-------------

Project Report with more infos.

`audio/**`
-------------

Example audio/midi files (for sources see report)

`README.md`
-------------

Infos and Instructions


## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0). By using, modifying, or distributing this software, you agree to comply with the terms of the GPL-3.0. This license guarantees end users the freedom to run, study, share, and modify the software. Any distributed version, whether it is the original or a derivative work, must also be licensed under the GPL-3.0 to ensure that these freedoms are preserved. For more details, refer to the LICENSE file included in this repository or visit https://www.gnu.org/licenses/gpl-3.0.html.


## Thanks and credits:

DEMUCS: 			https://github.com/facebookresearch/demucs <br>
Spleeter(Deezer): 	https://github.com/deezer/spleeter <br>
PTI: 				https://github.com/qiuqiangkong/piano_transcription_inference <br>
MT-3:				https://github.com/magenta/mt3 <br>
Filosax: 			https://dave-foster.github.io/filosax/ <br>
madmom: 			https://github.com/CPJKU/madmom <br>
Francesco Foscarin @ ICP - JKU <br>

-------------

