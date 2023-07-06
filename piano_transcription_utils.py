import os
from glob import glob
from IPython.display import Audio
from IPython.display import Image
import soundfile as sf
import numpy as np
import scipy
import librosa
import logging
import torch
import torchaudio
from pretty_midi import PrettyMIDI
from torchaudio.transforms import Fade
from torchaudio.utils import download_asset
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
from piano_transcription_inference import PianoTranscription, sample_rate, load_audio



def make_transcription(audio_path,output_path=str,device = "cpu"):
    '''
    PTI
    #################
    Transcribes piano - audiofile to midi.
    audio_path: path to audio data
    output_path: location to save output to.
    device: "cpu" or "cuda"
    returns: Nothing, saves .mid file to disk
    '''
    # Load audio
    (audio, _) = load_audio(audio_path, sr=sample_rate, mono=True)
    # Transcriptor
    transcriptor = PianoTranscription(device=device, checkpoint_path=None)  # device: 'cuda' | 'cpu'
    #Transcribe and write out to MIDI file
    transcribed_dict = transcriptor.transcribe(audio, output_path)
    logging.info(f"Transcribed {os.path.basename(output_path)} and saved to disk.")

def initialize_demucs():
    '''
    Demucs initialization, run it to load model.
    '''
    try:
        from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
        from mir_eval import separation
        logging.info("Initialized Demucs successfully")
    except ModuleNotFoundError:
        logging.error("Make sure you installed demucs correctly!")

def separate_sources(
        model,
        mix,
        segment=10.,
        overlap=0.1,
        device=None,
        sample_rate = 44100
):
    """
    DEMUCS
    #################
    Apply model to a given mixture. Use fade, and add segments together in order to add model segment by segment.

    Args:
        segment (int): segment length in seconds
        device (torch.device, str, or None): if provided, device on which to
            execute the computation, otherwise `mix.device` is assumed.
            When `device` is different from `mix.device`, only local computations will
            be on `device`, while the entire tracks will be stored on `mix.device`.
    """
    if device is None:
        device = mix.device
    else:
        device = torch.device(device)
    #_

    batch, channels, length = mix.shape

    chunk_len = int(sample_rate * segment * (1 + overlap))
    start = 0
    end = chunk_len
    overlap_frames = overlap * sample_rate
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape='linear')

    final = torch.zeros(batch, len(model.sources), channels, length, device=device)

    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        with torch.no_grad():
            out = model.forward(chunk)
        out = fade(out)
        final[:, :, :, start:end] += out
        if start == 0:
            fade.fade_in_len = int(overlap_frames)
            start += int(chunk_len - overlap_frames)
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0
    return final    
    
    
def separate_audio(file, sr = None,filename=None, source='piano', device='cpu', save_to_disk = False,out_path=None):
    '''
    DEMUCS
    ######################
    Separates piano from audiofile and outputs file to "./demucs_out/file.wav"
    file: filepath as string or waveform as tensor
    sr : sample rate
    filename: filename if only data is passed to file    
    source: 'piano' or 'bass' 
    device: torch-device, default is cpu, if available put 'cuda'
    save_to_disk: bool, if output should be data for further use or saved to disk
    out_path: path to write files to if save_to_disk = True
    returns: output-path to separated piano wav-file / waveform and sample rate
    '''
    # setup model
    bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
    model = bundle.get_model()
    if device != 'cpu':
        device = device
    # put "cuda" if available
    device = torch.device(device)
    model.to(device)
   
    # separate piano 
    #from file
    if type(file) == str:
        song = file
        waveform, sample_rate = torchaudio.load(song)
        logging.info(f"Separating track from file {file}...")
    #from waveform
    else:
        song = filename
        waveform = file
        sample_rate = sr
        logging.info(f"Separating track from local data ...")
        
    waveform = waveform.to(device)
    mixture = waveform
    # parameters
    segment: int = 10
    overlap = 0.1

    

    ref = waveform.mean(0)
    waveform = (waveform - ref.mean()) / ref.std()  # normalization

    sources = separate_sources(
        model,
        waveform[None],
        device=device,
        segment=segment,
        overlap=overlap,
        sample_rate = sample_rate
    )[0]
    sources = sources * ref.std() + ref.mean()

    sources_list = model.sources
    sources = list(sources)

    audios = dict(zip(sources_list, sources))
    logging.info("...finished.")
    N_FFT = 4096
    N_HOP = 4
    stft = torchaudio.transforms.Spectrogram(
        n_fft=N_FFT,
        hop_length=N_HOP,
        power=None,
    )


    segment_start = 0
    segment_end = 59

    frame_start = segment_start * sample_rate
    frame_end = segment_end * sample_rate

    drums_spec = audios["drums"][:, frame_start: frame_end].cpu()

    bass_spec = audios["bass"][:, frame_start: frame_end].cpu()

    vocals_spec = audios["vocals"][:, frame_start: frame_end].cpu()

    other_spec = audios["other"][:, frame_start: frame_end].cpu()
    mix_spec = mixture[:, frame_start: frame_end].cpu()
    if save_to_disk:
        if source == 'piano':
            output_path = os.path.join(out_path,'demucs',os.path.basename(song))
            torchaudio.save(output_path, other_spec,sample_rate)
            logging.info(f'Saved file to: {output_path} .')
        elif source == 'bass':
            output_path = f"./audio/src_sep/{song[:-4]}_demucs_bass.wav"
            torchaudio.save(output_path, bass_spec,sample_rate)
        else:
            raise ValueError('Unknown source, use str(piano) or str(bass)!')
        return output_path, sample_rate
    else:
        if source == 'piano':
            return other_spec,sample_rate
        elif source == 'bass':
            return bass_spec,sample_rate        
        else:
            raise ValueError('Unknown source, use str(piano) or str(bass)!')


def midi_to_audio(midi_path=str, sound_font_path=str):
    ''' Converts midi file to audio with a midi sound-font'''
    music = PrettyMIDI(midi_file=midi_path)
    waveform = music.fluidsynth(fs=44100., sf2_path = sound_font_path) #create waveform, enter sample rate as float, otherwise fluidsynth will raise an error
    return waveform, 44100.



def load_and_pan(input_file = str, pan = "1" , output_file = None, trim = False, start= 0, stop = None):
    """ Load audio data and pan (optional) or trim (optional)
    input_file: direction to input, assuming 44100 Hz
    pan : "0" for left, "1" for right, or None for no panning (usually piano is 1 (on right side) in aebersold)
    output_file : str : path of output file, sample rate and (input) filename OR waveform, samplerate and filename
    trim: boolean, set to true if you want to trim audio
    start: starting point of trimming in seconds
    stop: endpoint of trimming in seconds
    returns: waveform: torch.tensor, samplerate
    """
    
    y,sr = torchaudio.load(input_file)
    stop_sr = -1
    if stop:
        stop_sr= stop*sr
    if trim:
        y = y[:,start*sr:stop*sr]
    #overwrite unused channel, otherwise demucs will be upset(takes only stereo)
    #overwrite unused channel, otherwise demucs will be upset(takes only stereo)
    if y.shape[0] == 1:
        logging.info("File is mono, converting to stereo!")
        # copy mono channel and divide by two to keep overall energy
        y=y.repeat(2,1)
        y= torch.div(y, 2)
    if pan == '1': 
        y[0,:] = y[1,:]
    if pan == '0':
        y[1,:] = y[0,:]
    # write to file
    if output_file:
        if not os.path.isdir(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        _ = torchaudio.save(output_file, y, sr, format='wav')
        logging.info("Audio file written successfully!")
        return str(output_file), sr, input_file

    # return locally only
    else:
        return y, sr

def get_files(in_path: str,out_path:str,filetype:list,recursive_ = False):
    '''Function for loading audio files
    in_path: path to files
    out_path: path to store output later
    filetype: list of audio datatypes to look for
    recursive_: bool, searches all folder for audio files if <True> 
    returns: list of files found
    '''
    # check input path
    if not os.path.isdir(in_path):
        logging.error("Input path is not valid!")
        return None
    # check output path, if does not exist, create it
    os.makedirs(out_path,exist_ok=True)
    logging.info("Created output folders.")
    os.makedirs(os.path.join(out_path,"demucs"),exist_ok=True)
    # include subfolders
    rec = ""
    if recursive_:
        rec = "**"
        
    # get files
    fileslist = []
    for f_type in filetype:
        sublist = glob(os.path.join(in_path,rec,f"*.{f_type}"),recursive=recursive_)
        fileslist = sublist + fileslist
    logging.info(f"Found {len(fileslist)} audiofiles.")
    
    return sorted(fileslist)
