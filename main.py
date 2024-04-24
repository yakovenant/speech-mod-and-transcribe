import pyrubberband
import json
import numpy as np
from pydub import AudioSegment
from faster_whisper import WhisperModel


def transcribe(file_name: str, model_size='tiny'):
    model = WhisperModel(model_size, device="cpu")
    segments, info = model.transcribe(file_name)
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    segments = list(segments)
    result = 'Transcript: '
    for seg in segments:
        result += seg.text
    with open(file_name.split('.')[0] + '_log.json', 'w') as file:
        json.dump(segments, file)
    return result


def modify(file_name: str, vol_param=None, tempo_param=None):

    def _change_vol(y, vol_param):
        return y + vol_param

    def _change_tempo(y, tempo_param):
        if tempo_param != 1:
            s = pyrubberband.time_stretch(np.array(y.get_array_of_samples()), y.frame_rate, tempo_param)
            s = np.int16(s * 2 ** 15)
            return AudioSegment(s.tobytes(), frame_rate=y.frame_rate, sample_width=2, channels=1)
        else:
            return y

    y = AudioSegment.from_wav(file_name)
    if vol_param is not None:
        y = _change_vol(y, vol_param)
    if tempo_param is not None:
        y = _change_tempo(y, tempo_param)
    y.export(file_name.split('.')[0] + '_modified.wav', format='wav')


def main(mode: int, file_name: str):
    if mode == 1:
        print('Select volume:')
        vol_param = float(input())
        print('Select tempo:')
        tempo_param = float(input())
        return modify(file_name, vol_param, tempo_param)
    elif mode == 2:
        return print(transcribe(file_name))
    else:
        raise Exception('Wrong run mode')


if __name__ == '__main__':
    print('1 - audio modification\n2 - audio transcribe\nSelect run mode:')
    run_mode = int(input())
    print('Select WAV file:')
    file_name = input()
    main(run_mode, file_name)
    print('\nDone.')
