import os

ffmpegPath = r'C:\FFmpeg\bin\ffmpeg.exe'

ffprobePath = r'C:\FFmpeg\bin\ffprobe.exe'

from converter import Converter

conv = Converter(ffmpegPath, ffprobePath)

info = conv.probe(os.getcwd() + r'\datasets\videos\A1.avi')

convert = conv.convert(os.getcwd() + r'\datasets\videos\A1.avi', os.getcwd() + r'\datasets\videos\mp4\A1.mp4', {
    'format': 'mp4',
    'audio': {
        'codec': 'aac',
        'samplerate': 11025,
        'channels': 2
    },
    'video': {
        'codec': 'hevc',
        'width': 720,
        'height': 400,
        'fps': 25
    }})

for timecode in convert:
    print(f'\rConverting ({timecode:.2f}) ...')