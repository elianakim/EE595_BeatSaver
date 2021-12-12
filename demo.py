from pydub import AudioSegment
from pydub.playback import play
import re
import time

def main():
    beats = []
    f = open("results/beats_211212_meta_re_lr0.1_feat_re.txt")
    for line in f:
        values = re.split(",|\n", line)
        beat = int(values[1])
        beats.append(beat)

    beep = AudioSegment.from_mp3("tick.mp3")
    beep = beep[len(beep) / 3: len(beep) / 3 * 2]

    for beat in beats:
        if beat > 0:
            play(beep)
        else: # beat is 0
            # wait for 330ms
            time.sleep(0.33)
if __name__ == '__main__':
    main()
