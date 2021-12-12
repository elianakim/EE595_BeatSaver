from pydub import AudioSegment
from pydub.playback import play
import re
import time
import os

def main():
    beats = []
    f = open("results/beats_211212_meta_re_lr0.1_feat_re.txt")
    for line in f:
        values = re.split(",|\n", line)
        beat = int(values[1])
        beats.append(beat)

    beep = AudioSegment.from_mp3("tick.mp3")
    beep = beep[0: len(beep) / 3.5]

    os.system('clear')

    for beat in beats:
        if beat > 0:
            print("##########")
            print("##########")
            print("##########")
            print("##########")
            print("##########")
            print("##########")
            time.sleep(0.33)
            os.system('clear')

        # if beat == 1: 
        #     print("----------")
        #     print("--#####---")
        #     print("----###---")
        #     print("----###---")
        #     print("----###---")
        #     print("--#######-")
        #     time.sleep(0.33)
        #     os.system('clear')
        #     # play(beep)
        # elif beat == 2:
        #     print("-########-")
        #     print("------###-")
        #     print("-########-")
        #     print("-###------")
        #     print("-###------")
        #     print("-########-")
        #     time.sleep(0.33)
        #     os.system('clear')
        # elif beat == 3:
        #     print("#########-")
        #     print("------###-")
        #     print("-########-")
        #     print("------###-")
        #     print("#########-")
        #     print("----------")
        #     time.sleep(0.33)
        #     os.system('clear')
        else: # beat is 0
            # wait for 330ms
            time.sleep(0.33)
if __name__ == '__main__':
    main()
