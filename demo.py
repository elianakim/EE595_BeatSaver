from pydub import AudioSegment
from pydub.playback import play
import re
import time
import os
import pandas as pd

def main():
    #f = open("results/beats_211212_meta_re_lr0.1_feat_re.txt")
    f = open("results/beats_211212_meta_re_lr0.1_feat_re_dynamic.csv")
    df = pd.read_csv(f)
    beats = df['beats']
    dynamic = df['dynamic']
    '''
    for line in f:
        values = re.split(",|\n", line)
        beat = int(values[1])
        beats.append(beat)
    '''
    beep = AudioSegment.from_mp3("tick.mp3")
    beep = beep[0: len(beep) / 3.5]

    os.system('clear')

    #for beat in beats:
    for i in range(len(df)):
        _beat = beats[i]
        _dynamic = dynamic[i]
        if _beat > 0:
            if i == 0:
                print("##########")
                print("##########")
                print("##########")
                print("##########")
                print("##########")
                print("##########")
                time.sleep(0.33)
                os.system('clear')
            elif _dynamic == 'f':
                print("         ###")
                print("         ###")
                print("         ####")
                print("         ### ##")
                print("         ###  ##")
                print("         ###   ##")
                print("         ###  ##")
                print("         ###   ##")
                print("   #######")
                print(" ##########")
                print("  #########")
                print("  ######")
                time.sleep(0.33)
                os.system('clear')
            elif _dynamic == 'p':
                print("   ##")
                print("   # #")
                print("  ## #")
                print("####")
                print("###")
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
