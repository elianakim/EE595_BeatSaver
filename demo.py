from pydub import AudioSegment
from pydub.playback import play
import re
import time
import os
import pandas as pd

def drop_the_beat(type, dynamic):
    if type == 0: # 2 beats
        if dynamic == 'p':
            print("   ##       ##    ")
            print("   # #      # #   ")
            print("  ## #     ## #   ")
            print("####     ####     ")
            print("###      ###      ")
        elif dynamic == 'f':
            print("         ###                 ###        ")
            print("         ###                 ###        ")
            print("         ####                ####       ")
            print("         ### ##              ### ##     ")
            print("         ###  ##             ###  ##    ")
            print("         ###   ##            ###   ##   ")
            print("         ###  ##             ###  ##    ")
            print("         ###   ##            ###   ##   ")
            print("   #######             #######          ")
            print(" ##########          ##########         ")
            print("  #########           #########         ")
            print("   ######              ######           ")
    elif type == 1: # 3 beats
        if dynamic == 'p':
            print("   ##       ##       ##    ")
            print("   # #      # #      # #   ")
            print("  ## #     ## #     ## #   ")
            print("####     ####     ####     ")
            print("###      ###      ###      ")
        elif dynamic == 'f':
            print("         ###                 ###                 ###        ")
            print("         ###                 ###                 ###        ")
            print("         ####                ####                ####       ")
            print("         ### ##              ### ##              ### ##     ")
            print("         ###  ##             ###  ##             ###  ##    ")
            print("         ###   ##            ###   ##            ###   ##   ")
            print("         ###  ##             ###  ##             ###  ##    ")
            print("         ###   ##            ###   ##            ###   ##   ")
            print("   #######             #######             #######          ")
            print(" ##########          ##########          ##########         ")
            print("  #########           #########           #########         ")
            print("   ######              ######              ######           ")
    elif type == 2: # 4 beats
        if dynamic == 'p':
            print("   ##       ##       ##       ##    ")
            print("   # #      # #      # #      # #   ")
            print("  ## #     ## #     ## #     ## #   ")
            print("####     ####     ####     ####     ")
            print("###      ###      ###      ###      ")
        elif dynamic == 'f':
            print("         ###                 ###                 ###                 ###        ")
            print("         ###                 ###                 ###                 ###        ")
            print("         ####                ####                ####                ####       ")
            print("         ### ##              ### ##              ### ##              ### ##     ")
            print("         ###  ##             ###  ##             ###  ##             ###  ##    ")
            print("         ###   ##            ###   ##            ###   ##            ###   ##   ")
            print("         ###  ##             ###  ##             ###  ##             ###  ##    ")
            print("         ###   ##            ###   ##            ###   ##            ###   ##   ")
            print("   #######             #######             #######             #######          ")
            print(" ##########          ##########          ##########          ##########         ")
            print("  #########           #########           #########           #########         ")
            print("   ######              ######              ######              ######           ")


def main():
    #f = open("results/beats_211212_meta_re_lr0.1_feat_re.txt")
    # demo 3
    f_dynamics = open("results/demo_beats_change_3_dynamic.csv")
    df = pd.read_csv(f_dynamics)
    beats = df['beats']
    dynamic = df['dynamic']
    f_types = open("results/demo_beats_type_3.csv")
    typedf = pd.read_csv(f_types)
    type = typedf['type']



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
                # print("%s" % type[i/2])
                print("##########")
                print("##########")
                print("##########")
                print("##########")
                print("##########")
                print("##########")
                time.sleep(0.33)
                os.system('clear')
            else:
                drop_the_beat(type[i//2], _dynamic)
                time.sleep(0.33)
                os.system('clear')
        else: # beat is 0
            # wait for 330ms
            time.sleep(0.33)
if __name__ == '__main__':
    main()
