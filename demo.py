import argparse
import time
import os
import sys
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


def main(args):
    if args.beat == 3:
        # demo 3
        f_dynamics = open("results/final/demo_beat_change_3_final.csv")
        f_types = open("results/final/demo_beat_type_3_final.csv")
    elif args.beat == 4:
        # demo 4
        f_dynamics = open("results/final/demo_beat_change_4_final.csv")
        f_types = open("results/final/demo_beat_type_4_final.csv")
    elif args.beat == 2:
        # demo 2
        f_dynamics = open("results/final/demo_beat_change_2_final.csv")
        f_types = open("results/final/demo_beat_type_2_final.csv")

    df = pd.read_csv(f_dynamics)
    beats = df['beats']
    dynamic = df['dynamic']
    typedf = pd.read_csv(f_types)
    type = typedf['type']



    '''
    for line in f:
        values = re.split(",|\n", line)
        beat = int(values[1])
        beats.append(beat)
    '''
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

def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--beat', type=int, default=2,
                        help='Time signature to play. [2, 3, 4].')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
