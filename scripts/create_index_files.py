import os
import argparse
import random
import itertools as it
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_audio', default='./data/audio',
                        help="root for extracted audio files")
    parser.add_argument('--root_frame', default='./data/frames',
                        help="root for extracted video frames")
    parser.add_argument('--path_output', default='./data',
                        help="path to output index files")
    args = parser.parse_args()


    def get_performers_split():

        performers = {'cello': {'train': [], 'val': [], 'test': []}, 'doublebass': {'train': [], 'val': [], 'test': []}, 'guitar': {'train': [], 'val': [], 'test': []}, 'sax': {'train': [], 'val': [], 'test': []} , 'violin': {'train': [], 'val': [], 'test': []}}

        performers['cello']['train'] = ['person_'+str(i) for i in range(1, 9)]
        performers['cello']['val'] = ['person_'+str(i) for i in range(9, 10)]
        performers['cello']['test'] = ['person_'+str(i) for i in range(10, 11)]

        for instr in ['doublebass', 'guitar', 'sax', 'violin']:
            performers[instr]['train'] = ['person_'+str(i) for i in range(1, 10)]
            performers[instr]['val'] = ['person_'+str(i) for i in range(10, 12)]
            performers[instr]['test'] = ['person_'+str(i) for i in range(12, 14)]

        return performers


    performers = get_performers_split()


    for instr in ['cello', 'doublebass', 'guitar', 'sax', 'violin']:
        audio_instr_path = os.path.join(args.root_audio, instr)
        pc_instr_path = os.path.join(args.root_frame, instr)
        audio_instr_fn = [filename for filename in os.listdir(audio_instr_path) if filename.endswith('.mp3')]
        random.shuffle(audio_instr_fn)
        split_instr_fn = {'train': [], 'val': [], 'test': []}
        split_instr_fn['train'], split_instr_fn['val'], split_instr_fn['test'] = np.split(audio_instr_fn, [int(len(audio_instr_fn)*0.75), int(len(audio_instr_fn)*0.9)])

        for set in ['train', 'val', 'test']:
            num_frames_performers = [len(os.listdir(os.path.join(pc_instr_path, performer))) for performer in performers[instr][set]]
            filename = '{}.csv'.format(os.path.join(args.path_output, set))

            with open(filename, 'a') as f:
                for audio, performer, num_frames in zip(split_instr_fn[set], it.cycle(performers[instr][set]), it.cycle(num_frames_performers)):
                    audio_path = os.path.join(audio_instr_path, audio)
                    performer_path = os.path.join(pc_instr_path, performer)
                    f.write(','.join([audio_path, performer_path, str(num_frames)]) + '\n')

    print('Done!')

