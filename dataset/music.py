import os
import random
from .base import BaseDataset


class MUSICMixDataset(BaseDataset):
    def __init__(self, list_sample, opt, **kwargs):
        super(MUSICMixDataset, self).__init__(
            list_sample, opt, **kwargs)
        self.fps = opt.frameRate
        self.num_mix = opt.num_mix
        self.rgbs_feature = opt.rgbs_feature

    def __getitem__(self, index):
        N = self.num_mix
        points = [None for n in range(N)]
        coords = [None for n in range(N)]
        rgbs = [None for n in range(N)]
        audios = [None for n in range(N)]
        infos = [[] for n in range(N)]
        path_frames = [[] for n in range(N)]
        path_audios = ['' for n in range(N)]

        # the first point cloud video
        instruments = []
        infos[0] = self.list_sample[index]
        path_instr = self.list_sample[index][0]
        instr = os.path.basename(os.path.dirname(path_instr))
        instruments.append(instr)

        # sample other point cloud videos
        if not self.split == 'train':
            random.seed(index)

        while len(instruments) != N:
            indexN = random.randint(0, len(self.list_sample)-1)
            path_instr = self.list_sample[indexN][0]
            instr = os.path.basename(os.path.dirname(path_instr))
            if instr not in instruments:
                infos[len(instruments)] = self.list_sample[indexN]
                instruments.append(instr)

        # select point cloud frames
        idx_margin = (self.num_frames // 2) * self.stride_frames
        for n, infoN in enumerate(infos):
            path_audioN, path_frameN, count_framesN = infoN
            center_frameN = random.randint(
                idx_margin+1, int(count_framesN)-idx_margin-1)

            # absolute frame/audio paths
            for i in range(self.num_frames):
                idx_offset = (i - self.num_frames // 2) * self.stride_frames
                path_frames[n].append(
                    os.path.join(
                        path_frameN,
                        '{:05d}.ply'.format(center_frameN + idx_offset)))
            path_audios[n] = path_audioN

        for n, infoN in enumerate(infos):
            points[n], coords[n], rgbs[n] = self._load_frames(path_frames[n])
            audios[n] = self._load_audio(path_audios[n])
        mag_mix, mags, phase_mix = self._mix_n_and_stft(audios)

        ret_dict = {'mag_mix': mag_mix, 'mags': mags}
        if self.split != 'train':
            ret_dict['audios'] = audios
            ret_dict['phase_mix'] = phase_mix
            ret_dict['infos'] = infos

        return ret_dict, (coords, points, rgbs, self.rgbs_feature)
