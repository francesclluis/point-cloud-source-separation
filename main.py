import os
import random
import time
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io.wavfile as wavfile
import imageio
import MinkowskiEngine as ME
from mir_eval.separation import bss_eval_sources
from arguments import ArgParser
from dataset import MUSICMixDataset
from models import ModelBuilder, activate
from utils import AverageMeter, \
    magnitude2heatmap,\
    istft_reconstruction, warpgrid, \
    save_points, makedirs, plot_loss_metrics


# Network wrapper, defines forward pass
class NetWrapper(torch.nn.Module):
    def __init__(self, nets, crit):
        super(NetWrapper, self).__init__()
        self.net_sound, self.net_vision, self.net_synthesizer = nets
        self.crit = crit

    def forward(self, batch_data, args):
        mag_mix = batch_data['mag_mix']
        mags = batch_data['mags']
        feats = batch_data['feats']
        coords = batch_data['coords']
        mag_mix = mag_mix + 1e-10

        N = args.num_mix
        FN = args.num_frames
        B = mag_mix.size(0)
        T = mag_mix.size(3)

        # 0.0 warp the spectrogram
        if args.log_freq:
            grid_warp = torch.from_numpy(
                warpgrid(B, 256, T, warp=True)).to(args.device)
            mag_mix = F.grid_sample(mag_mix, grid_warp)
            for n in range(N):
                mags[n] = F.grid_sample(mags[n], grid_warp)

        # 0.1 calculate loss weighting coefficient: magnitude of input mixture
        if args.weighted_loss:
            weight = torch.log1p(mag_mix)
            weight = torch.clamp(weight, 1e-3, 10)
        else:
            weight = torch.ones_like(mag_mix)

        # 0.2 ground truth masks are computed after warpping!
        gt_masks = [None for n in range(N)]
        for n in range(N):
            if args.binary_mask:
                if N > 2:
                    binary_masks = torch.ones_like(mags[n])
                    for m in range(N):
                       binary_masks *= (mags[n]>=mags[m])
                    gt_masks[n] = binary_masks
                else:
                    gt_masks[n] = (mags[n] > 0.5 * mag_mix).float()

            else:
                gt_masks[n] = mags[n] / mag_mix
                # clamp to avoid large numbers in ratio masks
                gt_masks[n].clamp_(0., 5.)

        # LOG magnitude
        log_mag_mix = torch.log(mag_mix).detach()

        # 1. forward net_sound
        feat_sound = self.net_sound(log_mag_mix)
        feat_sound = activate(feat_sound, args.sound_activation)

        # 2. forward net_vision
        feat_frames = [None for n in range(N)]
        for n in range(N):
            frames_features = []
            for f in range(FN):
                sin = ME.SparseTensor(feats[n][f], coords[n][f].int(), allow_duplicate_coords=True) #Create SparseTensor
                frames_features.append(self.net_vision.forward(sin))
            frames_features = torch.stack(frames_features)
            frames_features = frames_features.permute(1, 2, 0)
            if args.frame_pool == 'maxpool':
                feat_frame = F.adaptive_max_pool1d(frames_features, 1)
            elif args.frame_pool == 'avgpool':
                feat_frame = F.adaptive_avg_pool1d(frames_features, 1)
            feat_frames[n] = feat_frame.squeeze()
            feat_frames[n] = activate(feat_frames[n], args.vision_activation)

        # 3. sound synthesizer
        pred_masks = [None for n in range(N)]
        for n in range(N):
            pred_masks[n] = self.net_synthesizer(feat_frames[n], feat_sound)
            pred_masks[n] = activate(pred_masks[n], args.output_activation)

        # 4. loss
        err = self.crit(pred_masks, gt_masks, weight).reshape(1)

        return err, \
            {'pred_masks': pred_masks, 'gt_masks': gt_masks,
             'mag_mix': mag_mix, 'mags': mags, 'weight': weight}


# Calculate metrics
def calc_metrics(batch_data, outputs, args):
    # meters
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()

    # fetch data and predictions
    mag_mix = batch_data['mag_mix']
    phase_mix = batch_data['phase_mix']
    audios = batch_data['audios']
    pred_masks_ = outputs['pred_masks']

    # unwarp log scale
    N = args.num_mix
    B = mag_mix.size(0)
    pred_masks_linear = [None for n in range(N)]
    for n in range(N):
        if args.log_freq:
            grid_unwarp = torch.from_numpy(
                warpgrid(B, args.stft_frame//2+1, pred_masks_[0].size(3), warp=False)).to(args.device)
            pred_masks_linear[n] = F.grid_sample(pred_masks_[n], grid_unwarp)
        else:
            pred_masks_linear[n] = pred_masks_[n]

    # convert into numpy
    mag_mix = mag_mix.numpy()
    phase_mix = phase_mix.numpy()
    for n in range(N):
        pred_masks_linear[n] = pred_masks_linear[n].detach().cpu().numpy()

        # threshold if binary mask
        if args.binary_mask:
            pred_masks_linear[n] = (pred_masks_linear[n] > args.mask_thres).astype(np.float32)

    # loop over each sample
    for j in range(B):
        # save mixture
        mix_wav = istft_reconstruction(mag_mix[j, 0], phase_mix[j, 0], hop_length=args.stft_hop)

        # save each component
        preds_wav = [None for n in range(N)]
        for n in range(N):
            # Predicted audio recovery
            pred_mag = mag_mix[j, 0] * pred_masks_linear[n][j, 0]
            preds_wav[n] = istft_reconstruction(pred_mag, phase_mix[j, 0], hop_length=args.stft_hop)

        # separation performance computes
        L = preds_wav[0].shape[0]
        gts_wav = [None for n in range(N)]
        valid = True
        for n in range(N):
            gts_wav[n] = audios[n][j, 0:L].numpy()
            valid *= np.sum(np.abs(gts_wav[n])) > 1e-5
            valid *= np.sum(np.abs(preds_wav[n])) > 1e-5
        if valid:
            sdr, sir, sar, _ = bss_eval_sources(
                np.asarray(gts_wav),
                np.asarray(preds_wav),
                False)
            sdr_mix, _, _, _ = bss_eval_sources(
                np.asarray(gts_wav),
                np.asarray([mix_wav[0:L] for n in range(N)]),
                False)
            sdr_mix_meter.update(sdr_mix.mean())
            sdr_meter.update(sdr.mean())
            sir_meter.update(sir.mean())
            sar_meter.update(sar.mean())

    return [sdr_mix_meter.average(),
            sdr_meter.average(),
            sir_meter.average(),
            sar_meter.average()]


# Visualize predictions
def output_visuals(batch_data, outputs, args):

    mag_mix = batch_data['mag_mix']
    phase_mix = batch_data['phase_mix']
    features = batch_data['feats']
    coords = batch_data['coords']
    infos = batch_data['infos']

    pred_masks_ = outputs['pred_masks']
    gt_masks_ = outputs['gt_masks']
    mag_mix_ = outputs['mag_mix']
    weight_ = outputs['weight']

    # unwarp log scale
    N = args.num_mix
    FN = args.num_frames
    B = mag_mix.size(0)
    pred_masks_linear = [None for n in range(N)]
    gt_masks_linear = [None for n in range(N)]
    for n in range(N):
        if args.log_freq:
            grid_unwarp = torch.from_numpy(
                warpgrid(B, args.stft_frame//2+1, gt_masks_[0].size(3), warp=False)).to(args.device)
            pred_masks_linear[n] = F.grid_sample(pred_masks_[n], grid_unwarp)
            gt_masks_linear[n] = F.grid_sample(gt_masks_[n], grid_unwarp)
        else:
            pred_masks_linear[n] = pred_masks_[n]
            gt_masks_linear[n] = gt_masks_[n]

    # convert into numpy
    mag_mix = mag_mix.numpy()
    mag_mix_ = mag_mix_.detach().cpu().numpy()
    phase_mix = phase_mix.numpy()
    weight_ = weight_.detach().cpu().numpy()
    for n in range(N):
        pred_masks_[n] = pred_masks_[n].detach().cpu().numpy()
        pred_masks_linear[n] = pred_masks_linear[n].detach().cpu().numpy()
        gt_masks_[n] = gt_masks_[n].detach().cpu().numpy()
        gt_masks_linear[n] = gt_masks_linear[n].detach().cpu().numpy()

        # threshold if binary mask
        if args.binary_mask:
            pred_masks_[n] = (pred_masks_[n] > args.mask_thres).astype(np.float32)
            pred_masks_linear[n] = (pred_masks_linear[n] > args.mask_thres).astype(np.float32)

    # loop over each sample
    for j in range(B):

        # video names
        prefix = []
        for n in range(N):
            prefix.append('-'.join(infos[n][0][j].split('/')[-2:]).split('.')[0])
        prefix = '+'.join(prefix)
        makedirs(os.path.join(args.vis, prefix))

        # save mixture
        mix_wav = istft_reconstruction(mag_mix[j, 0], phase_mix[j, 0], hop_length=args.stft_hop)
        mix_amp = magnitude2heatmap(mag_mix_[j, 0])
        weight = magnitude2heatmap(weight_[j, 0], log=False, scale=100.)
        filename_mixwav = os.path.join(prefix, 'mix.wav')
        filename_mixmag = os.path.join(prefix, 'mix.jpg')
        filename_weight = os.path.join(prefix, 'weight.jpg')
        imageio.imwrite(os.path.join(args.vis, filename_mixmag), mix_amp[::-1, :, :])
        imageio.imwrite(os.path.join(args.vis, filename_weight), weight[::-1, :])
        wavfile.write(os.path.join(args.vis, filename_mixwav), args.audRate, mix_wav)

        # save each component
        preds_wav = [None for n in range(N)]
        for n in range(N):
            # GT and predicted audio recovery
            gt_mag = mag_mix[j, 0] * gt_masks_linear[n][j, 0]
            gt_wav = istft_reconstruction(gt_mag, phase_mix[j, 0], hop_length=args.stft_hop)
            pred_mag = mag_mix[j, 0] * pred_masks_linear[n][j, 0]
            preds_wav[n] = istft_reconstruction(pred_mag, phase_mix[j, 0], hop_length=args.stft_hop)

            # output masks
            filename_gtmask = os.path.join(prefix, 'gtmask{}.jpg'.format(n+1))
            filename_predmask = os.path.join(prefix, 'predmask{}.jpg'.format(n+1))
            gt_mask = (np.clip(gt_masks_[n][j, 0], 0, 1) * 255).astype(np.uint8)
            pred_mask = (np.clip(pred_masks_[n][j, 0], 0, 1) * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(args.vis, filename_gtmask), gt_mask[::-1, :])
            imageio.imwrite(os.path.join(args.vis, filename_predmask), pred_mask[::-1, :])

            # ouput spectrogram (log of magnitude, show colormap)
            filename_gtmag = os.path.join(prefix, 'gtamp{}.jpg'.format(n+1))
            filename_predmag = os.path.join(prefix, 'predamp{}.jpg'.format(n+1))
            gt_mag = magnitude2heatmap(gt_mag)
            pred_mag = magnitude2heatmap(pred_mag)
            imageio.imwrite(os.path.join(args.vis, filename_gtmag), gt_mag[::-1, :, :])
            imageio.imwrite(os.path.join(args.vis, filename_predmag), pred_mag[::-1, :, :])

            # output audio
            filename_gtwav = os.path.join(prefix, 'gt{}.wav'.format(n+1))
            filename_predwav = os.path.join(prefix, 'pred{}.wav'.format(n+1))
            wavfile.write(os.path.join(args.vis, filename_gtwav), args.audRate, gt_wav)
            wavfile.write(os.path.join(args.vis, filename_predwav), args.audRate, preds_wav[n])

            #output pointclouds
            for f in range(FN):
                idx = torch.where(coords[n][f][:, 0] == j)
                path_point = os.path.join(args.vis, prefix, 'point{}_frame{}.ply'.format(n+1,f+1))
                if args.rgbs_feature:
                    colors = np.asarray(features[n][f][idx])
                    xyz = np.asarray(coords[n][f][idx][:, 1:4])
                    xyz = xyz*args.voxel_size
                    save_points(path_point, xyz, colors)
                else:
                    xyz = np.asarray(features[n][f][idx])
                    save_points(path_point, xyz)


def evaluate(netWrapper, loader, history, epoch, args):
    print('Evaluating at {} epochs...'.format(epoch))
    torch.set_grad_enabled(False)

    # remove previous viz results
    makedirs(args.vis, remove=True)

    # switch to eval mode
    netWrapper.eval()

    # initialize meters
    loss_meter = AverageMeter()
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()

    for i, batch_data in enumerate(loader):
        # forward pass
        err, outputs = netWrapper.forward(batch_data, args)
        err = err.mean()

        loss_meter.update(err.item())
        print('[Eval] iter {}, loss: {:.4f}'.format(i, err.item()))

        # calculate metrics
        sdr_mix, sdr, sir, sar = calc_metrics(batch_data, outputs, args)
        sdr_mix_meter.update(sdr_mix)
        sdr_meter.update(sdr)
        sir_meter.update(sir)
        sar_meter.update(sar)

        # output visualization
        output_visuals(batch_data, outputs, args)

    print('[Eval Summary] Epoch: {}, Loss: {:.4f}, '
          'SDR_mixture: {:.4f}, SDR: {:.4f}, SIR: {:.4f}, SAR: {:.4f}'
          .format(epoch, loss_meter.average(),
                  sdr_mix_meter.average(),
                  sdr_meter.average(),
                  sir_meter.average(),
                  sar_meter.average()))
    history['val']['epoch'].append(epoch)
    history['val']['err'].append(loss_meter.average())
    history['val']['sdr'].append(sdr_meter.average())
    history['val']['sir'].append(sir_meter.average())
    history['val']['sar'].append(sar_meter.average())

    # Plot figure
    if epoch > 0:
        print('Plotting figures...')
        plot_loss_metrics(args.ckpt, history)


# train one epoch
def train(netWrapper, loader, optimizer, history, epoch, args):
    torch.set_grad_enabled(True)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # switch to train mode
    netWrapper.train()

    # main loop
    torch.cuda.synchronize()
    tic = time.perf_counter()
    for i, batch_data in enumerate(loader):
        torch.cuda.synchronize()
        data_time.update(time.perf_counter() - tic)

        # forward pass
        netWrapper.zero_grad()
        err, _ = netWrapper.forward(batch_data, args)
        err = err.mean()

        # backward
        err.backward()
        optimizer.step()

        # measure total time
        torch.cuda.synchronize()
        batch_time.update(time.perf_counter() - tic)
        tic = time.perf_counter()

        # display
        if i % args.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_sound: {}, lr_vision: {}, lr_synthesizer: {}, '
                  'loss: {:.4f}'
                  .format(epoch, i, args.epoch_iters,
                          batch_time.average(), data_time.average(),
                          args.lr_sound, args.lr_vision, args.lr_synthesizer,
                          err.item()))
            fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['err'].append(err.item())


def checkpoint(nets, history, epoch, args):
    print('Saving checkpoints at {} epochs.'.format(epoch))
    (net_sound, net_vision, net_synthesizer) = nets
    suffix_latest = 'latest.pth'
    suffix_best = 'best.pth'

    torch.save(history,
               '{}/history_{}'.format(args.ckpt, suffix_latest))
    torch.save(net_sound.state_dict(),
               '{}/sound_{}'.format(args.ckpt, suffix_latest))
    torch.save(net_vision.state_dict(),
               '{}/vision_{}'.format(args.ckpt, suffix_latest))
    torch.save(net_synthesizer.state_dict(),
               '{}/synthesizer_{}'.format(args.ckpt, suffix_latest))

    cur_err = history['val']['err'][-1]
    if cur_err < args.best_err:
        args.best_err = cur_err
        torch.save(net_sound.state_dict(),
                   '{}/sound_{}'.format(args.ckpt, suffix_best))
        torch.save(net_vision.state_dict(),
                   '{}/vision_{}'.format(args.ckpt, suffix_best))
        torch.save(net_synthesizer.state_dict(),
                   '{}/synthesizer_{}'.format(args.ckpt, suffix_best))


def create_optimizer(nets, args):
    (net_sound, net_vision, net_synthesizer) = nets
    param_groups = [{'params': net_sound.parameters(), 'lr': args.lr_sound},
                    {'params': net_synthesizer.parameters(), 'lr': args.lr_synthesizer},
                    {'params': net_vision.features.parameters(), 'lr': args.lr_vision},
                    {'params': net_vision.fc.parameters(), 'lr': args.lr_sound}]
    return torch.optim.SGD(param_groups, momentum=args.beta1, weight_decay=args.weight_decay)


def collate_all(list_data):

    # Process samples to form a batch. Point Cloud information is processed to be compatible with Minkowski Engine (ME).
    # Note that ME creates a batch of point clouds coordinates putting the batch indice in the first column.
    # Samples come from dataset/music.py

    miscellaneous_data = []

    N = len(list_data[0][1][0])
    FN = len(list_data[0][1][0][0])

    #Create dicts to gather point cloud data
    pcloud_data = {'coords': {}, 'feats': {}}
    for i in range(N):
        pcloud_data['coords'][i] = {}
        pcloud_data['feats'][i] = {}
        for f in range(FN):
            pcloud_data['coords'][i][f] = []
            pcloud_data['feats'][i][f] = []

    for sample in list_data:
        miscellaneous_data.append(sample[0])
        coords, points, rgbs, rgbs_feature = sample[-1]
        for i in range(N):
            for f in range(FN):
                pcloud_data['coords'][i][f].append(coords[i][f])
                if rgbs_feature:
                    pcloud_data['feats'][i][f].append(rgbs[i][f])
                else:
                    pcloud_data['feats'][i][f].append(points[i][f])

    batched_coords = []
    batched_feats = []
    for i in range(N):
        f_coords = []
        f_feats = []
        for f in range(FN):
            f_coords.append(ME.utils.batched_coordinates(pcloud_data['coords'][i][f]))
            f_feats.append(torch.from_numpy(np.vstack(pcloud_data['feats'][i][f])).float())
        batched_coords.append(tuple(f_coords))
        batched_feats.append(tuple(f_feats))

    coords_batch = tuple(batched_coords)
    feats_batch = tuple(batched_feats)

    batched_dict = torch.utils.data.dataloader.default_collate(miscellaneous_data)

    batched_dict['coords'] = coords_batch
    batched_dict['feats'] = feats_batch

    return batched_dict


def make_data_loader(dset, args):

    args = {
        'batch_size': args.batch_size,
        'num_workers': int(args.workers),
        'collate_fn': collate_all,
        'pin_memory': False,
        'drop_last': False,
        'shuffle': True
    }

    loader = torch.utils.data.DataLoader(dset, **args)

    return loader

def main(args):
    # Network Builders
    builder = ModelBuilder()
    net_sound = builder.build_sound(
        arch=args.arch_sound,
        fc_dim=args.num_channels,
        weights=args.weights_sound)
    net_vision = builder.build_vision(
        arch=args.arch_vision,
        fc_dim=args.num_channels,
        weights=args.weights_vision)
    net_synthesizer = builder.build_synthesizer(
        arch=args.arch_synthesizer,
        fc_dim=args.num_channels,
        weights=args.weights_synthesizer)
    nets = (net_sound, net_vision, net_synthesizer)
    crit = builder.build_criterion(arch=args.loss)


    # Dataset and Loader
    dataset_train = MUSICMixDataset(
        args.list_train, args, split='train')
    dataset_val = MUSICMixDataset(
        args.list_val, args, max_sample=args.num_val, split='val')

    loader_train = make_data_loader(dataset_train, args)
    loader_val = make_data_loader(dataset_val, args)

    args.epoch_iters = len(dataset_train) // args.batch_size
    if args.mode == 'train':
        print('1 Epoch = {} iters'.format(args.epoch_iters))

    # Wrap networks
    netWrapper = NetWrapper(nets, crit)
    netWrapper = torch.nn.DataParallel(netWrapper, device_ids=range(args.num_gpus))
    netWrapper.to(args.device)

    # Set up optimizer
    optimizer = create_optimizer(nets, args)

    # History of performance
    history = {
        'train': {'epoch': [], 'err': []},
        'val': {'epoch': [], 'err': [], 'sdr': [], 'sir': [], 'sar': []}}

    # Eval mode
    evaluate(netWrapper, loader_val, history, 0, args)
    if args.mode == 'eval':
        print('Evaluation Done!')
        return

    # Training loop
    for epoch in range(1, args.num_epoch + 1):
        train(netWrapper, loader_train, optimizer, history, epoch, args)

        # Evaluation and visualization
        if epoch % args.eval_epoch == 0:
            evaluate(netWrapper, loader_val, history, epoch, args)

            # checkpointing
            checkpoint(nets, history, epoch, args)

    print('Training Done!')


if __name__ == '__main__':
    # arguments
    parser = ArgParser()
    args = parser.parse_train_arguments()
    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.device = torch.device("cuda")

    # experiment name
    if args.mode == 'train':
        args.id += '-{}mix'.format(args.num_mix)
        if args.rgbs_feature:
            args.id += '-rgb_depth'
        else:
            args.id += '-only_depth'
        if args.log_freq:
            args.id += '-LogFreq'
        args.id += '-{}-{}-{}'.format(
            args.arch_vision, args.arch_sound, args.arch_synthesizer)
        args.id += '-frames{}stride{}'.format(args.num_frames, args.stride_frames)
        args.id += '-{}'.format(args.frame_pool)
        if args.binary_mask:
            assert args.loss == 'bce', 'Binary Mask should go with BCE loss'
            args.id += '-binary'
        else:
            args.id += '-ratio'
        if args.weighted_loss:
            args.id += '-weightedLoss'
        args.id += '-channels{}'.format(args.num_channels)
        args.id += '-epoch{}'.format(args.num_epoch)

    print('Model ID: {}'.format(args.id))

    # paths to save/load output
    args.ckpt = os.path.join(args.ckpt, args.id)
    args.vis = os.path.join(args.ckpt, 'visualization/')
    if args.mode == 'train':
        makedirs(args.ckpt, remove=True)
    elif args.mode == 'eval':
        args.weights_sound = os.path.join(args.ckpt, 'sound_best.pth')
        args.weights_vision = os.path.join(args.ckpt, 'vision_best.pth')
        args.weights_synthesizer = os.path.join(args.ckpt, 'synthesizer_best.pth')

    # initialize best error with a big number
    args.best_err = float("inf")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
