import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch
import logging
import math
import importlib
import datetime
import random
import os
import sys
import argparse

import time
import glob
import h5py
import numpy as np
import torch.utils.data as data
import csv
from collections import defaultdict, OrderedDict
from tensorboardX import SummaryWriter
from pathlib import Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR_PATH = Path(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR_PATH, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import normal_estimation_utils
import torch.nn.functional as F

from dataset import PointcloudPatchDataset, RandomPointcloudPatchSampler, SequentialShapeRandomPointcloudPatchSampler, SequentialPointcloudPatchSampler


DEVICE = torch.device('cuda', 0)



def compute_loss_trans(pred, target, outputs, output_pred_ind, output_target_ind, output_loss_weight, normal_loss_type,
                 patch_rot=None, phase='train',
                 use_consistency=False, point_weights=None, neighbor_normals=None, opt=None, trans=None, trans2=None, z_trans_weight=0):

    loss = torch.zeros(1, device=trans.device, dtype=trans.dtype)
    n_loss = torch.zeros(1, device=trans.device, dtype=trans.dtype)
    consistency_loss = torch.zeros(1, device=trans.device, dtype=trans.dtype)
    z_trans_loss = torch.zeros(1, device=trans.device, dtype=trans.dtype)
    # # generate a inv normal distribution for weight kl div
    # add pointnet transformation regularization
    regularizer_trans = 0
    if trans is not None:
        regularizer_trans += 0.1 * torch.nn.MSELoss()(trans * trans.permute(0, 2, 1),
                                                torch.eye(3, device=trans.device).unsqueeze(0).repeat(
                                                    trans.size(0), 1, 1))
    if trans2 is not None:
        regularizer_trans += 0.01 * torch.nn.MSELoss()(trans2 * trans2.permute(0, 2, 1),
                                                 torch.eye(64, device=trans2.device).unsqueeze(0).repeat(
                                                     trans2.size(0), 1, 1))

    for oi, o in enumerate(outputs):
        if o == 'unoriented_normals' or o == 'oriented_normals':
            # o_pred = pred[:, output_pred_ind[oi]:output_pred_ind[oi]+3]
            c_pred = pred[0][:, output_pred_ind[oi]:output_pred_ind[oi]+3]
            r_pred = pred[1][:, output_pred_ind[oi]:output_pred_ind[oi]+3]
            o_target = target[output_target_ind[oi]]
            if patch_rot is not None:
                # transform predictions with inverse transform
                # since we know the transform to be a rotation (QSTN), the transpose is the inverse
                # o_pred = torch.bmm(o_pred.unsqueeze(1), patch_rot.transpose(2, 1)).squeeze(1)
                c_pred = torch.bmm(c_pred.unsqueeze(1), patch_rot.transpose(2, 1)).squeeze(1)
                r_pred = torch.bmm(r_pred.unsqueeze(1), patch_rot.transpose(2, 1)).squeeze(1)

            ## add z-direction transformation loss
            if z_trans_weight !=0 and trans is not None:
                batch_size = trans.shape[0]
                z_vector = torch.from_numpy(np.array([0, 0, 1]).astype(np.float32)).squeeze().repeat(batch_size, 1).to(trans.device)
                z_vector_rot = torch.bmm(z_vector.unsqueeze(1), trans.transpose(2, 1)).squeeze(1)
                z_vector_rot = F.normalize(z_vector_rot, dim=1)
                z_trans_loss_o = torch.norm(torch.cross(z_vector_rot, o_target, dim=-1), p=2, dim=1).mean()
                z_trans_loss = z_trans_weight * z_trans_loss_o
                # print(z_trans_loss)
            if o == 'unoriented_normals':
                if normal_loss_type == 'ms_euclidean':
                    normal_loss = torch.min((o_pred-o_target).pow(2).sum(1), (o_pred+o_target).pow(2).sum(1)).mean() * output_loss_weight[oi]
                elif normal_loss_type == 'ms_oneminuscos':
                    cos_ang = normal_estimation_utils.cos_angle(o_pred, o_target)
                    normal_loss = (1-torch.abs(cos_ang)).pow(2).mean() * output_loss_weight[oi]
                elif normal_loss_type == 'sin':
                    # normal_loss = 0.5 * torch.norm(torch.cross(o_pred, o_target, dim=-1), p=2, dim=1).mean()
                    normal_loss = 0.5 * torch.norm(torch.cross(c_pred, o_target, dim=-1), p=2, dim=1).mean() + \
                            0.5 * torch.norm(torch.cross(r_pred, o_target, dim=-1), p=2, dim=1).mean()
                else:
                    raise ValueError('Unsupported loss type: %s' % (normal_loss_type))
                loss = normal_loss
                # get angle value at test time (not in training to save runtime)
                if phase == 'test':
                    if not normal_loss_type == 'ms_oneminuscos':
                        cos_ang = torch.abs(normal_estimation_utils.cos_angle(r_pred, o_target))
                        cos_ang[cos_ang>1] = 1
                    angle = torch.acos(cos_ang)
                    err_angle = torch.mean(angle)
                else:
                    err_angle = None
            else:
                raise ValueError('Unsupported output type: %s' % (o))

        elif o == 'max_curvature' or o == 'min_curvature':
            o_pred = pred[:, output_pred_ind[oi]:output_pred_ind[oi]+1]
            o_target = target[output_target_ind[oi]]

            # Rectified mse loss: mean square of (pred - gt) / max(1, |gt|)
            normalized_diff = (o_pred - o_target) / torch.clamp(torch.abs(o_target), min=1)
            loss += normalized_diff.pow(2).mean() * output_loss_weight[oi]

        elif o == 'neighbor_normals':
            if use_consistency:
                c_pred = neighbor_normals[0]
                r_pred = neighbor_normals[1]

                o_target = target[output_target_ind[oi]] # 128,128,3
                
                batch_size, n_points, _ = c_pred.shape
                if patch_rot is not None:
                    # transform predictions with inverse transform
                    c_pred = torch.bmm(c_pred.view(-1, 1, 3),
                                       patch_rot.transpose(2, 1).repeat(1, n_points, 1, 1).view(-1, 3, 3)).view(batch_size, n_points, 3)
                    r_pred = torch.bmm(r_pred.view(-1, 1, 3),
                                       patch_rot.transpose(2, 1).repeat(1, n_points, 1, 1).view(-1, 3, 3)).view(batch_size, n_points, 3)


                # if opt.jet_order < 2: # when the jet has order higher than 2 the normal vector orientation matters.
                if normal_loss_type == 'ms_euclidean':
                    consistency_loss = torch.mean(point_weights * torch.min((c_pred - o_target).pow(2).sum(2),
                                                                            (c_pred + o_target).pow(2).sum(2)))
                    consistency_loss += torch.mean(point_weights * torch.min((r_pred - o_target).pow(2).sum(2),
                                                                            (r_pred + o_target).pow(2).sum(2)))                                 
                elif normal_loss_type == 'ms_oneminuscos':
                    cos_ang_c = normal_estimation_utils.cos_angle(c_pred.view(-1, 3),
                                                                o_target.view(-1, 3)).view(batch_size, n_points)
                    cos_ang_r = normal_estimation_utils.cos_angle(r_pred.view(-1, 3),
                                                                o_target.view(-1, 3)).view(batch_size, n_points)

                    consistency_loss = torch.mean(point_weights * (1 - torch.abs(cos_ang_c)).pow(2))
                    consistency_loss += torch.mean(point_weights * (1 - torch.abs(cos_ang_r)).pow(2))

                elif normal_loss_type == 'sin':
                    consistency_loss = 0.125 * torch.mean(point_weights * torch.norm(torch.cross(c_pred.view(-1, 3),
                                                    o_target.view(-1, 3), dim=-1).view(batch_size, -1, 3), p=2, dim=2)) 
                    consistency_loss += 0.125 * torch.mean(point_weights * torch.norm(torch.cross(r_pred.view(-1, 3),
                                                    o_target.view(-1, 3), dim=-1).view(batch_size, -1, 3), p=2, dim=2)) 

                else:
                    raise ValueError('Unsupported loss type: %s' % (normal_loss_type))

            if opt.con_reg == 'mean':
                regularizer = - 0.01 * torch.mean(point_weights)
            elif opt.con_reg == "log":
                regularizer = - 0.05 * torch.mean(point_weights.log())
            elif opt.con_reg == 'norm':
                regularizer = torch.mean((1/n_points)*torch.norm(point_weights-1, dim=1))
            else:
                raise ValueError("Unkonwn consistency regularizer")
            regularizer1 = regularizer_trans + regularizer
            # consistency_loss1 = consistency_loss + regularizer1

            loss = consistency_loss + normal_loss + z_trans_loss + regularizer1
        else:
            raise ValueError('Unsupported output type: %s' % (o))
    if phase =='train':
        return loss, regularizer_trans, regularizer, err_angle, consistency_loss, normal_loss, z_trans_loss_o
    else:
        return loss, regularizer_trans, regularizer, err_angle, consistency_loss, normal_loss



def get_target_features(opt):
    # get indices in targets and predictions corresponding to each output
    target_features = []
    output_target_ind = []
    output_pred_ind = []
    output_loss_weight = []
    pred_dim = 0
    for o in opt.outputs:
        if o == 'unoriented_normals' or o == 'oriented_normals':
            if 'normal' not in target_features:
                target_features.append('normal')

            output_target_ind.append(target_features.index('normal'))
            output_pred_ind.append(pred_dim)
            output_loss_weight.append(1.0)
            pred_dim += 3
        elif o == 'max_curvature' or o == 'min_curvature':
            if o not in target_features:
                target_features.append(o)

            output_target_ind.append(target_features.index(o))
            output_pred_ind.append(pred_dim)
            if o == 'max_curvature':
                output_loss_weight.append(0.7)
            else:
                output_loss_weight.append(0.3)
            pred_dim += 1
        elif o == 'neighbor_normals':
            target_features.append(o)
            output_target_ind.append(target_features.index(o))
            output_pred_ind.append(pred_dim)
        else:
            raise ValueError('Unknown output: %s' % (o))

    if pred_dim <= 0:
        raise ValueError('Prediction is empty for the given outputs.')

    return target_features, output_target_ind, output_pred_ind, output_loss_weight

def get_data_loaders(opt, target_features):
    # create train and test dataset loaders

    train_dataset = PointcloudPatchDataset(
        root=opt.indir,
        shape_list_filename=opt.trainset,
        patch_radius=opt.patch_radius,
        points_per_patch=opt.points_per_patch,
        patch_features=target_features,
        point_count_std=opt.patch_point_count_std,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs,
        use_pca=opt.use_pca,
        center=opt.patch_center,
        point_tuple=opt.point_tuple,
        cache_capacity=opt.cache_capacity,
        neighbor_search_method=opt.neighbor_search)
    if opt.training_order == 'random':
        train_datasampler = RandomPointcloudPatchSampler(
            train_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    elif opt.training_order == 'random_shape_consecutive':
        train_datasampler = SequentialShapeRandomPointcloudPatchSampler(
            train_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    else:
        raise ValueError('Unknown training order: %s' % (opt.training_order))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_datasampler,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers)
        )

    test_dataset = PointcloudPatchDataset(
        root=opt.indir,
        shape_list_filename=opt.testset,
        patch_radius=opt.patch_radius,
        points_per_patch=opt.points_per_patch,
        patch_features=target_features,
        point_count_std=opt.patch_point_count_std,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs,
        use_pca=opt.use_pca,
        center=opt.patch_center,
        point_tuple=opt.point_tuple,
        cache_capacity=opt.cache_capacity,
        neighbor_search_method=opt.neighbor_search)

    if opt.training_order == 'random':
        test_datasampler = RandomPointcloudPatchSampler(
            test_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    elif opt.training_order == 'random_shape_consecutive':
        test_datasampler = SequentialShapeRandomPointcloudPatchSampler(
            test_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    else:
        raise ValueError('Unknown training order: %s' % (opt.training_order))

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        sampler=test_datasampler,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers))

    return train_dataloader, train_dataset, train_datasampler, test_dataloader, test_dataset, test_datasampler

def train_pcpnet(opt):
    logging.info(str(opt))
    log_dirname = log_dir
    out_dir = os.path.join(log_dirname, 'trained_models')
    try:
        os.makedirs(out_dir)
    except OSError:
        pass
    params_filename = os.path.join(out_dir, 'params.pth')
    model_filename = os.path.join(out_dir, 'model.pth')
    desc_filename = os.path.join(out_dir, 'description.txt')
    log_filename = os.path.join(log_dirname, 'out.log')

    train_writer = SummaryWriter(os.path.join(log_dirname, 'train'))
    test_writer = SummaryWriter(os.path.join(log_dirname, 'test'))
    log_file = open(log_filename, 'w')


    if not opt.seed:
        seed = random.randint(1, 10000)
    else:
        seed = int(opt.seed)
    logging.info('Random Seed: %d' % seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model_module = importlib.import_module('.%s' % opt.model_name, 'models')
    net = torch.nn.DataParallel(model_module.Net(opt))

    target_features, output_target_ind, output_pred_ind, output_loss_weight = get_target_features((opt))
    train_dataloader, train_dataset, train_datasampler, test_dataloader, test_dataset, \
    test_datasampler = get_data_loaders(opt, target_features)

    # keep the exact training shape names for later reference
    opt.train_shapes = train_dataset.shape_names
    opt.test_shapes = test_dataset.shape_names

    logging.info('training set: %d patches (in %d batches) - test set: %d patches (in %d batches)' %
          (len(train_datasampler), len(train_dataloader), len(test_datasampler), len(test_dataloader)))

    total_trainable_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logging.info("===number of trainable parameters in upsampler: {:.4f} K === ".format(float(total_trainable_parameters / 1e3)))
    net.cuda()

    if hasattr(model_module, 'weights_init'):
        net.module.apply(model_module.weights_init)

    if opt.load_model:
        ckpt = torch.load(opt.load_model)
        net.module.load_state_dict(ckpt['net_state_dict'])
        logging.info("%s's previous weights loaded." % opt.model_name)

    lr = opt.lr
    optimizer = getattr(optim, opt.optimizer)

    if opt.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=opt.momentum)
    elif opt.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=opt.lr, weight_decay=0.0000001, eps=opt.opt_eps)
    elif opt.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), lr=opt.lr, weight_decay=0.0000001, eps=opt.opt_eps)
    else:
        raise ValueError("Unsupported optimizer")

    decay_epoch_list = [int(ep.strip()) for ep in opt.lr_step_decay_epochs.split(',')]

    if opt.scheduler_type == 'step':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=decay_epoch_list, gamma=opt.lr_decay_rate) # milestones in number of optimizer iterations
    else:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                                               verbose=False, threshold=0.0001, threshold_mode='rel',
                                               cooldown=5, min_lr=1e-012, eps=1e-08)


    train_num_batch = len(train_dataloader)
    test_num_batch = len(test_dataloader)

    # save parameters
    torch.save(opt, params_filename)


    # save description
    with open(desc_filename, 'w+') as text_file:
        print(opt.desc, file=text_file)
    

    for epoch in range(opt.start_epoch, opt.nepoch+1):
        # val(net, epoch, val_loss_meters, dataloader_test, best_epoch_losses)
        net.module.train()

        if opt.lr_decay:
            if opt.lr_decay_interval:
                # if epoch > 0 and epoch % args.lr_decay_interval == 0:
                #     lr = lr * args.lr_decay_rate
                lr = opt.lr * pow(opt.lr_decay_rate, epoch//opt.lr_decay_interval)
            elif opt.lr_step_decay_epochs:
                # if epoch in decay_epoch_list:
                ls_cnt = 0
                for inval in decay_epoch_list:
                    if epoch < inval:
                        break
                    else:
                        ls_cnt +=1
                    lr = opt.lr * pow(opt.lr_decay_rate, ls_cnt)
            if opt.lr_clip:
                lr = max(lr, opt.lr_clip)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        #torch.autograd.set_detect_anomaly(True)
        time0 = time.time()

        for i, data in enumerate(train_dataloader, 0):
            time1 = time.time()
#            print(time1-time0)

            optimizer.zero_grad()

            # get trainingset batch and upload to GPU
            points = data[0]
            target = data[1:-2]
            # n_effective_points = data[-1].squeeze()

            points = points.transpose(2, 1).cuda()
            target = tuple(t.cuda() for t in target)

            # forward pass
            pred, beta_pred, weights, trans, trans2, neighbor_normals, bias = net(points)
            loss, reg_trans, reg_weights, _, consistency_loss, normal_loss, z_trans_loss = compute_loss_trans(
                pred=pred, target=target,
                outputs=opt.outputs,
                output_pred_ind=output_pred_ind,
                output_target_ind=output_target_ind,
                output_loss_weight=output_loss_weight,
                normal_loss_type=opt.normal_loss,
                patch_rot=trans if opt.use_point_stn else None,
                use_consistency=opt.use_consistency, point_weights=weights, neighbor_normals=neighbor_normals,
                opt=opt, trans=trans, trans2=trans2, z_trans_weight=opt.z_trans_weight)

            bias_loss = 0.1 * torch.nn.MSELoss()(bias,torch.zeros_like(bias,device=trans.device, dtype=trans.dtype))
            loss += bias_loss


            loss.backward()
            optimizer.step()

            # print info and update log file
            if i % args.step_interval_to_print == 0:
                train_fraction_done = (i+1) / train_num_batch
                logging.info('%s [epoch %d: %d/%d] lr: %f  %s loss: %f, z_trans_loss: %f' % (opt.model_name, epoch, i, train_num_batch-1, lr, 'train', loss.mean().item(), z_trans_loss.mean().item()))

                train_writer.add_scalar('total_loss', loss.item(),
                                        (epoch + train_fraction_done) * train_num_batch * opt.batchSize)
                train_writer.add_scalar('reg_trans', reg_trans.item(),
                                    (epoch + train_fraction_done) * train_num_batch * opt.batchSize)
                train_writer.add_scalar('reg_weights', reg_weights.item(),
                                        (epoch + train_fraction_done) * train_num_batch * opt.batchSize)
                train_writer.add_scalar('consistency_loss', consistency_loss.item(),
                                        (epoch + train_fraction_done) * train_num_batch * opt.batchSize)
                train_writer.add_scalar('normal_loss', normal_loss.item(),
                                        (epoch + train_fraction_done) * train_num_batch * opt.batchSize)
                train_writer.add_histogram('weights', weights.detach().cpu().numpy(),
                                        (epoch + train_fraction_done) * train_num_batch * opt.batchSize)


        if epoch % opt.save_step == 0  or epoch == opt.nepoch-1:
            path = os.path.join(log_dir,'{:d}_network.pth'.format(epoch))
            torch.save({'net_state_dict': net.module.state_dict()}, path)
            net.module.eval()
            logging.info('Testing...')
            avg_test_loss = 0.0
            torch.cuda.empty_cache()

            for i, test_data in enumerate(test_dataloader, 0):
                # get testset batch and upload to GPU
                points = test_data[0]
                target = test_data[1:-2]
                data_trans = test_data[-2]
    
                points = points.transpose(2, 1)
                points = points.cuda()
                data_trans = data_trans.cuda()
       
                target = tuple(t.cuda() for t in target)

                # forward pass
                with torch.no_grad():
                    pred, beta_pred, weights, trans, trans2, neighbor_normals, bias = net(points)
        
                loss, reg_trans, reg_weights, err_angle, consistency_loss, normal_loss = compute_loss_trans(
                    pred=pred, target=target,
                    outputs=opt.outputs,
                    output_pred_ind=output_pred_ind,
                    output_target_ind=output_target_ind,
                    output_loss_weight=output_loss_weight,
                    normal_loss_type=opt.normal_loss,
                    patch_rot=trans if opt.use_point_stn else None, phase='test',
                    use_consistency=opt.use_consistency, point_weights=weights, neighbor_normals=neighbor_normals,
                    opt=opt, trans=trans, trans2=trans2)

                # avg_test_loss = avg_test_loss + loss.item()
                test_fraction_done = (i+1) / test_num_batch
                avg_test_loss = avg_test_loss + loss.item()
                # avg_test_loss = avg_test_loss + err_angle

            avg_test_loss = avg_test_loss / test_num_batch
            logging.info("Current epoch: {:d}, loss: {:.3f}".format(epoch, avg_test_loss))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train config file')

    # naming / file handling
    parser.add_argument('--name', type=str, default='our training', help='training run name')
    parser.add_argument('--model_name', type=str, default='GraphFit_ms_HG', help='training run name')
    parser.add_argument('--desc', type=str, default='My training run for multi-scale normal estimation.', help='description')
    parser.add_argument('--indir', type=str, default='/data1/PU_project/GraphFit-master/data/pcpnet', help='input folder (point clouds)')
    parser.add_argument('--logdir', type=str, default='./log/', help='training log folder')
    parser.add_argument('--trainset', type=str, default='trainingset_whitenoise.txt', help='training set file name')
    parser.add_argument('--testset', type=str, default='validationset_no_noise.txt', help='test set file name')
    parser.add_argument('--saveinterval', type=int, default='50', help='save model each n epochs')
    parser.add_argument('--resume', action="store_true", help='flag to resume the model, path determined by outri and model name')
    parser.add_argument('--start_epoch', type=int, default=0, help='resume model from this epoch')
    parser.add_argument('--load_model', type=str, default='', help='path of model')
    parser.add_argument('--overwrite', action="store_true", help='to overwrite existing log directory')
    parser.add_argument("--gpu_idx", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument('--multi_gpu', type=int, default=True, help='use point spatial transformer')

    # training parameters
    parser.add_argument('--nepoch', type=int, default=700, help='number of epochs to train for')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer Adam / SGD / rmsprop')
    parser.add_argument('--opt_eps', type=float, default=1e-08, help='optimizer epsilon')
    parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
    parser.add_argument('--patch_radius', type=float, default=[0.05], nargs='+', help='patch radius in multiples of the shape\'s bounding box diagonal, multiple values for multi-scale.')
    parser.add_argument('--patch_center', type=str, default='point', help='center patch at...\n'
                        'point: center point\n'
                        'mean: patch mean')
    parser.add_argument('--patch_point_count_std', type=float, default=0, help='standard deviation of the number of points in a patch')
    parser.add_argument('--patches_per_shape', type=int, default=1000, help='number of patches sampled from each shape in an epoch')
    parser.add_argument('--workers', type=int, default=6, help='number of data loading workers - 0 means same thread as main execution')
    parser.add_argument('--cache_capacity', type=int, default=100, help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')
    parser.add_argument('--seed', type=int, default=3627473, help='manual seed')
    parser.add_argument('--training_order', type=str, default='random', help='order in which the training patches are presented:\n'
                        'random: fully random over the entire dataset (the set of all patches is permuted)\n'
                        'random_shape_consecutive: random over the entire dataset, but patches of a shape remain consecutive (shapes and patches inside a shape are permuted)')
    parser.add_argument('--identical_epochs', type=int, default=False, help='use same patches in each epoch, mainly for debugging')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--scheduler_type', type=str, default='step', help='step or plateau')
    parser.add_argument('--momentum', type=float, default=0.9, help='gradient descent momentum')
    parser.add_argument('--lr_decay', type=int, default=True, help='learning rate')
    parser.add_argument('--lr_decay_interval', type=float, default=0, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_step_decay_epochs', type=str, default='300, 550', help='learning rate')
    parser.add_argument('--lr_clip', type=float, default=0.000001, help='learning rate')
    parser.add_argument('--normal_loss', type=str, default='sin', help='Normal loss type:\n'
                        'ms_euclidean: mean square euclidean distance\n'
                        'ms_oneminuscos: mean square 1-cos(angle error)\n'
                        'sin: mean sin(angle error)')
    parser.add_argument('--z_trans_weight', type=float, default=2, help='use z-direction transformation')
    parser.add_argument('--step_interval_to_print', type=int, default=100, help='step interval to print')
    parser.add_argument('--save_step', type=int, default=50, help='step interval to save model')


    # model hyperparameters
    parser.add_argument('--outputs', type=str, nargs='+', default=['unoriented_normals', 'neighbor_normals'], help='outputs of the network, a list with elements of:\n'
                        'unoriented_normals: unoriented (flip-invariant) point normals\n'
                        'oriented_normals: oriented point normals\n'
                        'max_curvature: maximum curvature\n'
                        'min_curvature: mininum curvature')
    parser.add_argument('--sym_op', type=str, default='max', help='symmetry operation')
    parser.add_argument('--point_tuple', type=int, default=1, help='use n-tuples of points as input instead of single points')

    parser.add_argument('--use_point_stn', type=int, default=True, help='use point spatial transformer')
    parser.add_argument('--use_feat_stn', type=int, default=True, help='use feature spatial transformer')
    parser.add_argument('--use_pca', type=int, default=True, help='use pca on point clouds, must be true for jet fit type')
    parser.add_argument('--n_gaussians', type=int, default=1, help='use feature spatial transformer')
    parser.add_argument('--jet_order', type=int, default=3, help='jet polynomial fit order')
    parser.add_argument('--points_per_patch', type=int, default=500, help='max. number of points per patch')
    parser.add_argument('--neighbor_search', type=str, default='k', help='[k | r] for k nearest and radius')
    parser.add_argument('--weight_mode', type=str, default="sigmoid", help='which function to use on the weight output: softmax, tanh, sigmoid')
    parser.add_argument('--use_consistency', type=int, default=True, help='flag to use consistency loss')
    parser.add_argument('--con_reg', type=str, default='log', help='choose consistency regularizer: mean, uniform')
    parser.add_argument('--k1', type=int, default=40, help='choose k1')
    parser.add_argument('--k2', type=int, default=20, help='choose k2')

    
    args = parser.parse_args()
    log_dir = os.path.join(args.logdir, args.name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if True:
        model_list = glob.glob(os.path.join(log_dir, '*.pth'))
        if len(model_list)>0:
            trained_epoch = [int(i.split('/')[-1].split('_')[0]) for i in model_list]
            args.start_epoch = max(trained_epoch)
            args.load_model = os.path.join(log_dir,'%s_network.pth' % args.start_epoch)
            print('load model from %s' % args.load_model)

        
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%a %d %b %Y %H:%M:%S', handlers=[logging.FileHandler(os.path.join(log_dir, 'train.log')),
                                                      logging.StreamHandler(sys.stdout)])
    train_pcpnet(args)



