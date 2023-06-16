from __future__ import print_function
import argparse
import os
import sys
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import importlib
import time
import munch
import yaml
import glob 
import logging


from pathlib import Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR_PATH = Path(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR_PATH, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
from dataset import PointcloudPatchDataset, SequentialShapeRandomPointcloudPatchSampler, RandomPointcloudPatchSampler, SequentialPointcloudPatchSampler


def test_n_est(opt):

    opt.models = opt.models.split()

    if opt.seed < 0:
        opt.seed = random.randint(1, 10000)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_dirname in opt.models:
       # fetch the model from the log dir

        # append model name to output directory and create directory if necessary
        model_log_dir =  os.path.join(opt.logdir, model_dirname, 'trained_models')
        model_list = glob.glob(os.path.join(opt.logdir, model_dirname, '*.pth'))
        if len(model_list)>0 and opt.start_epoch == 0:
            trained_epoch = [int(i.split('/')[-1].split('_')[0]) for i in model_list]
            opt.start_epoch = max(trained_epoch)
            model_filename = os.path.join(log_dir,'%s_network.pth' % opt.start_epoch)
        elif opt.start_epoch != 0:
            model_filename = os.path.join(log_dir,'%s_network.pth' % opt.start_epoch)

        print(model_filename)


        param_filename = os.path.join(model_log_dir, opt.parmpostfix)
        output_dir = os.path.join(opt.logdir, model_dirname, 'results_599')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Random Seed: %d" % (opt.seed))
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)

        # load model and training parameters

        trainopt = torch.load(param_filename)
        if not hasattr(trainopt, 'arch'):
            trainopt.arch = 'simple'

        if opt.batchSize == 0:
            opt.batchSize = trainopt.batchSize
        opt.batchSize = 128
        model_module = importlib.import_module('.%s' % trainopt.model_name, 'models')
        regressor = model_module.Net(trainopt) 
        ckpt = torch.load(model_filename)
        regressor.load_state_dict(ckpt['net_state_dict'])
        regressor.cuda()
        regressor.eval()

        # get indices in targets and predictions corresponding to each output
        target_features, output_target_ind, output_pred_ind, output_loss_weight, pred_dim = get_target_features((trainopt))
        dataloader, dataset, datasampler = get_data_loaders(opt, trainopt, target_features)


        shape_ind = 0
        shape_patch_offset = 0
        if opt.sampling == 'full':
            shape_patch_count = dataset.shape_patch_count[shape_ind]
        elif opt.sampling == 'sequential_shapes_random_patches':
            shape_patch_count = min(opt.patches_per_shape, dataset.shape_patch_count[shape_ind])
        else:
            raise ValueError('Unknown sampling strategy: %s' % opt.sampling)

        num_batch = len(dataloader)
        batch_enum = enumerate(dataloader, 0)

        shape_ind = 0
        normal_prop = torch.zeros([shape_patch_count, 3])
        avg_time = 0 

        for batchind, data in batch_enum:

            # get  batch and upload to GPU
            points = data[0]
            target = data[1:-2]
            data_trans = data[-2]
            n_effective_points = data[-1].squeeze()

            points = points.transpose(2, 1)
            points = points.to(device)
            data_trans = data_trans.to(device)
            target = tuple(t.to(device) for t in target)
            start_time = 0
            end_time = 0
            batch_size =  points.shape[0]
            with torch.no_grad():
                # if trainopt.arch == 'simple' or trainopt.arch == 'res' or trainopt.arch == '3dmfv':
                start_time = time.time()
                n_est, beta_pred, weights, trans, trans2, neighbor_normals, bias  = regressor(points)
                n_est = n_est[1] 
                end_time = time.time()
        

            if trainopt.use_point_stn:
                # transform predictions with inverse transform
                # since we know the transform to be a rotation (QSTN), the transpose is the inverse
                n_est[:, :] = torch.bmm(n_est.unsqueeze(1), trans.transpose(2, 1)).squeeze(dim=1)

            if trainopt.use_pca:
                # transform predictions with inverse pca rotation (back to world space)
                n_est[:, :] = torch.bmm(n_est.unsqueeze(1), data_trans.transpose(2, 1)).squeeze(dim=1)

            
            # if batchind % 50 ==0 or batchind==num_batch-1:
            if batchind % 50 ==0:
                logging.info('[%s %d/%d] shape %s' % (trainopt.model_name, batchind, num_batch-1, dataset.shape_names[shape_ind]))
                if avg_time!=0:
                    avg_time += 1000*(end_time-start_time) / opt.batchSize
                    avg_time = avg_time / 2. 
                else:
                    avg_time += 1000*(end_time-start_time) / opt.batchSize


                logging.info("elapsed_time per point: {} ms".format(1000*(end_time-start_time) / opt.batchSize))
            # Save estimated normals to file
            batch_offset = 0

            while batch_offset < n_est.shape[0] and shape_ind + 1 <= len(dataset.shape_names):
                shape_patches_remaining = shape_patch_count - shape_patch_offset
                batch_patches_remaining = n_est.shape[0] - batch_offset

                # append estimated patch properties batch to properties for the current shape on the CPU
                normal_prop[shape_patch_offset:shape_patch_offset + min(shape_patches_remaining,
                                                                        batch_patches_remaining), :] = \
                    n_est[batch_offset:batch_offset + min(shape_patches_remaining, batch_patches_remaining), :]

                batch_offset = batch_offset + min(shape_patches_remaining, batch_patches_remaining)
                shape_patch_offset = shape_patch_offset + min(shape_patches_remaining, batch_patches_remaining)

                if shape_patches_remaining <= batch_patches_remaining:
                    normals_to_write = normal_prop.cpu().numpy()
                    eps=1e-6
                    normals_to_write[np.logical_and(normals_to_write < eps, normals_to_write > -eps)] = 0.0
                    np.savetxt(os.path.join(output_dir, dataset.shape_names[shape_ind] + '.normals'),
                               normals_to_write)

                    logging.info('saved normals for ' + dataset.shape_names[shape_ind])
                    sys.stdout.flush()
                    shape_patch_offset = 0
                    shape_ind += 1
                    if shape_ind < len(dataset.shape_names):
                        shape_patch_count = dataset.shape_patch_count[shape_ind]
                        normal_prop = torch.zeros([shape_patch_count, 3])
        logging.info("avg elapsed_time per point: {} ms".format(avg_time))


def get_data_loaders(opt, trainopt, target_features):
    # create dataset loader
    if opt.batchSize == 0:
        model_batchSize = trainopt.batchSize
    else:
        model_batchSize = opt.batchSize

    test_dataset = PointcloudPatchDataset(
        root=opt.indir,
        shape_list_filename=opt.testset,
        patch_radius=trainopt.patch_radius,
        points_per_patch=trainopt.points_per_patch,
        patch_features=target_features,
        point_count_std=trainopt.patch_point_count_std,
        seed=opt.seed,
        identical_epochs=trainopt.identical_epochs,
        use_pca=trainopt.use_pca,
        center=trainopt.patch_center,
        point_tuple=trainopt.point_tuple,
        sparse_patches=opt.sparse_patches,
        cache_capacity=opt.cache_capacity,
        neighbor_search_method=trainopt.neighbor_search)
    if opt.sampling == 'full':
        test_datasampler = SequentialPointcloudPatchSampler(test_dataset)
    elif opt.sampling == 'sequential_shapes_random_patches':
        test_datasampler = SequentialShapeRandomPointcloudPatchSampler(
            test_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            sequential_shapes=True,
            identical_epochs=False)
    else:
        raise ValueError('Unknown sampling strategy: %s' % opt.sampling)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        sampler=test_datasampler,
        batch_size=model_batchSize,
        num_workers=int(opt.workers))
    return test_dataloader, test_dataset, test_datasampler


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

    return target_features, output_target_ind, output_pred_ind, output_loss_weight, pred_dim

def l2_norm(v):
    norm_v = np.sqrt(np.sum(np.square(v), axis=1))
    return norm_v


def evaluate(opt):
    FLAGS = opt
    PC_PATH = os.path.join(BASE_DIR, FLAGS.indir)
    normal_results_path = os.path.join(FLAGS.script_path, 'results_599')
    results_path = os.path.abspath(os.path.join(normal_results_path, os.pardir))
    sparse_patches = FLAGS.sparse_patches
    dataset_list = FLAGS.dataset_list

    for dataset in dataset_list:

        normal_gt_filenames = PC_PATH + dataset + '.txt'
        normal_gt_path = PC_PATH

        # get all shape names in the dataset
        shape_names = []
        with open(normal_gt_filenames) as f:
            shape_names = f.readlines()
        shape_names = [x.strip() for x in shape_names]
        shape_names = list(filter(None, shape_names))

        outdir = os.path.join(normal_results_path, 'summary/')
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        LOG_FOUT = open(os.path.join(outdir, dataset + '_%s_network_evaluation_results.txt' % opt.start_epoch), 'w')
        # if EXPORT:
        #     file_path = os.path.join(normal_results_path, 'images')
        #     if not os.path.exists(file_path):
        #         os.makedirs(file_path)

        def log_string(out_str):
            LOG_FOUT.write(out_str+'\n')
            LOG_FOUT.flush()
            print(out_str)

        experts_exist = False
        rms = []
        rms_o = []
        all_ang = []
        pgp10 = []
        pgp5 = []
        pgp_alpha = []

        for i, shape in enumerate(shape_names):
            print('Processing ' + shape + '...')

            # if EXPORT:
            #     # Organize the output folders
            #     idx_1 = shape.find('_noise_white_')
            #     idx_2 = shape.find('_ddist_')
            #     if idx_1 == -1 and idx_2 == -1:
            #         base_shape_name = shape
            #     elif idx_1 == -1:
            #         base_shape_name = shape[:idx_2]
            #     else:
            #         base_shape_name = shape[:idx_1]

            #     vis_output_path = os.path.join(file_path, base_shape_name)
            #     if not os.path.exists(vis_output_path):
            #         os.makedirs(vis_output_path)
            #     gt_normals_vis_output_path = os.path.join(vis_output_path, 'normal_gt')
            #     if not os.path.exists(gt_normals_vis_output_path):
            #         os.makedirs(gt_normals_vis_output_path)
            #     pred_normals_vis_output_path = os.path.join(vis_output_path, 'normal_pred')
            #     if not os.path.exists(pred_normals_vis_output_path):
            #         os.makedirs(pred_normals_vis_output_path)
            #     phi_teta_vis_output_path = os.path.join(vis_output_path, 'phi_teta_domain')
            #     if not os.path.exists(phi_teta_vis_output_path):
            #         os.makedirs(phi_teta_vis_output_path)

            # load the data
            points = np.loadtxt(os.path.join(normal_gt_path, shape + '.xyz')).astype('float32')
            normals_gt = np.loadtxt(os.path.join(normal_gt_path, shape + '.normals')).astype('float32')
            normals_results = np.loadtxt(os.path.join(normal_results_path, shape + '.normals')).astype('float32')
            points_idx = np.loadtxt(os.path.join(normal_gt_path, shape + '.pidx')).astype('int')

            n_points = points.shape[0]
            n_normals = normals_results.shape[0]
            if n_points != n_normals:
                sparse_normals = True
            else:
                sparse_normals = False

            points = points[points_idx, :]
            normals_gt = normals_gt[points_idx, :]
            # curvs_gt = curvs_gt[points_idx, :]
            if sparse_patches and not sparse_normals:
                normals_results = normals_results[points_idx, :]
            else:
                normals_results = normals_results[:, :]

            normal_gt_norm = l2_norm(normals_gt)
            normal_results_norm = l2_norm(normals_results)
            normals_results = np.divide(normals_results, np.tile(np.expand_dims(normal_results_norm, axis=1), [1, 3]))
            normals_gt = np.divide(normals_gt, np.tile(np.expand_dims(normal_gt_norm, axis=1), [1, 3]))

            # Not oriented rms
            nn = np.sum(np.multiply(normals_gt, normals_results), axis=1)
            nn[nn > 1] = 1
            nn[nn < -1] = -1

            ang = np.rad2deg(np.arccos(np.abs(nn)))  #  unoriented

            # error metrics
            rms.append(np.sqrt(np.mean(np.square(ang))))
            pgp10_shape = sum([j < 10.0 for j in ang]) / float(len(ang))  # portion of good points
            pgp5_shape = sum([j < 5.0 for j in ang]) / float(len(ang))  # portion of good points
            pgp10.append(pgp10_shape)
            pgp5.append(pgp5_shape)
            pgp_alpha_shape = []
            for alpha in range(30):
                pgp_alpha_shape.append(sum([j < alpha for j in ang]) / float(len(ang)))

            pgp_alpha.append(pgp_alpha_shape)

            # Oriented rms
            rms_o.append(np.sqrt(np.mean(np.square(np.rad2deg(np.arccos(nn))))))

            diff = np.arccos(nn)
            diff_inv = np.arccos(-nn)
            unoriented_normals = normals_results
            unoriented_normals[diff_inv < diff, :] = -normals_results[diff_inv < diff, :]

        avg_rms = np.mean(rms)
        avg_rms_o = np.mean(rms_o)
        avg_pgp10 = np.mean(pgp10)
        avg_pgp5 = np.mean(pgp5)
        avg_pgp_alpha = np.mean(np.array(pgp_alpha), axis=0)

        log_string('RMS per shape: ' + str(rms))
        log_string('RMS not oriented (shape average): ' + str(avg_rms))
        log_string('RMS oriented (shape average): ' + str(avg_rms_o))
        log_string('PGP10 per shape: ' + str(pgp10))
        log_string('PGP5 per shape: ' + str(pgp5))
        log_string('PGP10 average: ' + str(avg_pgp10))
        log_string('PGP5 average: ' + str(avg_pgp5))
        log_string('PGP alpha average: ' + str(avg_pgp_alpha))
        LOG_FOUT.close()



if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Test config file')
        # naming / file handling
    parser.add_argument('--indir', type=str, default='/data1/PU_project/GraphFit-master/data/pcpnet/', help='input folder (point clouds)')
    parser.add_argument('--testset', type=str, default='testset_no_noise.txt', help='shape set file name')
    parser.add_argument('--models', type=str, default='deepfit', help='names of trained models, can evaluate multiple models')
    parser.add_argument('--modelpostfix', type=str, default='_network.pth', help='model file postfix')
    parser.add_argument('--logdir', type=str, default='./workspace/', help='model folder')
    parser.add_argument('--parmpostfix', type=str, default='params.pth', help='parameter file postfix')
    parser.add_argument("--gpu_idx", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument('--multi_gpu', type=int, default=True, help='use point spatial transformer')
    parser.add_argument('--sparse_patches', type=int, default=True, help='evaluate on a sparse set of patches, given by a .pidx file containing the patch center point indices.')
    parser.add_argument('--sampling', type=str, default='full', help='sampling strategy, any of:\n'
                        'full: evaluate all points in the dataset\n'
                        'sequential_shapes_random_patches: pick n random points from each shape as patch centers, shape order is not randomized')
    parser.add_argument('--patches_per_shape', type=int, default=1000, help='number of patches evaluated in each shape (only for sequential_shapes_random_patches)')
    parser.add_argument('--seed', type=int, default=40938661, help='manual seed')
    parser.add_argument('--start_epoch', type=int, default=0, help='manual seed')
    parser.add_argument('--batchSize', type=int, default=256, help='batch size, if 0 the training batch size is used')
    parser.add_argument('--workers', type=int, default=6, help='number of data loading workers - 0 means same thread as main execution')
    parser.add_argument('--cache_capacity', type=int, default=100, help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')
    parser.add_argument('--script_path', default='/data1/VRCNet-main/log1/asfm_cd_com3d_0_0_0_HW0_HM0', help='path to config file')
    parser.add_argument('--dataset_list', type=str, default=['testset_no_noise'], nargs='+',
                        help='list of .txt files containing sets of point cloud names for evaluation')


    arg = parser.parse_args()
    log_dir = arg.script_path
   
    eval_opt = arg
    eval_opt.models = os.path.basename(log_dir)
    eval_opt.logdir = os.path.dirname(log_dir)
    print(eval_opt)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%a %d %b %Y %H:%M:%S', handlers=[logging.FileHandler(os.path.join(log_dir, 'test.log')),
                                                      logging.StreamHandler(sys.stdout)])
    test_n_est(eval_opt)
    evaluate(eval_opt)
