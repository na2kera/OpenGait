
import os
import argparse
import re
import torch
import torch.nn as nn
from modeling import models
from utils import config_loader, get_ddp_module, init_seeds, params_count, get_msg_mgr

parser = argparse.ArgumentParser(description='Main program for opengait.')
parser.add_argument('--local_rank', type=int, default=0,
                    help="passed by torch.distributed.launch module")
parser.add_argument('--local-rank', type=int, default=0,
                    help="passed by torch.distributed.launch module, for pytorch >=2.0")
parser.add_argument('--cfgs', type=str,
                    default='config/default.yaml', help="path of config file")
parser.add_argument('--phase', default='train',
                    choices=['train', 'test'], help="choose train or test phase")
parser.add_argument('--log_to_file', action='store_true',
                    help="log to file, default path is: output/<dataset>/<model>/<save_name>/<logs>/<Datetime>.txt")
parser.add_argument('--iter', type=int, default=0, help="iter to restore")
parser.add_argument('--allow-train-iter-override', action='store_true',
                    help="allow a positive --iter to replace a string warm-start checkpoint during training")
parser.add_argument('--base-seed', type=int, default=None,
                    help="base random seed; overrides base_seed in the config")
opt = parser.parse_args()


def resolve_base_seed(cfgs, save_name):
    base_seed = cfgs.get('base_seed', 0) if opt.base_seed is None else opt.base_seed
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a non-negative integer")
    seed_suffix = re.search(r'-s([0-9]+)$', save_name)
    if opt.base_seed is not None and seed_suffix is None:
        raise ValueError(
            "--base-seed requires save_name to end with a matching -sN suffix")
    if seed_suffix is not None and int(seed_suffix.group(1)) != base_seed:
        raise ValueError(
            "base_seed ({}) must match the save_name seed suffix: {}".format(
                base_seed, save_name))
    return base_seed


def find_existing_training_checkpoints(cfgs):
    trainer_cfg = cfgs['trainer_cfg']
    save_name = trainer_cfg['save_name']
    checkpoint_dir = os.path.join(
        'output', cfgs['data_cfg']['dataset_name'],
        cfgs['model_cfg']['model'], save_name, 'checkpoints')
    if not os.path.isdir(checkpoint_dir):
        return []
    prefix = '{}-'.format(save_name)
    with os.scandir(checkpoint_dir) as entries:
        return sorted(
            entry.path for entry in entries
            if (entry.is_file() and entry.name.startswith(prefix) and
                entry.name.endswith('.pt')))


def apply_iter_override(cfgs):
    if opt.iter < 0:
        raise ValueError("--iter must be a non-negative integer")
    if opt.allow_train_iter_override and opt.iter == 0:
        raise ValueError(
            "--allow-train-iter-override requires a positive --iter N")
    trainer_cfg = cfgs['trainer_cfg']
    warm_start = isinstance(trainer_cfg['restore_hint'], str)
    if opt.iter == 0:
        if opt.phase == 'train' and warm_start:
            existing = find_existing_training_checkpoints(cfgs)
            if existing:
                raise FileExistsError(
                    "Refusing to start new warm-start training because "
                    "existing checkpoints were found for save_name '{}': "
                    "{}. Resume explicitly or use a new save_name.".format(
                        trainer_cfg['save_name'], existing[0]))
        return False
    training_resume = opt.phase == 'train' and warm_start
    if opt.allow_train_iter_override and not training_resume:
        raise ValueError(
            "--allow-train-iter-override is only valid for training with a "
            "string warm-start checkpoint")
    if (training_resume and
            not opt.allow_train_iter_override):
        raise ValueError(
            "Refusing to replace the training warm-start checkpoint with "
            "--iter. Omit --iter for a new fine-tuning run, or also pass "
            "--allow-train-iter-override to resume intentionally.")
    if training_resume:
        if opt.iter >= trainer_cfg['total_iter']:
            raise ValueError(
                "Training resume --iter ({}) must be less than total_iter "
                "({})".format(opt.iter, trainer_cfg['total_iter']))
        trainer_cfg['optimizer_reset'] = False
        trainer_cfg['scheduler_reset'] = False
        trainer_cfg['restore_ckpt_strict'] = True
        trainer_cfg['require_training_state'] = True
        trainer_cfg['expected_training_iteration'] = opt.iter
    cfgs['evaluator_cfg']['restore_hint'] = opt.iter
    cfgs['trainer_cfg']['restore_hint'] = opt.iter
    return training_resume


def initialization(cfgs, training, training_resume=False):
    msg_mgr = get_msg_mgr()
    engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
    base_seed = resolve_base_seed(cfgs, engine_cfg['save_name'])
    output_path = os.path.join('output/', cfgs['data_cfg']['dataset_name'],
                               cfgs['model_cfg']['model'], engine_cfg['save_name'])
    if training:
        msg_mgr.init_manager(output_path, opt.log_to_file, engine_cfg['log_iter'],
                             engine_cfg['restore_hint'] if isinstance(engine_cfg['restore_hint'], (int)) else 0)
    else:
        msg_mgr.init_logger(output_path, opt.log_to_file)

    msg_mgr.log_info(engine_cfg)
    if training_resume:
        msg_mgr.log_warning(
            f"Resuming training from iteration {opt.iter} with model, "
            "optimizer, scheduler, and AMP scaler state. RNG and data-loader "
            "state are not restored, so "
            "the resumed run is not bit-identical to an uninterrupted run.")

    rank = torch.distributed.get_rank()
    seed = base_seed + rank
    msg_mgr.log_info(
        "Initialize random seed: base_seed={}, rank={}, seed={}".format(
            base_seed, rank, seed))
    init_seeds(seed)


def run_model(cfgs, training):
    msg_mgr = get_msg_mgr()
    model_cfg = cfgs['model_cfg']
    msg_mgr.log_info(model_cfg)
    Model = getattr(models, model_cfg['model'])
    model = Model(cfgs, training)
    if training and cfgs['trainer_cfg']['sync_BN']:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if cfgs['trainer_cfg']['fix_BN']:
        model.fix_BN()
    model = get_ddp_module(model, cfgs['trainer_cfg']['find_unused_parameters'])
    msg_mgr.log_info(params_count(model))
    msg_mgr.log_info("Model Initialization Finished!")

    if training:
        Model.run_train(model)
    else:
        Model.run_test(model)


if __name__ == '__main__':
    cfgs = config_loader(opt.cfgs)
    training_resume = apply_iter_override(cfgs)

    torch.distributed.init_process_group('nccl', init_method='env://')
    if torch.distributed.get_world_size() != torch.cuda.device_count():
        raise ValueError("Expect number of available GPUs({}) equals to the world size({}).".format(
            torch.cuda.device_count(), torch.distributed.get_world_size()))
    training = (opt.phase == 'train')
    initialization(cfgs, training, training_resume)
    run_model(cfgs, training)
