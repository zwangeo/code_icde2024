import numpy as np
import logging
import time
import os
import socket


def set_up_log(args, sys_argv):
    if args.gnn_type == 'gin':
        args.backbone = args.input_model_file.split('/')[0].split('_')[-1]
        args.obj = args.input_model_file.split('/')[-1].split('.')[0]
    else:
        args.backbone = args.gnn_type
        args.obj = args.input_model_file.split('/')[-1].split('.')[0].split('_')[-1]
    args.pretrained_model = '_'.join([args.backbone, args.obj])
    args.dataset = args.dataset.lower()
    for _ in [args.log_dir,
              os.path.join(args.log_dir, args.pretrained_model),
              os.path.join(args.log_dir, args.pretrained_model, args.dataset)]:
        if not os.path.exists(_):
            os.mkdir(_)

    args.file_path = os.path.join(args.log_dir, args.pretrained_model, args.dataset, '{}.log'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(args.file_path)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info('Create log file at {}'.format(args.file_path))
    logger.info('Command line executed: python ' + ' '.join(sys_argv))
    logger.info('Full args parsed:')
    logger.info(args)
    return logger


def save_performance_result(args, logger, model, time_recorder=[0, 0]):
    # summary_dir = os.path.join(args.log_dir, 'summary')
    summary_dir = os.path.join(args.log_dir, args.pretrained_model, 'summary')
    if not os.path.exists(summary_dir):
        os.mkdir(summary_dir)
    summary_path = os.path.join(summary_dir, f'{args.dataset}_{args.summary_file}')
    log_name = os.path.split(logger.handlers[1].baseFilename)[-1]
    server = socket.gethostname()

    line = '\t'.join([f'runseed: {args.runseed}',
                      f'dataset: {args.dataset}',
                      # f'pretrained_model: {args.pretrained_model}',
                      f'best_test_w_val: {model.best_test_w_val}',
                      f'best_test_wo_val: {model.best_test_wo_val}',
                      f'w_val_epoch: {model.best_test_w_val_epoch}',
                      f'wo_val_epoch: {model.best_test_wo_val_epoch}',
                      f'backbone: {args.backbone}',
                      f'obj: {args.obj}',
                      f'temp: {args.temp}',
                      f'search_adapter: {model.search_adapter}',
                      f'search_jk: {model.search_jk}',
                      f'search_pool: {model.search_pool}',
                      f"adapter: {'--'.join(model.searched_arch['ADAPTER'])}",
                      f"jk: {model.searched_arch['JK']}",
                      f"pool: {model.searched_arch['POOL']}",
                      log_name,
                      server,
                      f'bs: {args.batch_size}',
                      f'max_norm: {args.max_norm}',
                      f'ft_mode: {args.ft_mode}',
                      f't_search: {time_recorder[0]}',
                      f't_retrain: {time_recorder[1]}',
                      f'adapter_dim: {args.adapter_dim}',
                      f'disable_pretrain: {args.disable_pretrain}']) + '\n'

    with open(summary_path, 'a') as f:
        f.write(line)  # WARNING: process unsafe!
    f.close()

    # shut_down_log to avoid writing all in the first log
    # refer to https://blog.csdn.net/weixin_41956627/article/details/125784000?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2
    logger.warning(f'Shut down {log_name}')
    logger.handlers.clear()
