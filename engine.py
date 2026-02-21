import math
import sys
import os
import shutil           # shell utilities -> high-level file and directory operations
import torch
import cv2
import copy
import numpy as np
import torch_fidelity   # a pytorch-based toolkit for evaluating generative models, especially GANs and diffusion models
import utils.misc as misc
import utils.lr_scheduler as lr_scheduler

"""
    Here, a workflow summary:
    - Training: use model (with distributed data parallel, DDP) -> compute gradients -> update parameters
    - EMA update: update model_without_ddp with EMA of model parameters
    - Evaluation: Run interference with model_without_ddp (EMA version) to ensure more stable and consistent metrics

    The DDP wrapper itself is only necessary during gradient computation. 
    For inference/evaluation, there is no need for gradient syncing, model_without_ddp will be used.
"""

def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    """
        train the model for one epoch
    """
    model.train()
    metric_logger = misc.MetricLogger(delimiter="   ")  # delimiter = "\t"
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    optimizer.zero_grad()
    print_freq = 20

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (x, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # learning rate scheduler per iteration
        lr_scheduler.adjust_lr(optimizer, epoch + data_iter_step / len(data_loader), args)
        
        # normalize image to [-1, 1]
        x = x.to(device, non_blocking=True).to(torch.float32).div_(255)     # non_blocking=True allows async transfer when using pinned memory
        x = x * 2.0 - 1.0   # from [0, 1] to [-1, 1]
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):  # enable automatic mixed precision (AMP), run autocast on GPU, use bfloat16 precision for supported ops
            loss = model(x, labels)

        loss_value = loss.item()
        if not math.isfinite(loss_value):   # when loss_value == NaN, +inf, -inf
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)     # stop training and handle it

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()    # block CPU until all queued CUDA operations on GPU are finished
        model_without_ddp.update_ema()  # update the EMA of the model parameters to smooth or stabilize the model's weights over training steps
        
        # add loss_value and lr into deque (double-ended queue)
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # calculate mean value across multiple GPUs (processes)
        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            # use epoch_1000x as x-axis in TensorBoard to calibrate curves
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)   # a smoothed integer progress counter in units of 0.001 epochs
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)

def evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None):
    """
        generate images and computes FID / IS for evaluation
    """
    model_without_ddp.eval()
    world_size = misc.get_world_size()          # get total number of processes
    local_rank = misc.get_rank()                # get the unique ID of the current process, ranging from 0 to world_size - 1
    num_steps = args.num_images // (batch_size * world_size) + 1

    # generated images saving path - parameter settings from denoiser.py
    save_folder = os.path.join(
        args.output_dir,
        "{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1], args.num_images, args.img_size
        )
    )
    print(">>> Save to:", save_folder)
    if misc.get_rank() == 0 and not os.path.exists(save_folder):
        os.makedirs(save_folder)    # only the main process should create directories to avoid race conditions

    # switch the model's weights to the EMA (Exponential Moving Average) version before evaluation or saving
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())    # original weights
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())      # will be overwritten with EMA weights
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    print(">>> Switch to EMA weights")
    model_without_ddp.load_state_dict(ema_state_dict)

    # ensure that the number of generated images per class is equal
    class_num = args.class_num
    assert args.num_images % class_num == 0, "Number of generated images per class must be the same"
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(args.num_images)])

    for i in range(num_steps):
        print(">>> Generation step {}/{}".format(i, num_steps))

        global_batch = world_size * batch_size
        start_idx = global_batch * i + local_rank * batch_size
        end_idx = start_idx + batch_size
        labels_gen = torch.as_tensor(class_label_gen_world[start_idx:end_idx], 
                                     dtype=torch.long, 
                                     device="cuda")
        
        # run forward pass in bfloat16 mixed precision and keep certain ops in FP32, good choice on Ampere+ GPUs (A100, H100, RTX 30/40 series)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):  
            sampled_images = model_without_ddp.generate(labels_gen)     # generate images

        torch.distributed.barrier()     # force all processes to wait until everyone reaches this line

        # denormalize generated images
        sampled_images = (sampled_images + 1) / 2   # [-1, 1] -> [0, 1]
        sampled_images = sampled_images.detach().cpu()  # sampled_images -> (batch_size, channels, height, width)

        # compute global image IDs and save generated images
        for idx_in_batch in range(sampled_images.size(0)):     # loop over each generated image inside the current batch
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + idx_in_batch
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip(sampled_images[idx_in_batch].numpy().transpose([1, 2, 0]) * 255, 0, 255))    # gen_img -> shape: (H, W, C), pixel range: [0, 255]
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]      # reverse the third dimension (channels) from RGB -> BGR for OpenCV operations
            cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)  # save generated image

    torch.distributed.barrier()
       
    # back to original weights (no ema version)
    print(">>> Switch back to original weights...")
    model_without_ddp.load_state_dict(model_state_dict)

    # compute FID and IS
    """
        FID (Fréchet Inception Distance) compares real images vs generated images in feature space, it measures:
            - image quality
            - diversity
            - similarity to real data distribution
        IS (Inception Score) evaluates image quality and diversity
    """
    