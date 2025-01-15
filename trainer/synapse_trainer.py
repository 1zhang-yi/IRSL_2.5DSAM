import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn.functional as F
from tqdm import tqdm
from utils import DiceLoss, FocalDiceloss_IoULoss, generate_point, setting_prompt_none
from metrics import SegMetrics


def to_device(batch_input):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key == 'image' or key == 'label' or key == 'point_coords' or key == 'point_labels':
                device_input[key] = value.float().cuda()
            elif type(value) is list or type(value) is torch.Size:
                 device_input[key] = value
            else:
                device_input[key] = value.cuda()
        else:
            device_input[key] = value
    return device_input

def prompt_and_decoder(args, batched_input, model, image_embeddings, low_masks, image_res, decoder_iter=False):
    if batched_input["point_coords"] is not None:
        if len(batched_input["point_coords"].shape) == 4:
            points = (batched_input["point_coords"][0], batched_input["point_labels"][0])
        else:
            points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    if decoder_iter:
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=points,
                boxes=batched_input.get("boxes", None),
                masks=batched_input.get("mask_inputs", None),
            )

    else:
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
        )

    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_res=image_res,
        low_masks=low_masks,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=args.multimask,
    )

    if args.multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i + 1, idx])
        low_res_masks = torch.stack(low_res, 0)

    masks = F.interpolate(low_res_masks, (args.img_size, args.img_size), mode="bilinear", align_corners=False, )
    return masks, low_res_masks, iou_predictions


def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion):
    train_loader = tqdm(train_loader)
    train_losses = []
    train_iter_metrics = [0] * len(args.metrics)
    for batch, batched_input in enumerate(train_loader):
        batched_input = to_device(batched_input)
        labels = batched_input['label'][0]
        batched_image = batched_input['image'][0]
        batched_image_res = batched_input['image_res'][0]

        image_embeddings, image_res_embeddings = model.image_encoder(batched_image, batched_image_res)

        masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings,
                                                                   low_masks=None,
                                                                   image_res=None,
                                                                   decoder_iter=False)
        mask_i = masks[2: 3]
        label_i = labels[2: 3]
        loss = criterion(mask_i, label_i, iou_predictions)
        loss.backward(retain_graph=True)

        optimizer.step()
        optimizer.zero_grad()

        if int(batch + 1) % 50 == 0:
            logging.info(
                f'Epoch: {epoch + 1}, Batch: {batch + 1}, first point prompt: {SegMetrics(masks, labels, args.metrics)}')

        point_num = random.choice(args.point_list)
        batched_input = generate_point(masks, labels, low_res_masks, batched_input, point_num)
        batched_input = to_device(batched_input)

        image_embeddings = image_embeddings.detach().clone()
        image_res = image_res_embeddings.detach().clone()

        # image_embeddings_copy = torch.zeros((image_embeddings.shape)).cuda()
        # image_embeddings_copy[1:, :, :, :] = image_embeddings[0:image_embeddings.shape[0]-1, :, :, :]
        # image_feature_res = image_embeddings - image_embeddings_copy
        # image_feature_res[0, :, :, :] = 0
        #
        # res = image_res + image_feature_res

        init_mask_num = np.random.randint(1, args.iter_point - 1)
        for iter in range(args.iter_point):
            if iter == init_mask_num or iter == args.iter_point - 1:
                batched_input = setting_prompt_none(batched_input)

            low_masks = low_res_masks.detach().clone()
            if iter == int(args.iter_point // 2):
                masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings,
                                                                           low_masks=None,
                                                                           image_res=None,
                                                                           decoder_iter=True)
            else:
                masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings,
                                                                           low_masks=low_masks,
                                                                           image_res=image_res,
                                                                           decoder_iter=True)

            loss = criterion(masks[2: 3], labels[2: 3], iou_predictions)
            loss.backward(retain_graph=True)

            optimizer.step()
            optimizer.zero_grad()

            if iter != args.iter_point - 1:
                point_num = random.choice(args.point_list)
                batched_input = generate_point(masks, labels, low_res_masks, batched_input, point_num)
                batched_input = to_device(batched_input)

            if int(batch + 1) % 50 == 0:
                if iter == init_mask_num or iter == args.iter_point - 1:
                    logging.info(
                        f'Epoch: {epoch + 1}, Batch: {batch + 1}, mask prompt: {SegMetrics(masks, labels, args.metrics)}')
                else:
                    logging.info(
                        f'Epoch: {epoch + 1}, Batch: {batch + 1}, point {point_num} prompt: {SegMetrics(masks, labels, args.metrics)}')

        # if int(batch + 1) % 200 == 0:
        #     print(f"epoch:{epoch + 1}, iteration:{batch + 1}, loss:{loss.item()}")
        #     save_path = os.path.join(f"{args.work_dir}/models", args.run_name,
        #                              f"epoch{epoch + 1}_batch{batch + 1}_sam.pth")
        #     state = {'model': model.state_dict(), 'optimizer': optimizer}
        #     torch.save(state, save_path)


        train_losses.append(loss.item())

        # gpu_info = {}
        # gpu_info['gpu_name'] = args.device
        # train_loader.set_postfix(train_loss=loss.item(), gpu_info=gpu_info)

        train_batch_metrics = SegMetrics(masks[2: 3], labels[2: 3], args.metrics)
        train_iter_metrics = [train_iter_metrics[i] + train_batch_metrics[i] for i in range(len(args.metrics))]

    return train_losses, train_iter_metrics


def trainer_synapse(args, model, train_loader, snapshot_path):

    Sam = model.sam
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr

    criterion = FocalDiceloss_IoULoss()
    l = len(train_loader)
    if args.n_gpu > 1:
        Sam = nn.DataParallel(Sam)

    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr

    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, Sam.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, Sam.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)

    best_dice = 0.0
    for epoch in range(args.max_epochs):
        Sam.train()
        train_losses, train_iter_metrics = train_one_epoch(args, Sam, optimizer, train_loader, epoch, criterion)
        train_iter_metrics = [metric / l for metric in train_iter_metrics]
        print(train_iter_metrics)
        train_metrics = {args.metrics[i]: '{:.4f}'.format(train_iter_metrics[i]) for i in
                         range(len(train_iter_metrics))}

        epoch_dice = train_metrics['dice']
        epoch_dice = float(epoch_dice)
        average_loss = np.mean(train_losses)
        logging.info(f"epoch: {epoch + 1}, lr: {b_lr}, Train loss: {average_loss:.4f}, metrics: {train_metrics}")

        if epoch_dice >= best_dice:
            best_dice = epoch_dice
            save_mode_path = os.path.join(snapshot_path, 'best_epoch_' + str(epoch) + '_' + str(epoch_dice) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("Finding the better model, save model to {}".format(save_mode_path))

        save_interval = 1 # int(max_epoch/6)
        if (epoch + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch) + '_' + str(epoch_dice) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch >= args.max_epochs - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch) + '_' + str(epoch_dice) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            break


    return "Training Finished!"