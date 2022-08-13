


import os
import datetime
import argparse
import torch
from torch.utils import data
from utils import transforms
from networks import MaskRCNN
from backbone import resnet50_fpn_backbone
from utils.coco_dataset import CocoDetection
from train_utils import train_eval_utils as utils
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups




def create_model(num_classes, load_pretrain_weights=True):
    # 如果GPU显存很小，batch_size不能设置很大，建议将norm_layer设置成 FrozenBatchNorm2d(默认是nn.BatchNorm2d)
    # FrozenBatchNorm2d的功能与BatchNorm2d类似，但参数无法更新
    # trainable_layers包括['layer4', 'layer3', 'layer2', 'layer1', 'conv1']， 5代表全部训练
    # backbone = resnet50_fpn_backbone(norm_layer=FrozenBatchNorm2d, trainable_layers=3)
    # resnet50 imagenet weights url: https://download.pytorch.org/models/resnet50-0676ba61.pth

    backbone = resnet50_fpn_backbone(pretrain_path="weights/resnet50.pth", trainable_layers=3)
    model    = MaskRCNN(backbone, num_classes=num_classes)

    if load_pretrain_weights:
        # coco weights url: "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
        weights_dict = torch.load("./weights/maskrcnn_resnet50_fpn_coco.pth", map_location="cpu")
        for k in list(weights_dict.keys()):
            if ("box_predictor" in k) or ("mask_fcn_logits" in k):
                del weights_dict[k]

        print(model.load_state_dict(weights_dict, strict=False))
    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    # 用来保存coco_info的文件
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    det_results_file = f"det_results{now}.txt"
    seg_results_file = f"seg_results{now}.txt"

    data_transform = {"train": transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(0.5)]),
                      "val"  : transforms.Compose([transforms.ToTensor()])}

    # load train data set: coco2017 -> annotations -> instances_train2017.json
    train_dataset = CocoDetection(args.data_path, "train", data_transform["train"])

    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> train.txt
    # train_dataset = VOCInstances(data_root, year="2012", txt_name="train.txt", transforms=data_transform["train"])
    train_sampler = None

    # 是否按图片相似高宽比采样图片组成batch
    # 使用的话能够减小训练时所需GPU显存，默认使用
    if args.aspect_ratio_group_factor >= 0:
        train_sampler = data.RandomSampler(train_dataset)
        # 统计所有图像高宽比例在bins区间中的位置索引
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        # 每个batch图片从同一高宽比例区间中取
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = args.batch_size
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    nw = 1
    print('Using %g dataloader workers' % nw)

    if train_sampler:
        # 如果按照图片高宽比采样图片，dataloader中需要使用batch_sampler
        train_data_loader = data.DataLoader(train_dataset, batch_sampler=train_batch_sampler,
                                            pin_memory=True, num_workers=nw, collate_fn=train_dataset.collate_fn)
    else:
        train_data_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                            pin_memory=True, num_workers=nw, collate_fn=train_dataset.collate_fn)

    # load validation data set coco2017 -> annotations -> instances_val2017.json
    val_dataset = CocoDetection(args.data_path, "val", data_transform["val"])

    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
    # val_dataset = VOCInstances(data_root, year="2012", txt_name="val.txt", transforms=data_transform["val"])
    val_data_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True,
                                      num_workers=nw, collate_fn=train_dataset.collate_fn)

    # create model num_classes equal background + classes
    model = create_model(num_classes=args.num_classes + 1, load_pretrain_weights=args.pretrain)
    model.to(device)

    train_loss    = []
    learning_rate = []
    val_map       = []

    # define optimizer
    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum) #, weight_decay=args.weight_decay)
    scaler    = torch.cuda.amp.GradScaler() if args.amp else None

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    # 如果传入resume参数，即上次训练的权重地址，则接着上次的参数训练
    if args.resume:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        checkpoint = torch.load(args.resume, map_location='cpu')  # 读取之前保存的权重文件(包括优化器以及学习率策略)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch, printing every 50 iterations
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader, device, epoch,
                                              print_freq=50, warmup=True, scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        lr_scheduler.step()

        # det_info, seg_info = utils.evaluate(model, val_data_loader, device=device)

        # # write detection into txt
        # with open(det_results_file, "a") as f:
        #     # 写入的数据包括coco指标还有loss和learning rate
        #     result_info = [f"{i:.4f}" for i in det_info + [mean_loss.item()]] + [f"{lr:.6f}"]
        #     txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
        #     f.write(txt + "\n")
        #
        # # write seg into txt
        # with open(seg_results_file, "a") as f:
        #     # 写入的数据包括coco指标还有loss和learning rate
        #     result_info = [f"{i:.4f}" for i in seg_info + [mean_loss.item()]] + [f"{lr:.6f}"]
        #     txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
        #     f.write(txt + "\n")
        #
        # val_map.append(det_info[1])  # pascal mAP

        # save weights
        save_files = {  'model'       : model.state_dict(),
                        'optimizer'   : optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch'       : epoch}
        if args.amp:
            save_files["scaler"] = scaler.state_dict()
        torch.save(save_files, "./weights/model_{}.pth".format(epoch))

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from utils.plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        from utils.plot_curve import plot_map
        plot_map(val_map)




if __name__ == "__main__":
    parser = argparse.ArgumentParser( description=__doc__)
    parser.add_argument('--device',     default='cuda:0', help='device, 0 is cpu')
    parser.add_argument('--data-path',  default='E:/Data/COCO/coco2017', help='dataset')
    parser.add_argument('--num-classes', default=90, type=int, help='num_classes 检测目标类别数(不包含背景)')
    parser.add_argument('--output-dir', default='./weights', help='path where to save')
    parser.add_argument('--resume',     default='', type=str, help='resume from checkpoint')    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--epochs',     default=10, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr',         default=0.004, type=float, help='initial learning rate, 0.02 is the default value for training ' 'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--momentum',   default=0.9, type=float, metavar='M', help='SGD momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='SGD weight_decay')
    parser.add_argument('--lr-steps',   default=[16, 22], nargs='+', type=int, help='torch.optim.lr_scheduler.MultiStepLR decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma',   default=0.1, type=float, help='torch.optim.lr_scheduler.MultiStepLR decrease lr by a factor of lr-gamma')

    parser.add_argument('--batch_size', default=1, type=int, metavar='N', help='batch size when training.')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument("--pretrain",   type=bool, default=True, help="load COCO pretrain weights.")
    parser.add_argument("--amp",        default=True, help="Use torch.cuda.amp for mixed precision training 是否使用混合精度训练")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
