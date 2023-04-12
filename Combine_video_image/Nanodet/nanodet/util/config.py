from .yacs import CfgNode

cfg1 = CfgNode(new_allowed=True)
cfg1.save_dir = "./"
# common params for NETWORK
cfg1.model = CfgNode(new_allowed=True)
cfg1.model.arch = CfgNode(new_allowed=True)
cfg1.model.arch.backbone = CfgNode(new_allowed=True)
cfg1.model.arch.fpn = CfgNode(new_allowed=True)
cfg1.model.arch.head = CfgNode(new_allowed=True)

# DATASET related params
cfg1.data = CfgNode(new_allowed=True)
cfg1.data.train = CfgNode(new_allowed=True)
cfg1.data.val = CfgNode(new_allowed=True)
cfg1.device = CfgNode(new_allowed=True)
cfg1.device.precision = 32
# train
cfg1.schedule = CfgNode(new_allowed=True)

# logger
cfg1.log = CfgNode()
cfg1.log.interval = 50

# testing
cfg1.test = CfgNode()
# size of images for each device

cfg2 = CfgNode(new_allowed=True)
cfg2.save_dir = "./"
# common params for NETWORK
cfg2.model = CfgNode(new_allowed=True)
cfg2.model.arch = CfgNode(new_allowed=True)
cfg2.model.arch.backbone = CfgNode(new_allowed=True)
cfg2.model.arch.fpn = CfgNode(new_allowed=True)
cfg2.model.arch.head = CfgNode(new_allowed=True)

# DATASET related params
cfg2.data = CfgNode(new_allowed=True)
cfg2.data.train = CfgNode(new_allowed=True)
cfg2.data.val = CfgNode(new_allowed=True)
cfg2.device = CfgNode(new_allowed=True)
cfg2.device.precision = 32
# train
cfg2.schedule = CfgNode(new_allowed=True)

# logger
cfg2.log = CfgNode()
cfg2.log.interval = 50

# testing
cfg2.test = CfgNode()
# size of images for each device

def load_config(cfg, args_cfg):
    cfg.defrost()
    cfg.merge_from_file(args_cfg)
    cfg.freeze()


if __name__ == "__main__":
    import sys

    with open(sys.argv[1], "w") as f:
        print(cfg1, file=f)
