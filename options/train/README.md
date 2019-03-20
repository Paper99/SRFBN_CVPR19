# Description of Train Options

Let us take [`train_SRFBN_example.json`](./train_SRFBN_example.json) as an example. 

**Note**: Before you run `python train.py -opt options/train/*.json`, please carefully check options: `"scale"`, `"dataroot_HR"`, `"dataroot_LR"`, `"networks"` and `"pretrained_path"` (if `"pretrain"` option is set to `"resume"` or `"finetune"`).

```c++
{
    "mode": "sr", // solver type (only "sr" is provided)
    "use_cl": true, // whether use multiple losses (required by our SRFBN)
    "gpu_ids": [0], // GPU ID to use

    "scale": 4, // super resolution scale (*Please carefully check it*)
    "is_train": true, // whether train the model
    "use_chop": true, // whether enable memory-efficient test
    "rgb_range": 255, // maximum value of images
    "self_ensemble": false, // whether use self-ensemble strategy for test
    "save_image": false, // whether saving visual results during training

    // dataset specifications including "train" dataset (for training) and "val" dataset (for validation) (*Please carefully check dateset mode/root*)
    "datasets": { 
        // train dataset
        "train": { 
            "mode": "LRHR", // dataset mode: "LRHR" | "LR", "LRHR" is required during training
            "dataroot_HR": "/mnt/data/paper99/DIV2K/Augment/DIV2K_train_HR_aug/x4", // HR dataset root
            "dataroot_LR": "/mnt/data/paper99/DIV2K/Augment/DIV2K_train_LR_aug/x4", // LR dataset root
            "data_type": "npy", // data type: "img" (image files) | "npy" (binary files), "npy" is recommended for training
            "n_workers": 4, // number of threads for data loading
            "batch_size": 16, // input batch size
            "LR_size": 32, // input (LR) patch size
            // data augmentation
            "use_flip": true, // whether use horizontal and vertical flips
            "use_rot": true, // whether rotate 90 degrees
            "noise": "." // noise type: "." (noise free) | "G" (add Gaussian noise) | "S" (add Poisson noise)
        },
        // validation dataset
        "val": {
            "mode": "LRHR",  // dataset mode: "LRHR" | "LR", "LRHR" is required during validating
            "dataroot_HR": "./results/HR/Set5/x4",
            "dataroot_LR": "./results/LR/LRBI/Set5/x4",
            "data_type": "img"
        }
    },

    // networks specifications
    "networks": {
        "which_model": "SRFBN", // network name
        "num_features": 32, // number of base feature maps
        "in_channels": 3, // number of input channels
        "out_channels": 3, // number of output channels
        "num_steps": 4, // number of time steps (T)
        "num_groups": 3 // number of projection groups (G)
    },

    // solver specifications
    "solver": {
        "type": "ADAM", // optimizer to use (only "ADAM" is provided)
        "learning_rate": 0.0001, // learning rate
        "weight_decay": 0, // weight decay
        "lr_scheme": "MultiStepLR", // learning rate scheduler (only "MultiStepLR" is provided)
        "lr_steps": [200, 400, 600, 800], // milestone for learning rate scheduler
        "lr_gamma": 0.5, // gamma for learning rate scheduler
        "loss_type": "l1", // loss type: "l1" | "l2"
        "manual_seed": 0, // manual seed for random & torch.random
        "num_epochs": 1000, // number of epochs to train
        
        // skipping batch that has large error for stable training
        "skip_threshold": 3,
        // split the batch into smaller chunks for less GPU memory consumption during training
        "split_batch": 1,
        // how many epochs to wait before saving a new checkpoint during training
        "save_ckp_step": 1,
        // how many epochs to wait before saving visual results during training (only works when option "save_image" is true)
        "save_vis_step": 4,
        // pre-train mode: null (from scratch) | "resume" (resume from specific checkpoint) | "finetune" (finetune a new model based on a specific model)
        "pretrain": null,
        // path to *.pth (required by "resume"/"finetune")
        "pretrained_path": "./experiments/SRFBN_in3f32_x4/epochs/last_ckp.pth",
        // weights for multiple losses (only works when option "use_cl" is true)
        "cl_weights": [1.0, 1.0, 1.0, 1.0]
    }
}
```
