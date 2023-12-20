# AI_in_II_task2
DALLE compression

Semantic compression algorithm realisation with DALLE2 model.

You can run it via script: 
main.py --config_path <path to model config> --data_path <path to dataset of images and texts> --gpu <gpu  num>,

or via main.ipynb notebook.

Config path: ./configs/dalle2.json

Average RGB-PSNR > 30.

You should download the weights and place them in the directory ./weights. Names of the weights files are registered in config path:
prior.pth, decoder_1.pth and decoder_2.pth
