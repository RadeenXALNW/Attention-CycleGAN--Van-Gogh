![image](https://user-images.githubusercontent.com/66905164/194865584-2ce822ce-2389-4397-abe3-b41347af21e8.png)


**AttentionGAN Training/Testing**
Download a dataset using the previous script (e.g., horse2zebra).

sh ./scripts/train.sh
To see more intermediate results, check out ./checkpoints/horse2zebra_attentiongan/web/index.html.
How to continue train? Append --continue_train --epoch_count xxx on the command line.

The test results will be saved to a html file here: ./results/horse2zebra_attentiongan/latest_test/index.html.
Generating Images Using Pretrained Model
You need download a pretrained model (e.g., horse2zebra) with the following script:
sh ./scripts/download_attentiongan_model.sh horse2zebra
The pretrained model is saved at ./checkpoints/{name}_pretrained/latest_net_G.pth.
Then generate the result using
python test.py --dataroot ./datasets/horse2zebra --name horse2zebra_pretrained --model attention_gan --dataset_mode unaligned --norm instance --phase test --no_dropout --load_size 256 --crop_size 256 --batch_size 1 --gpu_ids 0 --num_test 5000 --epoch latest --saveDisk
The results will be saved at ./results/. Use --results_dir {directory_path_to_save_result} to specify the results directory. Note that if you want to save the intermediate results and have enough disk space, remove --saveDisk on the command line.

For your own experiments, you might want to specify --netG, --norm, --no_dropout to match the generator architecture of the trained model.
Image Translation with Geometric Changes Between Source and Target Domains
For instance, if you want to run experiments of Selfie to Anime Translation. Usage: replace attention_gan_model.py and networks with the ones in the AttentionGAN-geo folder.

Test the Pretrained Model
Download data and pretrained model according above instructions.

python test.py --dataroot ./datasets/selfie2anime/ --name selfie2anime_pretrained --model attention_gan --dataset_mode unaligned --norm instance --phase test --no_dropout --load_size 256 --crop_size 256 --batch_size 1 --gpu_ids 0 --num_test 5000 --epoch latest

Train a New Model
python train.py --dataroot ./datasets/selfie2anime/ --name selfie2anime_attentiongan --model attention_gan --dataset_mode unaligned --pool_size 50 --no_dropout --norm instance --lambda_A 10 --lambda_B 10 --lambda_identity 0.5 --load_size 286 --crop_size 256 --batch_size 4 --niter 100 --niter_decay 100 --gpu_ids 0 --display_id 0 --display_freq 100 --print_freq 100

Test the Trained Model
python test.py --dataroot ./datasets/selfie2anime/ --name selfie2anime_attentiongan --model attention_gan --dataset_mode unaligned --norm instance --phase test --no_dropout --load_size 256 --crop_size 256 --batch_size 1 --gpu_ids 0 --num_test 5000 --epoch latest

Evaluation Code
FID: Official Implementation
KID or Here: Suggested by UGATIT. Install Steps: conda create -n python36 pyhton=3.6 anaconda and pip install --ignore-installed --upgrade tensorflow==1.13.1. If you encounter the issue AttributeError: module 'scipy.misc' has no attribute 'imread', please do pip install scipy==1.1.0.
Citation
If you use this code for your research, please cite our papers.

`rgb(R,G,B)` @article{tang2021attentiongan,
  title={AttentionGAN: Unpaired Image-to-Image Translation using Attention-Guided Generative Adversarial Networks},
  author={Tang, Hao and Liu, Hong and Xu, Dan and Torr, Philip HS and Sebe, Nicu},
  journal={IEEE Transactions on Neural Networks and Learning Systems (TNNLS)},
  year={2021} 
}

@inproceedings{tang2019attention,
  title={Attention-Guided Generative Adversarial Networks for Unsupervised Image-to-Image Translation},
  author={Tang, Hao and Xu, Dan and Sebe, Nicu and Yan, Yan},
  booktitle={International Joint Conference on Neural Networks (IJCNN)},
  year={2019}
}




![IMG_20221010_163819-01-01 jpeg](https://user-images.githubusercontent.com/66905164/194865016-e8908521-2af5-412b-8ca5-8a9be0f6725f.jpg)
![IMG_20221010_164504-01 jpeg](https://user-images.githubusercontent.com/66905164/194865031-db5fb217-1d2f-453b-84d2-05a9f5956708.jpg)
![IMG_20221010_164853-01 jpeg](https://user-images.githubusercontent.com/66905164/194865041-7841a7f3-95d2-47d3-b9fb-6aacd66cfae9.jpg)

![Screenshot 2022-10-10 000046](https://user-images.githubusercontent.com/66905164/194805425-e80a9bf9-ecdf-4d3f-8e8b-b5db249e7a8e.png)


Reference :
1. https://arxiv.org/abs/1903.12296

