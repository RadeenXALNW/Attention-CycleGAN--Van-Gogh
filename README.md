![image](https://user-images.githubusercontent.com/66905164/194865584-2ce822ce-2389-4397-abe3-b41347af21e8.png)

**AttentionGAN Training/Testing**
> Download a dataset using the previous script (e.g., horse2zebra).


| First Header  | Second Header |
| ------------- | ------------- |
|Hardware  | CPU : Intel i7-10875H 
|         |RAM : 16G| 
|          |GPU : NVIDIA RTX 2070 Super 8G|
| Epoch | 49  |
| Learning rate | 2e-4



*Test the Pretrained Model with pretrained file*




Test the Trained Model
python test.py --dataroot ./datasets/selfie2anime/ --name selfie2anime_attentiongan --model attention_gan --dataset_mode unaligned --norm instance --phase test --no_dropout --load_size 256 --crop_size 256 --batch_size 1 --gpu_ids 0 --num_test 5000 --epoch latest



` @article{tang2021attentiongan,
  title={AttentionGAN: Unpaired Image-to-Image Translation using Attention-Guided Generative Adversarial Networks},
  author={Tang, Hao and Liu, Hong and Xu, Dan and Torr, Philip HS and Sebe, Nicu},
  journal={IEEE Transactions on Neural Networks and Learning Systems (TNNLS)},
  year={2021} `
}

`@inproceedings{tang2019attention,
  title={Attention-Guided Generative Adversarial Networks for Unsupervised Image-to-Image Translation},
  author={Tang, Hao and Xu, Dan and Sebe, Nicu and Yan, Yan},
  booktitle={International Joint Conference on Neural Networks (IJCNN)},
  year={2019}
}`


***Some outputs are given below***

![IMG_20221010_163819-01-01 jpeg](https://user-images.githubusercontent.com/66905164/194865016-e8908521-2af5-412b-8ca5-8a9be0f6725f.jpg)
![IMG_20221010_164504-01 jpeg](https://user-images.githubusercontent.com/66905164/194865031-db5fb217-1d2f-453b-84d2-05a9f5956708.jpg)
![IMG_20221010_164853-01 jpeg](https://user-images.githubusercontent.com/66905164/194865041-7841a7f3-95d2-47d3-b9fb-6aacd66cfae9.jpg)

***Computer Generated Van Gogh***
![Screenshot 2022-10-10 000046](https://user-images.githubusercontent.com/66905164/194805425-e80a9bf9-ecdf-4d3f-8e8b-b5db249e7a8e.png)


Reference :
1. https://arxiv.org/abs/1903.12296
2. https://github.com/victor369basu/CycleGAN-with-Self-Attention
3. https://github.com/Ha0Tang/AttentionGAN

