# batch_size
batch_size = 1

# Training data
print('Loading Training data...', end=' ')
train_transforms = TF.Compose([
    TF.Resize(256),
    TF.RandomHorizontalFlip(),
    ])

train_imagefolder = PairImageFolder('D:/deep learning research paper/van gogh/vangogh2photo',
                                              train_transforms,
                                              mode='train')
train_loader = torch.utils.data.DataLoader(train_imagefolder,num_workers=0, batch_size=batch_size, shuffle=True)
print("Done!")
print("Training data size : {}".format(len(train_imagefolder)))
train_batch = next(iter(train_loader))
