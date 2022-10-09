# A : Edge, B : Color
num_att=5
netG_A2B = Generator(input_nc=3, 
                              output_nc=3, 
                              num_att=num_att, 
                              ngf=64, 
                              n_middle=9, 
                              norm='IN', 
                              activation='relu',
                              pretrained=False).to(device)
netG_B2A =Generator(input_nc=3, 
                              output_nc=3, 
                              num_att=num_att, 
                              ngf=64, 
                              n_middle=9, 
                              norm='IN', 
                              activation='relu',
                              pretrained=False).to(device)

torch.backends.cudnn.benchmark = True
def load(netG_A2B, netG_B2A):
    print('Loading...', end=' ')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('D:/deep learning research paper/van gogh/ckpt.pth')
    netG_A2B.load_state_dict(checkpoint['netG_A2B'], strict=True)
    netG_B2A.load_state_dict(checkpoint['netG_B2A'], strict=True)    
    print("Done!")
load(netG_A2B, netG_B2A)
