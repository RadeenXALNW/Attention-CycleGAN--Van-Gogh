# netG_A2B=ResnetGenerator_our(in_channels=3,
#                        out_channels=3,
#                        ngf=64,
#                        n_blocks=9,).to(device)
# netG_B2A=ResnetGenerator_our(in_channels=3,
#                        out_channels=3,
#                        ngf=64,
#                        n_blocks=9,).to(device)


# netG_A2B=SelfAttnGenerator(in_ch=3).to(device)

# netG_B2A=SelfAttnGenerator(in_ch=3).to(device)
num_att = 5
netG_A2B = Generator(input_nc=3, 
                              output_nc=3, 
                              num_att=num_att, 
                              ngf=64, 
                              n_middle=9, 
                              norm='IN', 
                              activation='relu',
                              pretrained=False).to(device)
netG_B2A = Generator(input_nc=3, 
                              output_nc=3, 
                              num_att=num_att, 
                              ngf=64, 
                              n_middle=9, 
                              norm='IN', 
                              activation='relu',
                              pretrained=False).to(device)



netD_A=NLayerDiscriminator(in_channels=3,
                           ndf=64, 
                           n_layers=3, 
                           norm_layer=nn.BatchNorm2d).to(device)
netD_B=NLayerDiscriminator(in_channels=3,
                           ndf=64, 
                           n_layers=3, 
                           norm_layer=nn.BatchNorm2d).to(device)


# netD_A = Discriminator(input_nc=3, 
#                                 norm='IN', 
#                                 activation='lrelu', 
#                                 pretrained=False).to(device)
# netD_B = Discriminator(input_nc=3, 
#                                 norm='IN', 
#                                 activation='lrelu', 
#                                 pretrained=False).to(device)

num_params = sum(p.numel() for p in netG_A2B.parameters() if p.requires_grad) + sum(p.numel() for p in netD_A.parameters() if p.requires_grad)
print('Number of parameters: %d' % (num_params*2))

torch.backends.cudnn.benchmark = True
print(netG_A2B)
# print(netD_A)
