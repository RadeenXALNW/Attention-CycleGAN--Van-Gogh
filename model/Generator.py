class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, num_att, ngf=64, n_middle=9, norm='IN', activation='relu', pretrained=False):
        super(Generator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.num_att = num_att
        self.encoder = Encoder(self.input_nc, ngf=ngf, norm=norm, activation=activation, return_mid=False)
        self.middle = []
        for i in range(n_middle):
            self.middle.append(ConvBlock(ngf*4, ngf*4, 3, 1, 1, norm=norm, activation=activation, residual=True))         
        self.middle = nn.Sequential(*self.middle)
        self.decoder = Decoder(output_nc*(num_att-1), ngf=ngf, norm=norm, activation=activation) 
        self.att = AttentionNet(3, num_att, ngf=ngf, norm=norm, activation=activation)

    def forward(self, inputs):
        img = inputs
        encode = self.encoder(img)
        feature = self.middle(encode)
        content = self.decoder(feature)
        att = self.att(img)
        
        att_list = []
        content_list = []
        
        temp_att = att[:, 0:1, :, :]
        output = inputs * temp_att
        content_list.append(inputs)
        att_list.append(temp_att)

        for i in range(1, self.num_att):
            temp_att = att[:, i:i+1, :, :]
            temp_content = content[:, self.output_nc*(i-1):self.output_nc*i, :, :]
            content_list.append(temp_content)
            att_list.append(temp_att)
            output = output + temp_content * temp_att

        return output, content_list, att_list
        

# To initialize model weights
def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)
    else:
        pass
    
def toZeroThreshold(x, t=0.1):
    zeros = torch.cuda.FloatTensor(x.shape).fill_(0.0).cuda()
    return torch.where(x > t, x, zeros)
