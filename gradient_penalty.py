def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(real_data.size(0), 1, 1, 1).to(device)
    x_interpolate = ((1 - alpha) * real_data + alpha * fake_data).detach()

    x_interpolate.requires_grad = True
    d_inter_logit = netD(x_interpolate)
    grad = torch.autograd.grad(d_inter_logit, 
                               x_interpolate,
                               grad_outputs=torch.ones_like(d_inter_logit), 
                               create_graph=True)[0]
    norm = (grad.view(grad.size(0), -1)).norm(p=2, dim=1)

    d_gp = ((norm - 1) ** 2).mean()
    
    return d_gp
