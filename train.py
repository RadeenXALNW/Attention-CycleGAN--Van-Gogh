# epoch_num
current_epoch=0
epoch_num = 50
# learning rate
lr = 2e-4
# Loss functions
criterion_L1 = torch.nn.L1Loss() # L1 Loss
# Gamma
gamma = 10
# Lambda
lambda1 = 100
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
beta2 = 0.999
critic_iter = 1
# Setup Adam optimizers for both G and D
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=lr, betas=(beta1, beta2))
optimizer_D = torch.optim.Adam(itertools.chain(netD_A.parameters(), netD_B.parameters()), lr=lr, betas=(beta1, beta2))
def save(path, netG_A2B, netG_B2A, netD_A, netD_B, optimizer_G, optimizer_D, epoch):
    print('Saving...', end=' ')
    state = {
        'epoch': current_epoch,
        'netG_A2B': netG_A2B.state_dict(),
        'netG_B2A': netG_B2A.state_dict(),        
        'netD_A': netD_A.state_dict(),
        'netD_B': netD_B.state_dict(),        
        'optimizer_G': optimizer_G.state_dict(),
        'optimizer_D': optimizer_D.state_dict(),
        }
    torch.save(state,path)
    print("Done!")
# 3.2 Train the model
# Lists to keep track of progress
loss_list_D = []
loss_list_G = []
w_list = []
# Training Loop
print("Starting Training Loop...")
# For each epoch
start_epoch = current_epoch
last_epoch = epoch_num + current_epoch - 1
for epoch in range(current_epoch, epoch_num + current_epoch):
    current_epoch += 1
       
    start_time = time.time()
    total_time = 0
    running_w_list = []
    
    print('Epoch [{0}/{1}]'.format(epoch, last_epoch))
    for i, data in enumerate(train_loader, 0):                  
            
        # Set model input
        imgA = data[0].to(device)
        imgB = data[1].to(device)

        b_size = imgA.size(0)

        ###### Discriminator ######
        netD_A.zero_grad()
        netD_B.zero_grad()
        
        # Outputs
#         fakeA, _,attB = netG_B2A(imgB)
#         fakeB, _, attA = netG_A2B(imgA)
        fakeA ,_,attB= netG_B2A(imgB)
        fakeB,_,attA= netG_A2B(imgA)

        # GAN loss
        pred_fake = netD_A(fakeA.detach()) + netD_B(fakeB.detach())
        pred_real = netD_A(imgA) + netD_B(imgB)
        loss_D_GAN = pred_fake.mean() - pred_real.mean()
        
        gp = calc_gradient_penalty(netD_A, imgA, fakeA) + calc_gradient_penalty(netD_B, imgB, fakeB)
        
        # Total loss
        loss_D = loss_D_GAN + gamma*gp

        # Update
        loss_D.backward()
        optimizer_D.step()
        running_w_list.append(-loss_D_GAN.item())
        
        ###################################
        
        if i % critic_iter == 0:
            ###### Generators ######
            netG_A2B.zero_grad()
            netG_B2A.zero_grad()

            # Outputs
            #fakeA, _, attB = netG_B2A(imgB)
            #fakeB, _, attA = netG_A2B(imgA)
            idtA, _, _ = netG_B2A(imgA)
            idtB, _, _ = netG_A2B(imgB)
            recA, _, _ = netG_B2A(fakeB)
            recB, _, _ = netG_A2B(fakeA)

            # GAN loss
            pred_fake = netD_A(fakeA) + netD_B(fakeB)
            loss_G_GAN = -pred_fake.mean()

            # Reconstruction loss
            loss_rec = criterion_L1(imgA, recA) + criterion_L1(imgB, recB)
            
            # Identity Loss
            loss_idt = criterion_L1(imgA, idtA) + criterion_L1(imgB, idtB)
            
            # Total loss
            loss_G = loss_G_GAN + 100 * loss_rec + loss_idt

            # Update
            loss_G.backward()
            optimizer_G.step()

            ###################################

        # Time Info.
        end_time = time.time()
        taken_time = end_time - start_time
        total_time += taken_time
        average_time = total_time / (i+1)

        # Wasserstein Distance
        w_avg = sum(running_w_list) / len(running_w_list)

        # Output training stats
        print('\r[%d/%d] Loss D_GAN: %.2f (%.2f) / GP: %.2f / Loss_G_GAN: %.2f / Loss_Rec: %.2f / Loss_Idt: %.2f / Time : %.2f (%.2f)'
              % (i+1, len(train_loader), loss_D_GAN.item(), w_avg,
                gp.item(), loss_G_GAN.item(), loss_rec.item(), loss_idt.item(), taken_time, average_time), end='     ')
        start_time = end_time
            
        if i % 1000 == 0:
            with torch.no_grad():
                show_example([imgA, fakeB.detach(), recA.detach()], (30, 10)) 
                show_example(attA, (50, 10))
                show_example([imgB, fakeA.detach(), recB.detach()], (30, 10)) 
                show_example(attB, (50, 10))            
                print()
                save('D:/deep learning research paper/van gogh/ckpt.pth', netG_A2B, netG_B2A, netD_A, netD_B, optimizer_G, optimizer_D, current_epoch)
    
                         
    # Record loss
    loss_list_D.append(loss_D.cpu().item())
    loss_list_G.append(loss_G.cpu().item())
    w_list.append(w_avg)
        
    print()

print('Done')
