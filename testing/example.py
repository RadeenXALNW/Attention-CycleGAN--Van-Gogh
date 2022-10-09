temp_batch_iter=iter(val_loader)
netG_A2B.eval()
netG_B2A.eval()
temp_batch = next(temp_batch_iter)

with torch.no_grad():
    
    imgA = temp_batch[0].to(device)
    imgB = temp_batch[1].to(device)

    fakeA, _, attB = netG_B2A(imgB)
    fakeB, _, attA = netG_A2B(imgA)

#     show_example([imgA, fakeB.detach()], (20, 10)) 
#     show_example(attA, (50, 10))
    show_example([imgB, fakeA.detach()], (20, 10)) 
#     show_example(attB, (50, 10))        
