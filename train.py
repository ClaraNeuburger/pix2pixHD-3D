import os
import time
from datetime import datetime
import torch
import torchio as tio
from dataloader import Dataset
from dataloader_test_list import Dataset_testlist
from models.pix2pixHD_model import Pix2PixHDModel
from options.train_options import TrainOptions
from torch.utils.tensorboard import SummaryWriter
from util.util import print_log, format_train_log




def run_inference(model, patch_loader, aggregator1, aggregator2, step_visu, i):
    with torch.no_grad():
        for patches_batch in patch_loader:
            mri = patches_batch['mri'][tio.DATA].to(device).float()
            ct = patches_batch['ct'][tio.DATA].to(device)

            locations = patches_batch[tio.LOCATION]

            output_patches = model.inference(mri, mri)

            # Add the batch's predicted patches to the aggregator
            aggregator1.add_batch(output_patches, locations)
            aggregator2.add_batch(ct, locations)

    output_tensor = aggregator1.get_output_tensor()
    ct_tensor = aggregator2.get_output_tensor()

    output_tensor = output_tensor.squeeze()
    ct_tensor = ct_tensor.squeeze()
    middle_slice_index = output_tensor.shape[2] // 2
    middle_slice_sCT = output_tensor[:, :, middle_slice_index]

    middle_slice_CT = ct_tensor[:,:,middle_slice_index]

    # windowing and normalisation (just for visualisation of the TC in tensorboard)
    window_width = 400
    window_level = 40

    min_value = window_level - window_width // 2
    max_value = window_level + window_width // 2

    middle_slice_CT = torch.clamp(middle_slice_CT, min_value, max_value)
    normalized_slice_CT = (middle_slice_CT - min_value) / (max_value - min_value)

    combined_image = torch.cat((middle_slice_sCT, normalized_slice_CT), dim=1)

    tensorboard_writer.add_image(
                    f'test of disp {i}',
                    combined_image,
                    global_step=step_visu ,
                    dataformats='HW'
                )



opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
out_f = open(f"{os.path.join(opt.checkpoints_dir, opt.name)}/results.txt", 'w')

title = opt.name

if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.name, 'TensorBoard')):
    os.makedirs(os.path.join(opt.checkpoints_dir, opt.name, 'TensorBoard', title))

start_epoch, epoch_iter = 1, 0



if opt.use_test_list==True:
    dataset = Dataset_testlist(opt.dataroot,opt.input_nc, opt.plot_dataset,title,opt.testing_dataroot,opt.patch_size,opt.patch_overlap)
else:
    dataset = Dataset(opt.dataroot,opt.input_nc, opt.plot_dataset,title,opt.patch_size,opt.patch_overlap)
training_loader, validation_loader = dataset.get_loaders()


print('Dataset size:', len(training_loader), 'subjects')
dataset_size = len(training_loader)

print('#training images = %d' % dataset_size)

model = Pix2PixHDModel()
model.initialize(opt)

tensorboard_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, 'TensorBoard', title))

total_steps = 0
print_start_time = time.time()

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
step_visu = 0



for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):

    epoch_start_time = time.time()
    epoch_iter = 0


    for data in training_loader:
        torch.cuda.empty_cache()
        mri = data['mri'][tio.DATA].to(device)
        ct = data['ct'][tio.DATA].to(device)

        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        ############## Forward Pass ######################
        losses, generated = model(mri, ct)

        # sum per device losses
        losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
        loss_dict = dict(zip(model.loss_names, losses))

        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0) + loss_dict.get(
            'G_L1')

        loss_dict['loss_D']= loss_D
        loss_dict['loss_G'] = loss_G

        ############### Backward Pass ####################
        # update generator weights
        model.optimizer_G.zero_grad()
        loss_G.backward()
        model.optimizer_G.step()

        # update discriminator weights
        model.optimizer_D.zero_grad()
        loss_D.backward()
        model.optimizer_D.step()

        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == 0:
            t = (time.time() - print_start_time) / opt.batchSize
            errors = {k: v.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
            print_log(out_f, format_train_log(epoch, epoch_iter, errors, t))
            tensorboard_writer.add_scalars('Loss', errors, total_steps)
            print_start_time = time.time()

        ### display validation images in tensorboard
        if opt.use_test_list==False:
            if total_steps % opt.display_freq == 0:
                for i in range(0,7):
                    sub = dataset.subjects[i]
                    patch_loader, aggregator1, aggregator2 = dataset.get_val_subject(sub)

                    run_inference(model, patch_loader, aggregator1, aggregator2,step_visu,i+1)
                step_visu += 1

        if opt.use_test_list == True:
            if total_steps % opt.display_freq == 0:
                for i in range(0, 7):
                    sub = dataset.test_subjects[i]
                    patch_loader, aggregator1, aggregator2 = dataset.get_val_subject(sub)

                    run_inference(model, patch_loader, aggregator1, aggregator2, step_visu, i+1)
                step_visu += 1



        ### save latest model
    if epoch % opt.save_latest_freq == 0:
        print_log(out_f, 'saving the model at the end of epoch %d, iterations %d' %
                  (epoch, total_steps))
        model.save(epoch)
        model.save('latest')


    print_log(out_f, 'End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))


    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.update_learning_rate()

out_f.close()
tensorboard_writer.close()
