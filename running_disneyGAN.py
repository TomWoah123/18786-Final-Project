import argparse
import torch
import os
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import save_image
from fgnet_dataset import get_data_loaders
from models import DiscriminatorPatchGAN, DisneyGenerator

#######################
# Establishing device
#######################

# Set the random seed manually for reproducibility. (the mps seed is set using te torch.manual_seed)
SEED = 20
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    device = torch.device("cuda")

# creating the mps device
if torch.backends.mps.is_available():
    device = torch.device("mps")



#######################
# "Utility functions"
#######################

def to_var(x, torch_device = None):
    """Converts numpy to variable."""
    x = x.to(torch_device)
    return x

def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        x = x.cpu()
    return x.detach().numpy()


def create_dir(directory):
    """Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def checkpoint(iteration, G, D, opts):
    """Save the parameters of the generator G and discriminator D."""
    G_path = os.path.join(opts.checkpoint_dir, 'G_iter%d.pkl' % iteration)
    D_path = os.path.join(opts.checkpoint_dir, 'D_iter%d.pkl' % iteration)
    torch.save(G.state_dict(), G_path)
    torch.save(D.state_dict(), D_path)

def create_image_canvas(generated_image_path, real_image_path, iteration, opts):
    generated_image = Image.open(generated_image_path)
    real_image = Image.open(real_image_path)

    generated_image_age = generated_image_path.split("_")[1].split(".")[0]
    real_image_age = real_image_path.split("_")[1].split(".")[0]
    
    resized_generated = generated_image.resize((512, 800))
    resized_real = real_image.resize((512, 800))

    total_image_width = resized_generated.width + resized_real.width
    canvas_height = resized_real.width + 50
    canvas = Image.new('RGB', (total_image_width, canvas_height), "white")

    # canvas.paste(resized_generated, (0, 0))
    # canvas.paste(resized_real, (resized_generated.width, 0))
    canvas.paste(resized_real, (0, 0))
    canvas.paste(resized_generated, (resized_real.width, 0))

    # Draw the text on the canvas
    draw_age = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw_age.text((50, resized_real.height + 10), f"Age: {real_image_age}", fill="black", font=font)
    draw_age.text((resized_real.width + 50, resized_generated.height + 10), f"Age: {generated_image_age}", fill="black", font=font)
    

    if not os.path.exists(opts.output_image_dir):
        os.mkdir(opts.output_image_dir)

    # Saving the image
    canvas.save(f"{opts.output_image_dir}/glen_{real_image_age}_to_{generated_image_age}_iteration{iteration}.jpg")
    print(f"Aging face saved at {opts.output_image_dir}/glen_{real_image_age}_to_{generated_image_age}_iteration{iteration}.jpg")

    # # Deleting the intermediate file
    # try:
    #     os.remove(generated_image_path)
    #     print(f"Intermediate file '{generated_image_path}' deleted successfully.")
    # except FileNotFoundError:
    #     print(f"Error: File '{generated_image_path}' not found.")
    # except Exception as e:
    #     print(f"An error occurred: {e}")

def save_sample_image(G, iteration, opts):
    image = Image.open(f"{opts.sample_image_dir}/glen-powell_30.jpg").convert("L").convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((512, 512), Image.BICUBIC),
        transforms.ToTensor()
    ])
    target_age = 60
    image = transform(image)
    input_age_map = torch.full(size=(1, 512, 512), fill_value=30/69)
    target_age_map = torch.full(size=(1, 512, 512), fill_value=target_age/69)
    input_tensor = torch.cat([image, input_age_map, target_age_map], dim=0)
    input_tensor = input_tensor.unsqueeze(0)
    if opts.use_gpu:
        input_tensor = to_var(input_tensor, device)

    result = G(input_tensor)
    result = result[0]
    if opts.use_gpu:
        result = result.cpu()

    print(f"creating the image canvas (resulting image is of type: {type(result)})")
    generated_im_path = f"./reaged-glenn-iteration{iteration}_{target_age}.png"
    # Save intermediate file (will be deleted later)
    save_image(result, generated_im_path)
    
    real_im_path = f"{opts.sample_image_dir}/glen-powell_30.jpg"
    # Create the image canvas (image with the original image and the generated image)
    create_image_canvas(generated_im_path, real_im_path, iteration, opts)





#######################
# Parser
#######################

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # # Model hyper-parameters
    # parser.add_argument('--image_size', type=int, default=64)
    # parser.add_argument('--conv_dim', type=int, default=64)
    # parser.add_argument('--noise_size', type=int, default=100)

    # # Training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    # parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # # Data sources
    # parser.add_argument('--data', type=str, default='cat/grumpifyBprocessed')
    # parser.add_argument('--data_preprocess', type=str, default='advanced')
    # parser.add_argument('--ext', type=str, default='*.png')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--checkpoint_dir', default='./aging_model_checkpoint')
    parser.add_argument('--sample_image_dir', type=str, default='./TestImages')
    parser.add_argument('--output_image_dir', type=str, default='./magically_aged_images')
    parser.add_argument('--log_step', type=int, default=1) # I want to get a log after every epoch
    # parser.add_argument('--sample_every', type=int, default=1) # saving an image every 5 samples
    parser.add_argument('--sample_every', type=int, default=5) # saving an image every 5 samples
    parser.add_argument('--checkpoint_every', type=int, default=5) # save the model params every 5 epochs
    parser.add_argument('--use_gpu', type=bool, default=False) # whether you want to use cuda or mps

    return parser

def train_discriminator(D, G, d_optimizer, input_image, target_image, criterion):
    # TRAIN THE DISCRIMINATOR

    # 1. Compute the discriminator loss on the target images (real images)
    D_real_logits = D(target_image, input_image)  
    real_label = torch.ones_like(D_real_logits)  # filled with ones for real
    D_real_loss = criterion(D_real_logits, real_label)  
    
    # 2. Generate fake images from the generator
    fake_images = G(input_image)
    D_fake_logits = D(fake_images.detach(), input_image)  
    fake_label = torch.zeros_like(D_fake_logits)  # filled with 0s for fake
    
    D_fake_loss = criterion(D_fake_logits, fake_label)  
    
    # 3. Find total discriminator loss
    D_total_loss = D_real_loss + D_fake_loss

    # Update the discriminator D
    d_optimizer.zero_grad()
    D_total_loss.backward()
    d_optimizer.step()

    return D_real_loss, D_fake_loss, D_total_loss

def train_generator(D, G, g_optimizer, input_image, criterion):
    # TRAIN THE GENERATOR
    
    # 1. Generate fake images from the generators
    fake_images = G(input_image)
    D_fake_logits = D(fake_images, input_image)  
    

    real_label = torch.ones_like(D_fake_logits)  # filled with ones to make the fake images seem real
    # 2. Compute the generator loss
    G_loss = criterion(D_fake_logits, real_label)  

    # Update the generator G
    g_optimizer.zero_grad()
    G_loss.backward()
    g_optimizer.step()

    return G_loss



def training_loop(train_dataloader, opts):
    """Runs the training loop.
        * Saves checkpoints every opts.checkpoint_every iterations
        * Saves generated samples every opts.sample_every iterations
    """
    # Create generator and discriminator
    G = DisneyGenerator()
    D = DiscriminatorPatchGAN()

    if opts.use_gpu:
        G = to_var(G, device)
        D = to_var(D, device)

    # Create optimizers for the generators and discriminators
    g_optimizer = torch.optim.Adam(G.parameters(), opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = torch.optim.Adam(D.parameters(), opts.lr, [opts.beta1, opts.beta2])


    # Establishing the loss function to be used
    loss_func = torch.nn.BCEWithLogitsLoss()

    iteration = 1

    total_train_iters = opts.num_epochs * len(train_dataloader)

    for i in range(opts.num_epochs):
        print(f"EPOCH: {i}:\n")

        discriminator_epoch_loss = []
        generator_epoch_loss = []

        for input_image, target_image in train_dataloader:
            if opts.use_gpu:
                input_image = to_var(input_image, device)
                target_image = to_var(target_image, device)


            # TRAIN THE DISCRIMINATOR
            D_real_loss, D_fake_loss, D_total_loss = train_discriminator(D, G, d_optimizer, input_image, target_image, loss_func)

            # TRAIN THE GENERATOR
            G_loss = train_generator(D, G, g_optimizer, input_image, loss_func)

            # Print the log info
            if iteration % opts.log_step == 0:
                print(
                    'Iteration [{:4d}/{:4d}] | D_real_loss: {:6.4f} | '
                    'D_fake_loss: {:6.4f} | G_loss: {:6.4f}'.format(
                        iteration, total_train_iters, D_real_loss.item(),
                        D_fake_loss.item(), G_loss.item()
                    )
                )

            # Save the generated samples/image
            if iteration % opts.sample_every == 0:
                save_sample_image(G, iteration, opts)
    

            # Save the model parameters
            if iteration % opts.checkpoint_every == 0:
                checkpoint(iteration, G, D, opts)

            iteration += 1


        # for batch in train_dataloader:

        #     real_images = batch
        #     if opts.gpu_type == "cuda":
        #         real_images = utils.to_var(real_images, opts)
        #     elif opts.gpu_type == "mps":
        #         real_images = utils.to_var(real_images, opts, mps_device)

        #     # TRAIN THE DISCRIMINATOR
        #     D_real_loss, D_fake_loss, D_total_loss = train_discriminator(D, G, d_optimizer, batch, real_images, opts)

        #     # TRAIN THE GENERATOR
        #     G_loss = train_generator(D, G, g_optimizer, opts)

        #     # append the losses
        #     if isnan(D_total_loss):
        #         discriminator_epoch_loss.append(0)
        #     else:
        #         discriminator_epoch_loss.append(utils.to_data(D_total_loss))
            
        #     if isnan(G_loss):
        #         generator_epoch_loss.append(0)
        #     else:
        #         generator_epoch_loss.append(utils.to_data(G_loss))


        #     # Print the log info
        #     if iteration % opts.log_step == 0:
        #         print(
        #             'Iteration [{:4d}/{:4d}] | D_real_loss: {:6.4f} | '
        #             'D_fake_loss: {:6.4f} | G_loss: {:6.4f}'.format(
        #                 iteration, total_train_iters, D_real_loss.item(),
        #                 D_fake_loss.item(), G_loss.item()
        #             )
        #         )
        #         logger.add_scalar('D/fake', D_fake_loss, iteration)
        #         logger.add_scalar('D/real', D_real_loss, iteration)
        #         logger.add_scalar('D/total', D_total_loss, iteration)
        #         logger.add_scalar('G/total', G_loss, iteration)

        #     # Save the generated samples
        #     if iteration % opts.sample_every == 0:
        #         save_samples(G, fixed_noise, iteration, opts)
        #         save_images(real_images, iteration, opts, 'real')
    

        #     # Save the model parameters
        #     if iteration % opts.checkpoint_every == 0:
        #         checkpoint(iteration, G, D, opts)

        #     iteration += 1

        # discriminator_losses.append(np.mean(np.array(discriminator_epoch_loss)))
        # generator_losses.append(np.mean(np.array(generator_epoch_loss)))
        



def main(opts):
    """Loads the data and starts the training loop."""

    # Create a dataloader for the training images
    train_loader, test_loader  = get_data_loaders(opts.batch_size)

    # Create checkpoint and sample directories
    create_dir(opts.checkpoint_dir)

    training_loop(train_loader, opts)

    # TODO: Work on testing



if __name__ == '__main__':
    parser = create_parser()
    opts = parser.parse_args()
    print(opts)
    main(opts)
