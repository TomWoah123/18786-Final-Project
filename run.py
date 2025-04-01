import argparse
from models import Discriminator, Generator
import torch
from torchvision import transforms
import facenet_pytorch
from PIL import Image
import data_loader


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--conv_dim', type=int, default=64)
    parser.add_argument('--noise_size', type=int, default=100)

    # Training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # Data sources
    parser.add_argument('--data', type=str, default='cat/grumpifyBprocessed')
    parser.add_argument('--data_preprocess', type=str, default='advanced')
    parser.add_argument('--ext', type=str, default='*.png')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--checkpoint_dir', default='./checkpoints_vanilla')
    parser.add_argument('--sample_dir', type=str, default='vanilla')
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=200)
    parser.add_argument('--checkpoint_every', type=int, default=400)

    return parser


train_loader, test_loader = data_loader.get_data_loaders()
discriminator = Discriminator()
generator = Generator()
num_epochs = 30
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
adversarial_loss = torch.nn.BCELoss()
recycle_loss = torch.nn.L1Loss()
cycle_loss = torch.nn.L1Loss()
for epoch in range(num_epochs):
    for images, ages in train_loader:
        print(f"Epoch: {epoch}")
        dis_optimizer.zero_grad()
        real_labels = torch.full(size=(len(images),), fill_value=1, dtype=float)
        p_real = discriminator(images).double()
        d_real_loss = adversarial_loss(input=p_real, target=real_labels)
        new_target_ages = 1 - ages
        fake_labels = torch.full(size=(len(images),), fill_value=0, dtype=float)
        generated_images = generator(images, new_target_ages)
        p_fake = discriminator(generated_images).double()
        d_fake_loss = adversarial_loss(input=p_fake, target=fake_labels)
        discriminator_loss = d_real_loss + d_fake_loss
        discriminator_loss.backward()
        dis_optimizer.step()
        print(f"Optimized Discriminator. Loss: {discriminator_loss.item()}")

        gen_optimizer.zero_grad()
        fake_labels = torch.full(size=(len(images),), fill_value=1, dtype=float)
        generated_images = generator(images, new_target_ages)
        p_real = discriminator(generated_images).double()
        g_real_loss = adversarial_loss(input=p_real, target=fake_labels)
        reconstruction_loss = recycle_loss(images, generated_images)
        recycle_images = generator(generated_images, ages)
        gen_cycle_loss = cycle_loss(images, recycle_images)
        generator_loss = g_real_loss + reconstruction_loss + gen_cycle_loss
        generator_loss.backward()
        gen_optimizer.step()
        print(f"Optimized Generator. Loss: {generator_loss.item()}")
        break
    break



