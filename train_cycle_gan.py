import torch
from torch import nn
from facenet_pytorch import InceptionResnetV1
from utk_dataset import get_data_loaders
from utk_models import Discriminator, Generator
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import os
import math
import random

random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
face_encoder = InceptionResnetV1(pretrained="vggface2").to(device).eval()

if not os.path.exists("images"):
    os.mkdir("images")

if not os.path.exists("models_improv"):
    os.mkdir("models_improv")


def add_noise(image_encodings):
    num_samples = image_encodings.shape[0]
    noise_vectors = torch.randn((num_samples, 128), device=device)
    augmented_noise = torch.cat([image_encodings, noise_vectors], dim=1).unsqueeze(2).unsqueeze(3)
    return augmented_noise


batch_size = 16
young_faces_dataloader = get_data_loaders(age_group="20-29", batch_size=batch_size)
old_faces_dataloader = get_data_loaders(age_group="50-69", batch_size=batch_size)
# Add sigmoid=True to discriminators for BCE Loss
discriminator_young = Discriminator().to(device)
discriminator_young.load_state_dict(torch.load("models_improv/d_young.pth"))
discriminator_old = Discriminator().to(device)
discriminator_old.load_state_dict(torch.load("models_improv/d_old.pth"))
generator_young_to_old = Generator(noise_size=640).to(device)
generator_young_to_old.load_state_dict(torch.load("models_improv/g_yto.pth"))
generator_old_to_young = Generator(noise_size=640).to(device)
generator_old_to_young.load_state_dict(torch.load("models_improv/g_oty.pth"))

discriminator_young_optimizer = torch.optim.Adam(discriminator_young.parameters(), lr=2e-4, betas=(0.5, 0.999))
discriminator_old_optimizer = torch.optim.Adam(discriminator_old.parameters(), lr=2e-4, betas=(0.5, 0.999))
generator_yto_optimizer = torch.optim.Adam(generator_young_to_old.parameters(), lr=2e-4, betas=(0.5, 0.999))
generator_oty_optimizer = torch.optim.Adam(generator_old_to_young.parameters(), lr=2e-4, betas=(0.5, 0.999))

adversarial_loss = nn.MSELoss()  # Change this to BCE Loss for normal
cycle_loss = nn.L1Loss()
identity_preservation_loss = nn.L1Loss()
num_epochs = 50
total_train_iters = num_epochs * math.ceil(5200 / batch_size)
iteration = 1
real_label = 0.9  # Change this to 1 to remove label smoothing
fake_label = 0.1  # Change this to 0 to remove label smoothing
for epoch in range(num_epochs):
    print(f"Epoch: {epoch}")
    for young_images, old_images in zip(young_faces_dataloader, old_faces_dataloader):
        young_images = young_images.to(device)
        old_images = old_images.to(device)
        young_image_encodings = add_noise(face_encoder(young_images))
        old_image_encodings = add_noise(face_encoder(old_images))

        # Optimizing generator young to old
        real_labels = torch.full(size=(len(young_images),), fill_value=1.0, device=device)
        generated_old_images = generator_young_to_old(young_image_encodings)
        p_old_generated = discriminator_old(generated_old_images)
        g_old_adv = adversarial_loss(p_old_generated, real_labels)
        old_identity_loss = identity_preservation_loss(face_encoder(young_images), face_encoder(generated_old_images))
        reconstructed_young_images = generator_old_to_young(add_noise(face_encoder(generated_old_images)))
        young_cycle_loss = cycle_loss(reconstructed_young_images, young_images)

        # Optimizing generator old to young
        real_labels = torch.full(size=(len(old_images),), fill_value=1.0, device=device)
        generated_young_images = generator_old_to_young(old_image_encodings)
        p_young_generated = discriminator_young(generated_young_images)
        g_young_adv = adversarial_loss(p_young_generated, real_labels)
        young_identity_loss = identity_preservation_loss(face_encoder(old_images), face_encoder(generated_young_images))
        reconstructed_old_images = generator_young_to_old(add_noise(face_encoder(generated_young_images)))
        old_cycle_loss = cycle_loss(reconstructed_old_images, old_images)

        # Creating loss functions and optimizing
        generator_yto_optimizer.zero_grad()
        generator_oty_optimizer.zero_grad()
        g_yto_loss = g_old_adv + 5 * old_identity_loss + 10 * old_cycle_loss
        g_oty_loss = g_young_adv + 5 * young_identity_loss + 10 * young_cycle_loss
        g_total_loss = g_oty_loss + g_yto_loss
        g_total_loss.backward(retain_graph=True)
        generator_yto_optimizer.step()
        generator_oty_optimizer.step()

        # Optimizing discriminator young
        discriminator_young_optimizer.zero_grad()
        real_labels = torch.full(size=(len(young_images),), fill_value=real_label, device=device)
        p_young_real = discriminator_young(young_images)
        adv_young_real = adversarial_loss(p_young_real, real_labels)
        fake_young_images = generator_old_to_young(old_image_encodings)
        fake_labels = torch.full(size=(len(young_images),), fill_value=fake_label, device=device)
        p_young_fake = discriminator_young(fake_young_images)
        adv_young_fake = adversarial_loss(p_young_fake, fake_labels)
        d_young_loss = 0.5 * (adv_young_real + adv_young_fake)
        d_young_loss.backward(retain_graph=True)
        discriminator_young_optimizer.step()

        # Optimizing discriminator old
        discriminator_old_optimizer.zero_grad()
        real_labels = torch.full(size=(len(old_images),), fill_value=real_label, device=device)
        p_old_real = discriminator_old(old_images)
        adv_old_real = adversarial_loss(p_old_real, real_labels)
        fake_old_images = generator_young_to_old(young_image_encodings)
        fake_labels = torch.full(size=(len(old_images),), fill_value=fake_label, device=device)
        p_old_fake = discriminator_old(fake_old_images)
        adv_old_fake = adversarial_loss(p_old_fake, fake_labels)
        d_old_loss = 0.5 * (adv_old_real + adv_old_fake)
        d_old_loss.backward()
        discriminator_old_optimizer.step()

        if iteration % 10 == 0:
            print('Iteration [{:4d}/{:4d}] | D_young_loss: {:6.4f} | '
                  'D_old_loss: {:6.4f} | G_oty_loss: {:6.4f} | G_yto_loss: {:6.4f}'.format(
                iteration, total_train_iters, d_young_loss.item(),
                d_old_loss.item(), g_oty_loss.item(), g_yto_loss.item()
            ))

        if iteration % 200 == 0:
            zanette_image = Image.open("zanette.png").convert("RGB")
            savvides_image = Image.open("savvides.png").convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((128, 128), Image.BICUBIC),
                transforms.CenterCrop((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            zanette_image = transform(zanette_image).unsqueeze(0).to(device)
            savvides_image = transform(savvides_image).unsqueeze(0).to(device)
            zanette_encoding = add_noise(face_encoder(zanette_image))
            savvides_encoding = add_noise(face_encoder(savvides_image))
            old_zanette = generator_young_to_old(zanette_encoding)
            old_zanette = (old_zanette[0] + 1) / 2
            young_savvides = generator_old_to_young(savvides_encoding)
            young_savvides = (young_savvides[0] + 1) / 2
            save_image(old_zanette, f"images/old_zanette_{iteration}.png")
            save_image(young_savvides, f"images/young_savvides_{iteration}.png")
            torch.save(discriminator_old.state_dict(), f"models_improv/d_old.pth")
            torch.save(discriminator_young.state_dict(), f"models_improv/d_young.pth")
            torch.save(generator_old_to_young.state_dict(), f"models_improv/g_oty.pth")
            torch.save(generator_young_to_old.state_dict(), f"models_improv/g_yto.pth")

        iteration += 1
