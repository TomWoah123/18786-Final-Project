import torch
from torch import nn
from facenet_pytorch import InceptionResnetV1
from create_utk_dataset import get_data_loaders, get_both_data_loaders
from utk_models import Discriminator, Generator
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import os
import math
import random

random.seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"------------ STARTING TRAINING: (using device: {device}) ------------")
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


batch_size = 8

# Add sigmoid=True to discriminators for BCE Loss
discriminator_young = Discriminator().to(device)
discriminator_old = Discriminator().to(device)
generator_young_to_old = Generator(noise_size=640).to(device)
generator_old_to_young = Generator(noise_size=640).to(device)

# Checking if there are saved models for each generator/discriminator
have_saved_model = {"models_improv/wcgan_d_young.pth": False, "models_improv/wcgan_d_old.pth": False, \
                    "models_improv/wcgan_g_yto.pth": False, "models_improv/wcgan_g_oty.pth": False}
# Loading in the correct saved model if it exists
for model_file, file_exists in have_saved_model.items():
    if os.path.exists(model_file):
        print(f"Found a saved model; relative path = {model_file}")
        have_saved_model[model_file] = True
        if model_file == "models_improv/wcgan_d_young.pth":
            discriminator_young.load_state_dict(torch.load(model_file))
        elif model_file == "models_improv/wcgan_d_old.pth":
            discriminator_old.load_state_dict(torch.load(model_file))
        elif model_file == "models_improv/wcgan_g_yto.pth":
            generator_young_to_old.load_state_dict(torch.load(model_file))
        elif model_file == "models_improv/wcgan_g_oty.pth":
            generator_old_to_young.load_state_dict(torch.load(model_file))

discriminator_young_optimizer = torch.optim.Adam(discriminator_young.parameters(), lr=2e-4, betas=(0.5, 0.999))
discriminator_old_optimizer = torch.optim.Adam(discriminator_old.parameters(), lr=2e-4, betas=(0.5, 0.999))
generator_yto_optimizer = torch.optim.Adam(generator_young_to_old.parameters(), lr=2e-4, betas=(0.5, 0.999))
generator_oty_optimizer = torch.optim.Adam(generator_old_to_young.parameters(), lr=2e-4, betas=(0.5, 0.999))

def wasserstein_loss(prob_fake_image=None, prob_real_image=None, type=None): 
    if type == "Generator":
        g_loss = -torch.mean(prob_fake_image) # Want to minimize the probability of the fake image is fake
        return g_loss
    elif type == "Discriminator":
        d_loss = -torch.mean(prob_real_image + prob_real_image)
        return d_loss



cycle_loss = nn.L1Loss()
identity_preservation_loss = nn.L1Loss()
num_epochs = 50
total_train_iters = num_epochs * math.ceil(5200 / batch_size)
iteration = 1
real_label = 0.9  # Change this to 1 to remove label smoothing
fake_label = 0.1  # Change this to 0 to remove label smoothing
for epoch in range(num_epochs):
    print(f"Epoch: {epoch}")
    # Reinitializing the dataloader so it doesn't get exhausted
    both_ages_dataloader = get_both_data_loaders(young_age_group="20-29", old_age_group="50-69", batch_size=batch_size)
    for young_images, old_images in both_ages_dataloader:
        young_images = young_images.to(device)
        old_images = old_images.to(device)
        young_image_encodings = add_noise(face_encoder(young_images))
        old_image_encodings = add_noise(face_encoder(old_images))

        # Optimizing generator young to old
        generated_old_images = generator_young_to_old(young_image_encodings) # image created from sampled noise
        p_old_generated = discriminator_old(generated_old_images) 
        g_old_wasserstein = wasserstein_loss(prob_fake_image=p_old_generated, prob_real_image=None, type="Generator")
        old_identity_loss = identity_preservation_loss(face_encoder(young_images), face_encoder(generated_old_images))
        reconstructed_young_images = generator_old_to_young(add_noise(face_encoder(generated_old_images)))
        young_cycle_loss = cycle_loss(reconstructed_young_images, young_images)

        # Optimizing generator old to young
        generated_young_images = generator_old_to_young(old_image_encodings)
        p_young_generated = discriminator_young(generated_young_images)
        g_young_wasserstein = wasserstein_loss(prob_fake_image=p_young_generated, prob_real_image=None, type="Generator")
        young_identity_loss = identity_preservation_loss(face_encoder(old_images), face_encoder(generated_young_images))
        reconstructed_old_images = generator_young_to_old(add_noise(face_encoder(generated_young_images)))
        old_cycle_loss = cycle_loss(reconstructed_old_images, old_images)

        # Creating loss functions and optimizing
        generator_yto_optimizer.zero_grad()
        generator_oty_optimizer.zero_grad()
        g_yto_loss = g_old_wasserstein + 5 * old_identity_loss + 10 * old_cycle_loss
        g_oty_loss = g_young_wasserstein + 5 * young_identity_loss + 10 * young_cycle_loss
        g_total_loss = g_oty_loss + g_yto_loss
        g_total_loss.backward(retain_graph=True)
        generator_yto_optimizer.step()
        generator_oty_optimizer.step()

        # Optimizing discriminator young
        discriminator_young_optimizer.zero_grad()
        p_young_real = discriminator_young(young_images)
        fake_young_images = generator_old_to_young(old_image_encodings)
        p_young_fake = discriminator_young(fake_young_images)
        d_young_loss_wasserstein = wasserstein_loss(prob_fake_image=fake_young_images, prob_real_image=p_young_real, type="Discriminator")
        d_young_loss_wasserstein.backward(retain_graph=True)
        discriminator_young_optimizer.step()

        # Optimizing discriminator old
        discriminator_old_optimizer.zero_grad()
        p_old_real = discriminator_old(old_images)
        fake_old_images = generator_young_to_old(young_image_encodings)
        p_old_fake = discriminator_old(fake_old_images)
        d_old_loss_wasserstein = wasserstein_loss(prob_fake_image=fake_old_images, prob_real_image=p_old_real, type="Discriminator")
        d_old_loss_wasserstein.backward()
        discriminator_old_optimizer.step()

        if iteration % 10 == 0:
            print('Iteration [{:4d}/{:4d}] | D_young_loss: {:6.4f} | '
                  'D_old_loss: {:6.4f} | G_oty_loss: {:6.4f} | G_yto_loss: {:6.4f}'.format(
                iteration, total_train_iters, d_young_loss_wasserstein.item(),
                d_old_loss_wasserstein.item(), g_oty_loss.item(), g_yto_loss.item()
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
            save_image(old_zanette, f"images/_wcgan_old_zanette_{iteration}.png")
            save_image(young_savvides, f"images/wcgan_young_savvides_{iteration}.png")
            torch.save(discriminator_old.state_dict(), f"models_improv/wcgan_d_old.pth")
            torch.save(discriminator_young.state_dict(), f"models_improv/wcgan_d_young.pth")
            torch.save(generator_old_to_young.state_dict(), f"models_improv/wcgan_g_oty.pth")
            torch.save(generator_young_to_old.state_dict(), f"models_improv/wcgan_g_yto.pth")

        iteration += 1
