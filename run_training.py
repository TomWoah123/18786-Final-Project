from fgnet_dataset import get_data_loaders
from unet_model import Generator, Discriminator
import torch
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from perceptual_loss import PerceptualLoss
import torchmetrics.image as tm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader = get_data_loaders()
discriminator = Discriminator()
generator = Generator()
discriminator.to(device)
generator.to(device)
num_epochs = 30
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
gen_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(gen_optimizer, gamma=0.9)
dis_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(dis_optimizer, gamma=0.9)
adversarial_loss = torch.nn.BCELoss()
perceptual_loss = tm.lpip._LPIPS(pretrained=True, net="vgg").to(device)
l1_loss = torch.nn.L1Loss()
lambda_adv = 0.05
lambda_per = lambda_l1 = 1
sample_every = 3
for epoch in range(num_epochs):
    print(f"Epoch: {epoch}")
    for input_images, target_images in train_loader:
        input_images = input_images.to(device)
        target_images = target_images.to(device)
        dis_optimizer.zero_grad()
        real_labels = torch.full(size=(len(input_images),), fill_value=1, dtype=float).to(device)
        p_real = discriminator(input_images[:, :4, :, :]).double()
        d_real_loss = adversarial_loss(input=p_real, target=real_labels)

        fake_labels = torch.full(size=(len(input_images),), fill_value=0, dtype=float).to(device)
        generated_images = generator(input_images)
        target_age_maps = input_images[:, 4:, :, :]
        generated_images_with_target_age = torch.cat([generated_images, target_age_maps], dim=1).to(device)
        p_fake = discriminator(generated_images_with_target_age).double()
        d_fake_loss = adversarial_loss(input=p_fake, target=fake_labels)
        discriminator_loss = d_real_loss + d_fake_loss
        discriminator_loss.backward()
        dis_optimizer.step()
        d_loss = discriminator_loss.item()
        print(f"Optimized Discriminator. Loss: {d_loss}")

        gen_optimizer.zero_grad()
        real_labels = torch.full(size=(len(input_images),), fill_value=1, dtype=float).to(device)
        generated_images = generator(input_images)
        target_age_maps = input_images[:, 4:, :, :]
        generated_images_with_target_age = torch.cat([generated_images, target_age_maps], dim=1).to(device)
        p_real = discriminator(generated_images_with_target_age).double()
        g_real_loss = adversarial_loss(input=p_real, target=real_labels)
        reconstruction_loss = l1_loss(input=generated_images, target=target_images)
        perception_loss = perceptual_loss(generated_image=generated_images, target_image=target_images)
        generator_loss = lambda_adv * g_real_loss + lambda_l1 * reconstruction_loss + lambda_per * perception_loss
        generator_loss.backward()
        gen_optimizer.step()
        g_loss = generator_loss.item()
        print(f"Optimized Generator. Loss: {g_loss}")
    dis_lr_scheduler.step()
    gen_lr_scheduler.step()

    with torch.no_grad():
        for input_images, target_images in test_loader:
            input_images = input_images.to(device)
            target_images = target_images.to(device)
            real_labels = torch.full(size=(len(input_images),), fill_value=1, dtype=float).to(device)
            p_real = discriminator(input_images[:, :4, :, :]).double()
            d_real_loss = adversarial_loss(input=p_real, target=real_labels)

            fake_labels = torch.full(size=(len(input_images),), fill_value=0, dtype=float).to(device)
            generated_images = generator(input_images)
            target_age_maps = input_images[:, 4:, :, :]
            generated_images_with_target_age = torch.cat([generated_images, target_age_maps], dim=1).to(device)
            p_fake = discriminator(generated_images_with_target_age).double()
            d_fake_loss = adversarial_loss(input=p_fake, target=fake_labels)
            discriminator_loss = d_real_loss + d_fake_loss
            d_loss = discriminator_loss.item()
            print(f"Evaluating Discriminator Loss on Test: {d_loss}")

            fake_labels = torch.full(size=(len(input_images),), fill_value=1, dtype=float).to(device)
            generated_images = generator(input_images)
            target_age_maps = input_images[:, 4:, :, :]
            generated_images_with_target_age = torch.cat([generated_images, target_age_maps], dim=1).to(device)
            p_real = discriminator(generated_images_with_target_age).double()
            g_real_loss = adversarial_loss(input=p_real, target=fake_labels)
            reconstruction_loss = l1_loss(input=generated_images, target=target_images)
            perception_loss = perceptual_loss(generated_image=generated_images, target_image=target_images)
            generator_loss = lambda_adv * g_real_loss + lambda_l1 * reconstruction_loss + lambda_per * perception_loss
            g_loss = generator_loss.item()
            print(f"Evaluating Generator Loss on Test: {g_loss}")

    if epoch % sample_every == 0:
        image = Image.open("data/andrea_zanette.png").convert("L").convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((512, 512), Image.BICUBIC),
            transforms.ToTensor()
        ])
        image = transform(image).to(device)
        input_age_map = torch.full(size=(1, 512, 512), fill_value=30/69).to(device)
        target_age_map = torch.full(size=(1, 512, 512), fill_value=60/69).to(device)
        input_tensor = torch.cat([image, input_age_map, target_age_map], dim=0)
        input_tensor = input_tensor.unsqueeze(0)
        result = generator(input_tensor)
        result = result[0]
        save_image(result, f"reaged_zanette_{epoch}.png")







