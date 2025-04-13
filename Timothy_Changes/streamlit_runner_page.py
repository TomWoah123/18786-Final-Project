import streamlit as st
from PIL import Image
import torch
from utk_models import Generator
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

# Title of the web-ui
st.title("AGE-IFY")
st.markdown("Age up 👴👵 or down 🧔‍♂️👩")

# Specifying the file/image that the user wants to upload
filepath = st.file_uploader("Upload Photo", type=["png", "jpg", "jpeg"])

# Load in both generators
generator_young_to_old = Generator(noise_size=640)
generator_old_to_young = Generator(noise_size=640)
generator_young_to_old.load_state_dict(torch.load("models_improv/ORIGINAL_g_yto.pth")) # The paths here are just the paths to each of the models (wherever you have them stored)
generator_old_to_young.load_state_dict(torch.load("models_improv/ORIGINAL_g_oty.pth")) # The paths here are just the paths to each of the models (wherever you have them stored)

# Setting up the transform needed for the image to be generated (same used for training)
transform = transforms.Compose([
            transforms.Resize((128, 128), Image.BICUBIC),
            transforms.CenterCrop((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

# The VGG encodeer
face_encoder = InceptionResnetV1(pretrained="vggface2").eval()

def add_noise(image_encodings):
    num_samples = image_encodings.shape[0]
    noise_vectors = torch.randn((num_samples, 128))
    augmented_noise = torch.cat([image_encodings, noise_vectors], dim=1).unsqueeze(2).unsqueeze(3)
    return augmented_noise

# When we need to turn the generated tensor into an image
tensor_to_pil = transforms.ToPILImage()

# Let's open the file
if filepath is not None:
    image = Image.open(filepath)
    st.image(image, caption="Uploaded Image")
    age = st.radio("Current Image Age:", ["20-29", "50-69"])

    transformed_image = transform(image).unsqueeze(0)
    endcoded_image = add_noise(face_encoder(transformed_image))
    
    # Picking an age
    if age == "20-29":
        st.write("We will age up to 50-69!")
        if st.button("Age UP"):
            with st.spinner("Generating older image..."):
                older_image = generator_young_to_old(endcoded_image)
                older_image = (older_image[0] + 1) / 2
                old_pil_image = tensor_to_pil(older_image)
                st.balloons()
                st.image(old_pil_image, caption="Older Image")

    elif age == "50-69":
        st.write("We will age down to 20-29!")
        if st.button("Age DOWN"):
            with st.spinner("Generating younger image..."):
                younger_image = generator_old_to_young(endcoded_image)
                younger_image = (younger_image[0] + 1) / 2
                young_pil_image = tensor_to_pil(younger_image)
                st.balloons()
                st.image(young_pil_image, caption="Younger Image")
                
# If the filepath isn't able to be opened 
else:
    st.write("Please upload an image")