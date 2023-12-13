import torch
import torchvision.transforms as transforms
from PIL import Image
from imageio import get_reader
from HandGestureCNN import HandGestureCNN
import matplotlib.pyplot as plt

PATH = "INSERT PATH OF MODEL WEIGHTS"

# Load the trained model
model = HandGestureCNN()
model.load_state_dict(torch.load(PATH))
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize the image to 128x128
    transforms.Grayscale(),         # Convert the image to grayscale
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Open a camera for video capturing
cap = get_reader('<video0>',  'ffmpeg')

plt.ion()  # Turn on interactive mode for matplotlib

for frame in cap:
    # Convert the frame to PIL Image, resize, and convert to grayscale
    frame_pil = Image.fromarray(frame)
    frame_transformed = transform(frame_pil)
    frame_transformed_unsqueezed = frame_transformed.unsqueeze(0)  # Add batch dimension

    # Predict
    with torch.no_grad():
        prediction = model(frame_transformed_unsqueezed)
        predicted_class = torch.argmax(prediction, dim=1)
        print("Predicted Class:", predicted_class.item())

    # Display the frame
    plt.imshow(frame)
    plt.title(f"Predicted Class: {predicted_class.item()}")
    plt.pause(0.01)  # Pause to update the figure
    plt.clf()  # Clear the current figure

# Release the VideoCapture object
cap.close()
plt.ioff()  # Turn off interactive mode
