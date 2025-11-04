import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from model import HybridModel # Assumes your HybridModel class is in model.py

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Cancer Detection Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
MODEL_PATH = "C:/Users/VIRAJ/OneDrive/Documents/AI-based-cancer-detection-project/output/Best_model/best_model.pt"
CLASS_NAMES = ['Benign', 'Malignant']
# IMPORTANT: You must verify this is the correct final convolutional layer from your model.
# This is a likely candidate from your CNN part before it goes to the ViT.
TARGET_LAYER_NAME = 'cnn.blocks.6.0.conv_pwl'


# --- Model Loading ---
@st.cache_resource
def load_model(model_path):
    """Loads the pre-trained HybridModel."""
    try:
        model = HybridModel(num_classes=2) # Initialize your model architecture
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        # The state dict is nested inside the 'model_state' key
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model(MODEL_PATH)


# --- Image Transformations ---
# These transforms MUST be the same as the ones used for validation/testing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# --- Grad-CAM Implementation ---
# NOTE: This is a standard Grad-CAM implementation.
# If your 'gradcam.py' is different, you can replace this section with your own code.
class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None

        # Find the target layer
        target_layer = None
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                target_layer = module
                break
        if target_layer is None:
            raise ValueError(f"Target layer '{target_layer_name}' not found in the model.")

        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, index=None):
        output = self.model(x)
        if index is None:
            index = torch.argmax(output)

        self.model.zero_grad()
        output[0][index].backward(retain_graph=True)

        # Pooling the gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()

        # Weighting the activations
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.cpu(), 0)
        heatmap /= torch.max(heatmap)
        return heatmap.numpy()

def generate_gradcam_overlay(image, heatmap):
    """Generates a Grad-CAM overlay on the original image."""
    heatmap = cv2.resize(heatmap, (image.width, image.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convert PIL image to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Create the superimposed image
    superimposed_img = heatmap * 0.4 + img_cv
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    # Convert back to RGB for display
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
    return superimposed_img

grad_cam = GradCAM(model, TARGET_LAYER_NAME)


# --- Main Application UI ---
st.title("🔬 AI-Powered Cancer Detection Assistant")
st.markdown("A first-phase verification tool for clinicians. Upload a medical image to get a prediction and a visual explanation.")

# --- Sidebar ---
with st.sidebar:
    st.header("Instructions")
    st.info(
        "1. **Upload Image:** Click the 'Browse files' button to upload a medical image (JPG, PNG).\n"
        "2. **Analyze:** The system will automatically process the image and display the results.\n"
        "3. **Review:** Check the prediction, confidence score, and the Grad-CAM heatmap which highlights areas the model focused on."
    )
    st.warning(
        "**Disclaimer:** This is an AI-based tool and not a substitute for professional medical advice. "
        "The results should be verified by a qualified radiologist."
    )
    st.subheader("Project Details")
    st.metric("Model Accuracy", "87.6%")
    st.metric("Model AUC", "0.95")


# --- Main Content ---
uploaded_file = st.file_uploader("Choose a medical image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, caption="Your uploaded image.", use_container_width=True)

    # Transform the image and make a prediction
    input_tensor = transform(image).unsqueeze(0)

    with st.spinner('Analyzing the image...'):
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_class_idx = torch.max(probabilities, 1)
            predicted_class_name = CLASS_NAMES[predicted_class_idx.item()]

        # Generate Grad-CAM
        heatmap = grad_cam(input_tensor, index=predicted_class_idx.item())
        gradcam_overlay = generate_gradcam_overlay(image, heatmap)

    with col2:
        st.subheader("Analysis Results")
        
        # Display color-coded result
        if predicted_class_name == 'Malignant':
            st.error(f"**Prediction: {predicted_class_name}**")
        else:
            st.success(f"**Prediction: {predicted_class_name}**")

        st.metric("Confidence Score", f"{confidence.item():.2%}")
        st.progress(confidence.item())

        st.image(gradcam_overlay, caption="Grad-CAM Heatmap Explanation", use_container_width=True)
        st.info(
            "The highlighted areas on the heatmap indicate the regions of the image that were most influential "
            "in the model's decision-making process."
        )

else:
    st.info("Please upload an image to begin the analysis.")
