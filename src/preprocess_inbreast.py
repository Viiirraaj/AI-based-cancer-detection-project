import os
import xml.etree.ElementTree as ET
import pydicom
import matplotlib.pyplot as plt

# Folders
XML_FOLDER = 'C:/Users/VIRAJ/OneDrive/Documents/AI-based-cancer-detection-project/data/raw/INbreast Release 1.0/AllXML'
DCM_FOLDER = 'C:/Users/VIRAJ/OneDrive/Documents/AI-based-cancer-detection-project/data/raw/INbreast Release 1.0/AllDICOMs'
OUT_FOLDER = 'C:/Users/VIRAJ/OneDrive/Documents/AI-based-cancer-detection-project/data/processed/Inbreast/images'

os.makedirs(os.path.join(OUT_FOLDER, "benign"), exist_ok=True)
os.makedirs(os.path.join(OUT_FOLDER, "malignant"), exist_ok=True)

def get_label_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    rois = root.findall('.//dict/array/dict')
    labels = []

    for roi in rois:
        roi_dict = {roi[i].text: roi[i+1].text for i in range(0, len(roi), 2)}
        roi_type = roi_dict.get("Type")
        if roi_type == "15":
            labels.append("benign")
        elif roi_type == "19":
            labels.append("malignant")

    if labels:
        return max(set(labels), key=labels.count)
    return None

def convert_dcm_to_png(dcm_path, output_path):
    ds = pydicom.dcmread(dcm_path)
    plt.imshow(ds.pixel_array, cmap='gray')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

# Loop through XML files
for xml_file in os.listdir(XML_FOLDER):
    if xml_file.endswith('.xml'):
        file_id = os.path.splitext(xml_file)[0]
        xml_path = os.path.join(XML_FOLDER, xml_file)
        dcm_path = os.path.join(DCM_FOLDER, file_id + '.dcm')

        if not os.path.exists(dcm_path):
            print(f"Missing DICOM: {file_id}")
            continue

        label = get_label_from_xml(xml_path)
        if label is None:
            print(f"Could not determine label for: {file_id}")
            continue

        output_path = os.path.join(OUT_FOLDER, label, file_id + '.png')
        convert_dcm_to_png(dcm_path, output_path)
        print(f"Saved: {output_path}")
"C:/Users/VIRAJ/OneDrive\Documents/AI-based-cancer-detection-project/data/processed/malignant"