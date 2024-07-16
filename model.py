import fitz  # PyMuPDF
import os
from PIL import Image
import io
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import pytesseract

# Path to the PDF file
pdf_path = 'progressive_identifier.pdf'  # Replace with the actual PDF path
output_dir = "extracted_engravings"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Open the PDF file
pdf_document = fitz.open(pdf_path)

# Iterate through the pages from 23 to 181 (0-indexed in PyMuPDF, so 22 to 180)
for page_num in range(22, 181):
    page = pdf_document[page_num]
    text = page.get_text("text")

    # Skip lenses that are discontinued
    if "This lens was discontinued" in text:
        continue

    # Extract images from the page
    images = page.get_images(full=True)
    if not images:
        continue

    # Extract the first image from the page (assuming it contains the lens engravings)
    xref = images[0][0]
    base_image = pdf_document.extract_image(xref)
    image_bytes = base_image["image"]
    image = Image.open(io.BytesIO(image_bytes))

    # Define regions for left and right engravings (this will vary based on the actual image layout)
    width, height = image.size
    left_region = (0, 0, width // 2, height)
    right_region = (width // 2, 0, width, height)

    # Crop the regions
    left_engraving = image.crop(left_region)
    right_engraving = image.crop(right_region)

    # Use OCR to extract text from the regions
    
    left_text = pytesseract.image_to_string(left_engraving, config='--psm 6')
    right_text = pytesseract.image_to_string(right_engraving, config='--psm 6')

    # Save the cropped regions as images
    left_engraving.save(os.path.join(output_dir, f"page_{page_num + 1}_left_engraving.png"))
    right_engraving.save(os.path.join(output_dir, f"page_{page_num + 1}_right_engraving.png"))

    # Print extracted text (or save it in a file/database as needed)
    print(f"Page {page_num + 1} - Left Engraving Text: {left_text.strip()}")
    print(f"Page {page_num + 1} - Right Engraving Text: {right_text.strip()}")

print(f"Engravings extracted and saved to directory: {output_dir}")

extracted_dir = "extracted_engravings"

# List of filenames to exclude
excluded_files = [
    "page_39_left_engraving.jpeg", "page_39_right_engraving.jpeg",
    "page_40_left_engraving.jpeg", "page_43_left_engraving.jpeg",
    "page_44_left_engraving.jpeg" , "page_46_left_engraving.jpeg",
    "page_47_left_engraving.jpeg",  "page_49_left_engraving.jpeg",
    "page_47_left_engraving.jpeg",  "page_49_left_engraving.jpeg",
    "page_51_left_engraving.jpeg",  "page_51_right_engraving.jpeg",
    "page_52_left_engraving.jpeg", "page_53_left_engraving.jpeg",
    "page_54_left_engraving.jpeg", "page_55_left_engraving.jpeg",
    "page_56_left_engraving.jpeg", "page_57_left_engraving.jpeg",
    "page_58_left_engraving.jpeg", "page_59_left_engraving.jpeg",
    "page_60_left_engraving.jpeg", "page_61_left_engraving.jpeg",
    "page_62_left_engraving.jpeg", "page_63_left_engraving.jpeg",
    "page_64_left_engraving.jpeg", "page_65_left_engraving.jpeg",
    "page_66_left_engraving.jpeg", "page_67_left_engraving.jpeg",
    "page_68_left_engraving.jpeg", "page_69_left_engraving.jpeg",
    "page_70_left_engraving.jpeg", "page_71_right_engraving.jpeg",
    "page_71_left_engraving.jpeg", "page_72_left_engraving.jpeg",
    "page_73_left_engraving.jpeg", "page_74_left_engraving.jpeg",
    "page_75_left_engraving.jpeg", "page_76_left_engraving.jpeg",
    "page_77_left_engraving.jpeg", "page_77_right_engraving.jpeg",
    "page_78_left_engraving.jpeg", "page_79_left_engraving.jpeg",
    "page_81_left_engraving.jpeg", "page_82_left_engraving.jpeg",
    "page_83_left_engraving.jpeg", "page_84_left_engraving.jpeg",
    "page_85_left_engraving.jpeg", "page_86_left_engraving.jpeg",
    "page_87_left_engraving.jpeg", "page_87_right_engraving.jpeg",
    "page_88_right_engraving.jpeg", "page_89_left_engraving.jpeg",
    "page_90_left_engraving.jpeg", "page_91_left_engraving.jpeg",
    "page_92_left_engraving.jpeg", "page_93_left_engraving.jpeg",
    "page_94_left_engraving.jpeg", "page_95_left_engraving.jpeg",
    "page_96_left_engraving.jpeg", "page_96_right_engraving.jpeg",
    "page_97_left_engraving.jpeg", "page_97_right_engraving.jpeg",
    "page_98_left_engraving.jpeg", "page_98_right_engraving.jpeg",
    "page_99_left_engraving.jpeg", "page_99_right_engraving.jpeg",
    "page_107_right_engraving.jpeg", "page_41_left_engraving.jpeg"
]  # Add other filenames as needed

# Iterate through the files in the directory and remove the unwanted ones
for filename in os.listdir(extracted_dir):
    if filename in excluded_files:
        file_path = os.path.join(extracted_dir, filename)
        os.remove(file_path)
        print(f"Removed file: {file_path}")
# get rid of unwanted .png and .jpef files
print("Unwanted images have been removed.")

augmented_dir = "augmented_symbols"
# Data augmentation
# Data augmentation
def augment_images(original_dir, augmented_dir):
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    for filename in os.listdir(original_dir):
        if filename.endswith(".png") or filename.endswith(".jpeg"):
            img_path = os.path.join(original_dir, filename)
            img = load_img(img_path)
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)

            img_dir = os.path.join(augmented_dir, os.path.splitext(filename)[0])
            os.makedirs(img_dir, exist_ok=True)

            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=img_dir, save_prefix='aug', save_format='png'):
                i += 1
                if i >= 20:  # Generate 20 augmented images per original image
                    break

    print(f"Augmented images saved to directory: {augmented_dir}")

augment_images(output_dir, augmented_dir)

# Prepare Data for OCR with CNNs
def load_images_and_labels(augmented_dir):
    images = []
    labels = []
    label_map = {}  # Mapping from filename to label
    label_counter = 0
    
    for root, dirs, files in os.walk(augmented_dir):
        for filename in files:
            if filename.endswith(".png") or filename.endswith(".jpeg"):
                img_path = os.path.join(root, filename)
                img = Image.open(img_path)
                img = img.resize((64, 64))  # Resize to a fixed size
                img = img.convert('L')  # Convert to grayscale
                img = np.array(img)
                images.append(img)
                
                label = os.path.basename(root)  # Use the directory name as the label
                if label not in label_map:
                    label_map[label] = label_counter
                    label_counter += 1
                labels.append(label_map[label])
    
    print(f"Loaded {len(images)} images with {len(labels)} labels.")
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

images, labels = load_images_and_labels(augmented_dir)

# Check if the dataset is loaded correctly
if len(images) == 0 or len(labels) == 0:
    print("No images or labels found. Please check the augmentation step.")
else:
    images = images.astype('float32') / 255.0
    images = np.expand_dims(images, axis=-1)  # Add channel dimension

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train)

    # Build CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(len(np.unique(labels)), activation='softmax')
    ])

    # Define the optimizer with a specific learning rate
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Define callbacks for early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

    # Train the model
    model.fit(datagen.flow(X_train, y_train, batch_size=32),
              epochs=200,
              validation_data=(X_test, y_test),
              callbacks=[early_stopping, reduce_lr])

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Set - Loss: {loss}, Accuracy: {accuracy}")

    model.save('symbol_classifier.keras')


    # some of the images are repeated, need to deal with
    # maybe alternatively, I can use the engravings from the other pdf file.