import os
import shutil

base_directory = r"C:/Users/Anthony/Desktop/multiclass-tensorflow-vision/dataset"
directories_to_merge = ["train", "valid", "test"]
merged_directory = os.path.join(base_directory, "merged")

# Ensure merged directory exists
if not os.path.exists(merged_directory):
    os.makedirs(merged_directory)

for directory in directories_to_merge:
    source_dir = os.path.join(base_directory, directory)

    # For each class sub-directory in the source directory
    for class_name in os.listdir(source_dir):
        source_class_dir = os.path.join(source_dir, class_name)
        merged_class_dir = os.path.join(merged_directory, class_name)

        # If this class sub-directory doesn't exist in the merged directory, create it
        if not os.path.exists(merged_class_dir):
            os.makedirs(merged_class_dir)

        # Copy each file from the source class sub-directory to the merged class sub-directory
        for filename in os.listdir(source_class_dir):
            source_file = os.path.join(source_class_dir, filename)

            # Prepend the directory name to the filename
            new_filename = directory + "_" + filename
            destination_file = os.path.join(merged_class_dir, new_filename)

            # If there's a name collision, append a number to the new filename
            if os.path.exists(destination_file):
                base, extension = os.path.splitext(new_filename)
                counter = 1
                new_destination_file = os.path.join(merged_class_dir, f"{base}_{counter}{extension}")

                while os.path.exists(new_destination_file):
                    counter += 1
                    new_destination_file = os.path.join(merged_class_dir, f"{base}_{counter}{extension}")

                shutil.copy2(source_file, new_destination_file)
            else:
                shutil.copy2(source_file, destination_file)

print("Merge completed!")
