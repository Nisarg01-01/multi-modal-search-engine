# scripts/analyze_coco.py
import fiftyone as fo
import os
import json


def analyze_coco_dataset():
    """
    Manually builds a dataset from the COCO-2017 validation split and its
    caption annotations to get a definitive count.
    """
    print("Manually loading COCO-2017 validation split from downloaded files...")

    # --- Define Paths ---
    dataset_dir = os.path.join(
        os.path.expanduser("~"), "fiftyone", "coco-2017", "validation"
    )
    data_path = os.path.join(dataset_dir, "data")
    labels_path = os.path.join(
        os.path.expanduser("~"), "fiftyone", "coco-2017", "raw", "captions_val2017.json"
    )

    if not os.path.exists(labels_path):
        print("Error: Caption file not found.")
        return

    # --- Manually Load and Parse Annotations ---
    print("Parsing caption annotation file...")
    with open(labels_path, "r") as f:
        captions_data = json.load(f)

    # Create a mapping from image_id to a list of captions
    image_id_to_captions = {}
    for ann in captions_data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in image_id_to_captions:
            image_id_to_captions[image_id] = []
        image_id_to_captions[image_id].append(ann["caption"])

    # Create a mapping from image_id to filepath
    image_id_to_filepath = {
        img["id"]: os.path.join(data_path, img["file_name"])
        for img in captions_data["images"]
    }

    # --- Manually Create FiftyOne Dataset ---
    dataset_name = "coco-2017-validation-captions-final"
    if fo.dataset_exists(dataset_name):
        dataset = fo.load_dataset(dataset_name)
        dataset.delete()  # Ensure a clean slate

    dataset = fo.Dataset(dataset_name)

    samples = []
    for image_id, filepath in image_id_to_filepath.items():
        captions = image_id_to_captions.get(image_id)
        if captions:  # Only add samples that have captions
            sample = fo.Sample(filepath=filepath)
            # Add the list of captions as a 'ground_truth' field of type 'Detections'
            sample["ground_truth"] = fo.Detections(
                detections=[fo.Detection(label=caption) for caption in captions]
            )
            samples.append(sample)

    dataset.add_samples(samples)

    print("\n--- MS-COCO Analysis Complete ---")
    print(f"Total valid image-caption pairs found: {len(dataset)}")


if __name__ == "__main__":
    analyze_coco_dataset()
