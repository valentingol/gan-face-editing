import sys
import zipfile
from PIL import Image

reference_image_names = {
    "0_0_0_0_0_1_1_0_0",
    "0_0_0_0_1_0_2_0_0",
    "0_0_0_1_0_0_1_0_0",
    "0_0_0_1_1_1_1_0_0",
    "0_0_1_0_0_0_2_0_0",
    "0_0_1_0_1_0_1_0_0",
    "0_0_1_1_0_0_2_1_1",
    "0_0_1_1_1_1_2_0_0",
    "0_1_0_0_0_0_2_0_0",
    "0_1_0_0_1_0_0_0_0",
    "0_1_0_1_0_0_0_1_0",
    "0_1_0_1_1_1_1_0_0",
    "0_1_1_0_0_0_1_0_0",
    "0_1_1_0_1_0_0_0_1",
    "0_1_1_1_0_0_1_1_0",
    "0_1_1_1_1_0_1_0_0",
    "0_2_0_0_0_0_3_0_0",
    "0_2_0_0_1_0_3_0_0",
    "0_2_0_1_0_0_3_1_0",
    "0_2_0_1_1_0_3_0_0",
    "0_2_1_0_0_0_3_0_0",
    "0_2_1_0_1_0_1_0_0",
    "0_2_1_1_0_0_3_0_0",
    "0_2_1_1_1_1_1_0_0",
    "1_0_0_0_0_0_0_0_0",
    "1_0_0_0_1_0_3_0_0",
    "1_0_0_1_0_0_0_1_0",
    "1_0_0_1_1_0_0_0_0",
    "1_0_1_0_0_1_0_0_0",
    "1_0_1_0_1_1_0_0_0",
    "1_0_1_1_0_0_2_0_0",
    "1_0_1_1_1_1_0_0_1",
    "1_1_0_0_0_0_0_0_0",
    "1_1_0_0_1_0_4_1_2",
    "1_1_0_1_0_0_0_0_0",
    "1_1_0_1_1_0_2_0_0",
    "1_1_1_0_0_0_0_0_0",
    "1_1_1_0_1_0_2_0_1",
    "1_1_1_1_0_1_2_0_0",
    "1_1_1_1_1_0_2_1_1",
    "1_2_0_0_0_0_0_0_0",
    "1_2_0_0_1_0_0_0_0",
    "1_2_0_1_0_0_3_0_0",
    "1_2_0_1_1_0_0_0_0",
    "1_2_1_0_0_0_2_0_0",
    "1_2_1_0_1_0_0_0_0",
    "1_2_1_1_0_0_3_0_0",
    "1_2_1_1_1_0_3_0_0",
    "2_0_0_0_0_0_2_0_1",
    "2_0_0_0_1_0_0_0_0",
    "2_0_0_1_0_0_0_0_0",
    "2_0_0_1_1_0_0_0_0",
    "2_0_1_0_0_0_0_0_1",
    "2_0_1_0_1_0_0_0_1",
    "2_0_1_1_0_0_0_0_1",
    "2_0_1_1_1_0_0_0_1",
    "2_1_0_0_0_0_0_0_1",
    "2_1_0_0_1_0_0_0_0",
    "2_1_0_1_0_0_4_1_2",
    "2_1_0_1_1_0_0_0_1",
    "2_1_1_0_0_0_0_0_0",
    "2_1_1_0_1_0_2_0_1",
    "2_1_1_1_0_0_0_0_1",
    "2_1_1_1_1_0_4_0_2",
    "2_2_0_0_0_0_0_0_1",
    "2_2_0_0_1_0_0_0_1",
    "2_2_0_1_0_0_0_1_1",
    "2_2_0_1_1_0_3_0_0",
    "2_2_1_0_0_0_0_1_1",
    "2_2_1_0_1_0_0_0_0",
    "2_2_1_1_0_0_0_0_1",
    "2_2_1_1_1_0_0_0_0",
}

possible_values = {
    "Sk": {"0", "1", "2"},
    "A": {"0", "1", "2"},
    "Se": {"0", "1"},
    "C": {"0", "1"},
    "P": {"0", "1"},
    "B": {"0", "1"},
    "Hc": {"0", "1", "2", "3"}, # baldness is treated elsewhere
    "D": {"0", "1"},
    "Hs": {"0", "1"}, # baldness is treated elsewhere
}

indicative_features = ("C", "P")
baldness_sensitive_features = ("Hc", "Hs")
cursor_features = ("Be", "N", "Pn", "Bp", "Bn", "Ch")


def build_expected_images_set():

    result = set()

    for reference_name in reference_image_names:

        Sk, A, Se, C, P, B, Hc, D, Hs = reference_name.split("_")
        current_value = {
            "Sk": Sk,
            "A": A,
            "Se": Se,
            "C": C,
            "P": P,
            "B": B,
            "Hc": Hc,
            "D": D,
            "Hs": Hs,
        }

        # Adding to the list of expected results all variations which
        # don't rely on a potential baldness
        for feature_identifier in possible_values:
            if (feature_identifier not in indicative_features
                and feature_identifier not in baldness_sensitive_features):
                for possible_value in possible_values[feature_identifier]:
                    if possible_value != current_value[feature_identifier]:
                        result.add(reference_name + "/" + feature_identifier \
                                   + "_" + possible_value+".png")
        for identifier in cursor_features:
            result.update({reference_name+"/"+identifier+"_min"+".png",
                           reference_name+"/"+identifier+"_max"+".png"})

        # Adding expected results if the reference image is not bald
        if Hc != "4":
            for possible_value in possible_values["Hc"]:
                if possible_value != current_value["Hc"]:
                    result.add(reference_name + "/" + "Hc" + "_" \
                               + possible_value + ".png")
            for possible_value in possible_values["Hs"]:
                if possible_value != current_value["Hs"]:
                    result.add(reference_name+ "/" + "Hs" + "_" \
                        + possible_value + ".png")
            result.add(reference_name+"/"+"bald"+".png")

        # Adding expected results if the reference image is bald
        else:
            for possible_value in possible_values["Hc"]:
                result.add(reference_name+"/"+"Hc"+"_"+possible_value+".png")
            for possible_value in possible_values["Hs"]:
                result.add(reference_name+"/"+"Hs"+"_"+possible_value+".png")

    return result




def check_exactly_all_needed_folders_are_there(zip_object):

    zip_item_names = zip_object.namelist()
    folder_names = set([item for item in zip_item_names if item[-1] == "/"])

    reference_image_names_with_slash = set(
        [folder_name+"/" for folder_name in reference_image_names]
        )
    if folder_names != reference_image_names_with_slash:
        print("The folders in the submission are not the expected ones.")

        # Determining what folders are missing / should not be there
        expected_folders_not_in_the_submission = reference_image_names_with_slash.difference(folder_names)
        unexpected_folders_in_the_submission = folder_names.difference(reference_image_names_with_slash)

        # If there are missing folders in the submission
        if len(expected_folders_not_in_the_submission) > 0:
            print("- Some folders are missing: "
                  f"{expected_folders_not_in_the_submission}")

        # If there are folders which should not be in the submission
        if len(unexpected_folders_in_the_submission) > 0:
            print("- Some folders are in the submission but were unexpected: "
                  f"{unexpected_folders_in_the_submission}")

        print("Make sure every folder has the name of a reference image and "
              "that all folders are present.")

        return False
    return True


def check_exactly_all_images_are_there(zip_object):

    zip_item_names = zip_object.namelist()
    file_names = set([item for item in zip_item_names if item[-1] != "/"])

    expected_images = build_expected_images_set()

    if file_names != expected_images:
        print("The images in the submission are not the expected ones.")

        # Determining what folders are missing / should not be there
        expected_images_not_in_the_submission = expected_images.difference(file_names)
        unexpected_files_in_the_submission = file_names.difference(expected_images)

        # If there are missing images in the submission
        if len(expected_images_not_in_the_submission) > 0:
            print(f"- Some images are missing: {expected_images_not_in_the_submission}. "
                  "Make sure that all required images are there (include a "
                  "copy of the reference image with the proper modified name "
                  "if you could not process it to the target).")

        # If there are folders which should not be in the submission
        if len(unexpected_files_in_the_submission) > 0:
            print("- Some files are in the submission but were unexpected: "
                  f"{unexpected_files_in_the_submission}. Make sure your "
                  "submission only contains the folders and images that were asked.")

        return False

    return True

def check_images_have_the_right_format(zip_file):

    for entry in zip_file.infolist():

        if entry.filename[-1] == "/":
            continue

        with zip_file.open(entry) as image_file:

            try:
                image = Image.open(image_file)
            except:
                print("Submitted images should be proper image files. "
                      f"At least {entry.filename} is not.")
                return False
            if image.mode != "RGB":
                print("Submitted images should be at RGB format. At least "
                      f"{entry.filename} is not.")
                return False
            if image.size != (512, 512):
                print("Submitted images should be 512*512 images. At least "
                      f"{entry.filename} is not, it is {image.size}.")
                return False

    return True

def check_submission(zipfile_path):

    with zipfile.ZipFile(zipfile_path) as zip_object:

        all_good = check_exactly_all_needed_folders_are_there(zip_object)

        if all_good:
            all_good = check_exactly_all_images_are_there(zip_object)

        if all_good:
            all_good = check_images_have_the_right_format(zip_object)

        if all_good:
            print("Your submission looks good!")
        else:
            print("\nYour submission is incorrect as it is currently. "
                  "Please take into account the feedback above to correct it.")

if __name__ == "__main__":
    submission_path = sys.argv[-1]
    check_submission(submission_path)
