import concurrent.futures
import os

import config
from freq_feature import FreqFeature
from utils import save_train_data


def process_file(each_category, each_file_path):
    print("Processing {}".format(each_file_path))
    full_path = os.path.join(config.data_path, each_category, each_file_path)
    freq_feature = FreqFeature(full_path, csv_path=os.path.join(config.data_path, each_category, "REFERENCE-SQI.csv"), use_dwt=True)
    this_wav_mfccs, labels = freq_feature.get_mfccs_features_and_labels()
    save_train_data(this_wav_mfccs, labels,
                    save_dir=os.path.join(config.train_data_dwt_save_dir, "{}_".format(each_file_path.split(".")[0])))
    return this_wav_mfccs.shape, labels.shape


def extract_features():
    category_list = ["training-a", "training-b", "training-c", "training-d", "training-e", "training-f"]
    for each_category in category_list:
        file_path_list = [wav_file for wav_file in os.listdir(os.path.join(config.data_path, each_category))
                          if wav_file.endswith(".wav")]

        current_category_len = len(file_path_list)
        print("Processing {} files in category {}".format(current_category_len, each_category))

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(process_file, [each_category] * current_category_len, file_path_list)

            for count, result in enumerate(results, start=1):
                print("Processing Done! {}/{}".format(count, current_category_len), result)


if __name__ == "__main__":
    extract_features()
