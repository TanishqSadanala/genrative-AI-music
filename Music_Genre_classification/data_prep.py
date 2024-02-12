import os
import librosa
import math
import json

dataset_path = "C:/Users/tanis/OneDrive/Desktop/Python/Generative_AI/genres_original"
json_path = "C:/Users/tanis/OneDrive/Desktop/Python/Generative_AI/data.json"
sample_rate = 22050
track_duration = 30
samples_per_track = sample_rate * track_duration

def mfcc_saving(dataset_path, json_path, n_mfcc=13, n_fft=2048, offset = 512, num_segments=5):
    data = {
        "genre" : [],
        "mfcc"  : [],
        "labels": []
    }
    num_samples_per_segment = int(samples_per_track/num_segments)
    expected_mfcc_vectors = math.ceil(num_samples_per_segment/offset)
    
    for i,(root, dirs, files) in enumerate(os.walk(dataset_path)):
        print(f'Current directory: {root}')
        print(f'\nSubdirectories: {dirs}')
        print(f'\nFiles: {files}')
        if root is not dataset_path:
            root_components = root.split("/")
            semantic_label = root_components[-1]
            data["genre"].append(semantic_label)
            print("\nworking on {}".format(semantic_label))
            for f in files:
                file_path = os.path.join(root, f)
                signal, sr = librosa.load(file_path, sr=sample_rate)
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    end_sample = start_sample + num_samples_per_segment
                    mfcc = librosa.feature.mfcc(y=signal[start_sample:end_sample], sr = sr, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=offset) 
                    mfcc = mfcc.T
                    if len(mfcc) == expected_mfcc_vectors:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, s+1))
        
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
if __name__ == "__main__":
    mfcc_saving(dataset_path, json_path, num_segments=10)
    
