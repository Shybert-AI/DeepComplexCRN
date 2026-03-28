import os
from glob import iglob
from pathlib import Path

from config import Config

config = Config()
test_0_list = config.test_0_list
tr_list = config.tr_list
dev_list = config.dev_list
os.makedirs(os.path.dirname(tr_list),exist_ok=True)
os.makedirs(os.path.dirname(tr_list),exist_ok=True)
os.makedirs(os.path.dirname(tr_list),exist_ok=True)

base_path =  "D:/mywork/pythonProject/android_project/DeepComplexCRN-main"
for type_ in ["train","test"]:
    basepath = f"dataset/THCHS-30/data_synthesized/{type_}/0dB/clean/*wav"
    dataset = list(iglob(basepath,recursive=True))
    if type_ == "test":
        with open(dev_list,"w") as f:
            with open(test_0_list, "w") as f1:
                for i,line in enumerate(dataset):

                    clean_path = Path(os.path.join(base_path,line)).as_posix()
                    noisy_path = clean_path.replace('clean','noisy')
                    f.write(f"{noisy_path.replace('.wav','_cafe.wav')} {clean_path}\n")
                    f.write(f"{noisy_path.replace('.wav','_car.wav')} {clean_path}\n")
                    f.write(f"{noisy_path.replace('.wav','_white.wav')} {clean_path}\n")
                    if i < 20:
                        f1.write(f"{noisy_path}\n")

    else:
        with open(tr_list,"w") as f:
            for line in dataset:
                clean_path = Path(os.path.join(base_path,line)).as_posix()
                noisy_path = clean_path.replace('clean','noisy')
                f.write(f"{noisy_path.replace('.wav', '_cafe.wav')} {clean_path}\n")
                f.write(f"{noisy_path.replace('.wav', '_car.wav')} {clean_path}\n")
                f.write(f"{noisy_path.replace('.wav', '_white.wav')} {clean_path}\n")


