
"""
Usage:
python prepare_catch.py --runid "run0" --instance 3
"""

import glob
import shutil
import random
import argparse
import subprocess 
from pathlib import Path
from moviepy.editor import *

# src = "/is/cluster/fast/scratch/rdanecek/testing/enspark/ablations/2023_05_08_20-36-09_8797431074914794141_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmmmLmm/mturk_videos_lrs3/pretrain"
src = "/is/cluster/work/rdanecek/testing/enspark/ablations/2023_05_18_01-26-32_-6224330163499889169_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmmmLmm/mturk_videos_lrs3/test"
CT_pretrain = "/is/cluster/fast/scratch/rdanecek/testing/enspark/baselines/CT_reorg"
MT_pretrain = "/is/cluster/fast/scratch/rdanecek/testing/enspark/baselines/MT_new_reorg"
FF_pretrain = "/is/cluster/fast/scratch/rdanecek/testing/enspark/baselines/FF_reorg"     
VOCA_pretrain = "/is/cluster/fast/scratch/rdanecek/testing/enspark/baselines/VOCA_reorg"

CT = "/is/cluster/fast/scratch/rdanecek/testing/enspark/baselines/CT_test_reorg"
MT = "/is/cluster/fast/scratch/rdanecek/testing/enspark/baselines/MT_test_reorg"
FF = "/is/cluster/fast/scratch/rdanecek/testing/enspark/baselines/FF_test_reorg"     
VOCA = "/is/cluster/fast/scratch/rdanecek/testing/enspark/baselines/VOCA_test_reorg"

mismatched_aud = "/is/cluster/scratch/kchhatre/Work/ENSPARC/baselines/mismatch_audio/mismatched.mp3"
catch_paths = "/is/cluster/fast/scratch/rdanecek/testing/enspark/catch_trials/new_renders"

info = {
    "study_1a": {"classify": 1, "lipsync": 0, "pair": 1, "sota": 0},
    "study_1b": {"classify": 0, "lipsync": 1, "pair": 1, "sota": 0},
    "study_2": {"classify": 1, "lipsync": 0, "pair": 0, "sota": 0},
    "study_3": {"classify": 0, "lipsync": 1, "pair": 1, "sota": 1},
    "study_4": {"classify": 0, "lipsync": 1, "pair": 0, "sota": 0}
}
SOTA = [CT, MT, FF, VOCA]
# SOTA = [CT, MT]
Emotions = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise", "Contempt"]
studies = ["study_1a", "study_2", "study_3", "study_1b", "study_4"]

"""
1a: CLASSIFY our model,  x2 (neutral, emo: happy/ sad) save emo
1b: LIPSYNC our model, x2 (fetch diff video)
2: CLASSIFY x1 our model (easy emo)
3: LIPSYNC SOTA x2 (fetch diff video- all Neutral emos) (ours, baseline 4)
4: LIPSYNC x1 our model (fetch diff video)
"""

class catchTrials():
    def __init__(self, runid, instance):
        self.runid = runid
        self.instance = instance
        self.random_audio_duration = AudioFileClip(mismatched_aud).duration
        self.ensparc_dirs = [str(p) for p in Path(src).glob("*")]
        runid_dir = Path(catch_paths) / runid
        if Path(runid_dir).is_dir():
            if any(runid_dir.iterdir()): shutil.rmtree(runid_dir)
        Path(runid_dir).mkdir(parents=True, exist_ok=True)
        for study in studies:
            study_path = Path(catch_paths) / runid / study
            study_path.mkdir(parents=True, exist_ok=True)
        self.replace_audio = CompositeAudioClip([AudioFileClip(mismatched_aud)])
    
    def prepare_catch_v0(self):
        
        print(f"Preparing catch trials for {self.runid}...")
        
        for study in studies:
            k, v = study, info[study]
            if v["classify"]:
                if v["pair"]: self.classify_pair(k) 
                else: self.classify_single(k, self.instance) 
            if v["lipsync"]:
                if v["sota"]: self.lipsync_pair_sota(k) 
                else:
                    if v["pair"]: self.lipsync_pair(k) 
                    else: self.lipsync_single(k) 
        for path in Path(catch_paths).rglob("*only*"): path.unlink()
        
    def prepare_catch_v1(self):
        
        print(f"Preparing catch trials for {self.runid}...")
        
        for study in studies:
            k, v = study, info[study]
            if v["classify"]:
                # if v["pair"]: self.classify_pair_v1(k) 
                if v["pair"]: self.classify_pair_v2(k)
                else: self.classify_single(k, 2) 
            if v["lipsync"]:
                if v["sota"]: self.lipsync_pair_sota_v1(k) 
                else:
                    if v["pair"]: self.lipsync_pair_v2(k) 
                    else: self.lipsync_single(k) 
        for path in Path(catch_paths).rglob("*only*"): path.unlink()

    def classify_pair(self, key):
        print(f"Preparing study {key}...")
        while True:
            audio_gen = random.choice(self.ensparc_dirs)
            self.ensparc_dirs = self.ensparc_dirs.remove(audio_gen)
            all_emo_audio_gen = [str(p) for p in Path(audio_gen).glob("*/*")]
            neutral_audio_gen = [j for j in all_emo_audio_gen if "Neutral" in j][0]
            neutral_video = [k for k in glob.glob(f"{neutral_audio_gen}/*") if "mp4" in k] 
            if len(neutral_video) == 0: continue
            neutral_video = neutral_video[0]
            all_emo_audio_gen.remove(neutral_audio_gen)
            break
        for i in range(self.instance):
            emo1 = random.choice(all_emo_audio_gen)
            emotion = [j for j in Emotions if j in emo1][0]
            all_emo_audio_gen.remove(emo1)
            emo1_video = [j for j in glob.glob(f"{emo1}/*") if "mp4" in j][0]
            shutil.copy(neutral_video, Path(catch_paths) / self.runid / key / f"pair_{i}_neutral_answer_false.mp4")
            shutil.copy(emo1_video, Path(catch_paths) / self.runid / key / f"pair_{i}_{emotion}_answer_true.mp4")
        print(f"Study {key} done")       
    
    def classify_pair_v2(self, key):
        print(f"Preparing study {key}...")
        M003_dirs = [p for p in Path(src).glob("*/*/*") if "M003" in str(p)]
        for kk in range(2):
            while True:
                one_gen = random.choice(M003_dirs)
                one_gen_root = str(one_gen.parent)
                neutral_dir = [k for k in glob.glob(f"{one_gen_root}/*") if "Neutral" in k and "M003" in k]
                happy_dir = [k for k in glob.glob(f"{one_gen_root}/*") if "Happy" in k and "M003" in k]
                sad_dir = [k for k in glob.glob(f"{one_gen_root}/*") if "Sad" in k and "M003" in k]
                neutral_video = [j for j in glob.glob(f"{neutral_dir[0]}/*") if "mp4" in j]
                happy_video = [j for j in glob.glob(f"{happy_dir[0]}/*") if "mp4" in j]
                sad_video = [j for j in glob.glob(f"{sad_dir[0]}/*") if "mp4" in j]
                if len(neutral_video) == 0 or len(happy_video) == 0 or len(sad_video) == 0: continue
                break
            shutil.copy(neutral_video[0], Path(catch_paths) / self.runid / key / f"pair_{kk}_neutral_answer_false.mp4")
            if kk == 0:
                shutil.copy(happy_video[0], Path(catch_paths) / self.runid / key / f"pair_{kk}_Happy_answer_true.mp4")
            else:
                shutil.copy(sad_video[0], Path(catch_paths) / self.runid / key / f"pair_{kk}_Sad_answer_true.mp4")
        print(f"Study {key} done")
    
    def classify_pair_v1(self, key):
        print(f"Preparing study {key}...")
        M003_dirs = [p for p in Path(src).glob("*/*/*") if "M003" in str(p)]
        M003_dirs = [str(p.parents[1]) for p in M003_dirs]
        for kk in range(2):
            while True:
                # audio_gen = random.choice(self.ensparc_dirs)
                audio_gen = random.choice(M003_dirs) # M003 conditioned ensparc generations
                # self.ensparc_dirs = self.ensparc_dirs.remove(audio_gen)
                all_emo_audio_gen = [str(p) for p in Path(audio_gen).glob("*/*")]
                neutral_audio_gen = [j for j in all_emo_audio_gen if "Neutral" in j][0]
                neutral_video = [k for k in glob.glob(f"{neutral_audio_gen}/*") if "mp4" in k]
                if len(neutral_video) == 0: continue
                neutral_video = neutral_video[0]
                break
            shutil.copy(neutral_video, Path(catch_paths) / self.runid / key / f"pair_{kk}_neutral_answer_false.mp4")
        while True:
            audio_gen = random.choice(M003_dirs)
            happy_audio_gen = [j for j in all_emo_audio_gen if "Happy" in j] 
            happy_audio_gen = random.choice(happy_audio_gen)
            happy_video = [k for k in glob.glob(f"{happy_audio_gen}/*") if "mp4" in k]
            if len(happy_video) == 0: continue
            happy_video = happy_video[0]
            sad_audio_gen = [j for j in all_emo_audio_gen if "Sad" in j]
            sad_audio_gen = random.choice(sad_audio_gen)
            sad_video = [k for k in glob.glob(f"{sad_audio_gen}/*") if "mp4" in k]
            if len(sad_video) == 0: continue
            sad_video = sad_video[0]
            shutil.copy(happy_video, Path(catch_paths) / self.runid / key / f"pair_0_Happy_answer_true.mp4")
            shutil.copy(sad_video, Path(catch_paths) / self.runid / key / f"pair_1_Sad_answer_true.mp4")
            break
        print(f"Study {key} done")   
    
    def classify_single(self, key, instances):
        print(f"Preparing study {key}...")
        copied_videos = (Path(catch_paths) / self.runid / "study_1a").glob("*")
        emotional_videos = [str(p) for p in copied_videos if "answer_true" in str(p)]
        assert len(emotional_videos) == instances, "Incorrect number of videos copied"
        for i in range(instances):
            emotion = [j for j in Emotions if j in emotional_videos[i]][0]
            shutil.copy(emotional_videos[i], Path(catch_paths) / self.runid / key / f"pair_{i}_{emotion}_answer_true.mp4")
        print(f"Study {key} done")

    def lipsync_pair_v2(self, key):
        print(f"Preparing study {key}...")
        self.ensparc_dirs = [str(p) for p in Path(src).glob("*")]
        M003_dirs = [p for p in Path(src).glob("*/*/*") if "M003" in str(p)]
        for i in range(self.instance):
            while True:
                
                one_gen = random.choice(M003_dirs)
                M003_dirs.remove(one_gen)
                one_gen_root = str(one_gen.parent)
                neutral_dir = [k for k in glob.glob(f"{one_gen_root}/*") if "Neutral" in k and "M003" in k]
                lipsynced_video = [j for j in glob.glob(f"{neutral_dir[0]}/*") if "mp4" in j]
                
                one_gen = random.choice(M003_dirs)
                M003_dirs.remove(one_gen)
                one_gen_root = str(one_gen.parent)
                neutral_dir = [k for k in glob.glob(f"{one_gen_root}/*") if "Neutral" in k and "M003" in k]
                nonlipsynced_video = [j for j in glob.glob(f"{neutral_dir[0]}/*") if "mp4" in j]
                
                if len(lipsynced_video) == 0 or len(nonlipsynced_video) == 0: continue
                break
            
            lipsynced_video = lipsynced_video[0]
            nonlipsynced_video = nonlipsynced_video[0]
            shutil.copy(lipsynced_video, str(Path(catch_paths) / self.runid / key / f"pair_{i}_answer_true.mp4"))
            subprocess.call(f"ffmpeg -i {nonlipsynced_video} -c copy -an {str(Path(catch_paths) / self.runid / key / f'pair_{i}_only_video.mp4')}", shell=True)
            subprocess.call(f"ffmpeg -i {lipsynced_video} -q:a 0 -map a {str(Path(catch_paths) / self.runid / key / f'pair_{i}_only_audio.mp3')}", shell=True)
            subprocess.call(f"ffmpeg -i {str(Path(catch_paths) / self.runid / key / f'pair_{i}_only_video.mp4')} -i {str(Path(catch_paths) / self.runid / key / f'pair_{i}_only_audio.mp3')} -c copy {str(Path(catch_paths) / self.runid / key / f'pair_{i}_answer_false.mp4')}", shell=True)
        print(f"Study {key} done")
    
    def lipsync_pair_v1(self, key):
        print(f"Preparing study {key}...")
        self.ensparc_dirs = [str(p) for p in Path(src).glob("*")]
        M003_dirs = [p for p in Path(src).glob("*/*/*") if "M003" in str(p)]
        M003_dirs = [str(p.parents[1]) for p in M003_dirs]
        for i in range(self.instance):
            while True:
                # one_gen = random.choice(self.ensparc_dirs)
                one_gen = random.choice(M003_dirs)
                one_gen_videos = [str(p) for p in glob.glob(f"{one_gen}/*/*/*") if "mp4" in str(p)]
                if len(one_gen_videos) == 0: continue
                lipsynced_video = [j for j in one_gen_videos if "Neutral" in j][0]
                # lipsynced_video = random.choice(one_gen_videos)
                # self.ensparc_dirs.remove(one_gen)
                # unpaired_gen = random.choice(self.ensparc_dirs)
                # self.ensparc_dirs.remove(unpaired_gen)
                M003_dirs.remove(one_gen)
                unpaired_gen = random.choice(M003_dirs)
                M003_dirs.remove(unpaired_gen)
                unpaired_gen_videos = [str(p) for p in glob.glob(f"{unpaired_gen}/*/*/*") if "mp4" in str(p)]
                if len(unpaired_gen_videos) == 0: continue
                nonlipsynced_video = [j for j in unpaired_gen_videos if "Neutral" in j][0]
                # nonlipsynced_video = random.choice(unpaired_gen_videos)
                break
            # 2 vids: lipsynced_video, nonlipsynced_video
            shutil.copy(lipsynced_video, str(Path(catch_paths) / self.runid / key / f"pair_{i}_answer_true.mp4"))
            subprocess.call(f"ffmpeg -i {nonlipsynced_video} -c copy -an {str(Path(catch_paths) / self.runid / key / f'pair_{i}_only_video.mp4')}", shell=True)
            # nonlipsync_duration = VideoFileClip(nonlipsynced_video).duration
            subprocess.call(f"ffmpeg -i {lipsynced_video} -q:a 0 -map a {str(Path(catch_paths) / self.runid / key / f'pair_{i}_only_audio.mp3')}", shell=True)
            # rand_audio_start = random.randint(0, int(self.random_audio_duration - nonlipsync_duration)-5)
            subprocess.call(f"ffmpeg -i {str(Path(catch_paths) / self.runid / key / f'pair_{i}_only_video.mp4')} -i {str(Path(catch_paths) / self.runid / key / f'pair_{i}_only_audio.mp3')} -c copy {str(Path(catch_paths) / self.runid / key / f'pair_{i}_answer_false.mp4')}", shell=True)
            # subprocess.call(f"ffmpeg -ss {rand_audio_start} -t {nonlipsync_duration} -i {mismatched_aud} {str(Path(catch_paths) / self.runid / key / f'pair_{i}_only_audio.mp3')}", shell=True)
            # subprocess.call(f"ffmpeg -i {str(Path(catch_paths) / self.runid / key / f'pair_{i}_only_video.mp4')} -i {str(Path(catch_paths) / self.runid / key / f'pair_{i}_only_audio.mp3')} -c copy {str(Path(catch_paths) / self.runid / key / f'pair_{i}_answer_false.mp4')}", shell=True)
        print(f"Study {key} done")
    
    def lipsync_pair(self, key):
        print(f"Preparing study {key}...")
        self.ensparc_dirs = [str(p) for p in Path(src).glob("*")]
        for i in range(self.instance):
            while True:
                one_gen = random.choice(self.ensparc_dirs)
                one_gen_videos = [str(p) for p in glob.glob(f"{one_gen}/*/*/*") if "mp4" in str(p)]
                if len(one_gen_videos) == 0: continue
                lipsynced_video = random.choice(one_gen_videos)
                self.ensparc_dirs.remove(one_gen)
                unpaired_gen = random.choice(self.ensparc_dirs)
                self.ensparc_dirs.remove(unpaired_gen)
                unpaired_gen_videos = [str(p) for p in glob.glob(f"{unpaired_gen}/*/*/*") if "mp4" in str(p)]
                if len(unpaired_gen_videos) == 0: continue
                nonlipsynced_video = random.choice(unpaired_gen_videos)
                break
            shutil.copy(lipsynced_video, str(Path(catch_paths) / self.runid / key / f"pair_{i}_answer_true.mp4"))
            subprocess.call(f"ffmpeg -i {nonlipsynced_video} -c copy -an {str(Path(catch_paths) / self.runid / key / f'pair_{i}_only_video.mp4')}", shell=True)
            nonlipsync_duration = VideoFileClip(nonlipsynced_video).duration
            rand_audio_start = random.randint(0, int(self.random_audio_duration - nonlipsync_duration)-5)
            subprocess.call(f"ffmpeg -ss {rand_audio_start} -t {nonlipsync_duration} -i {mismatched_aud} {str(Path(catch_paths) / self.runid / key / f'pair_{i}_only_audio.mp3')}", shell=True)
            subprocess.call(f"ffmpeg -i {str(Path(catch_paths) / self.runid / key / f'pair_{i}_only_video.mp4')} -i {str(Path(catch_paths) / self.runid / key / f'pair_{i}_only_audio.mp3')} -c copy {str(Path(catch_paths) / self.runid / key / f'pair_{i}_answer_false.mp4')}", shell=True)
        print(f"Study {key} done")

    def lipsync_single(self, key):
        print(f"Preparing study {key}...")
        copied_videos = (Path(catch_paths) / self.runid / "study_1b").glob("*")
        incorrect_videos = [str(p) for p in copied_videos if "answer_false" in str(p)]
        assert len(incorrect_videos) == self.instance, "Incorrect number of videos copied"
        for i in range(self.instance):
            shutil.copy(incorrect_videos[i], Path(catch_paths) / self.runid / key / f"pair_{i}_answer_true.mp4")
        print(f"Study {key} done")

    def lipsync_pair_sota_v1(self, key):
        print(f"Preparing study {key}...")
        self.ensparc_dirs = [str(p) for p in Path(src).glob("*/*")]
        for i in range(self.instance):
            for sota in SOTA:
                which_sota = sota.split("/")[-1].split("_")[0]
                sota_paths = [str(p) for p in Path(sota).glob("*/*")]
                while True:
                    sota_path = random.choice(sota_paths)
                    sota_paths.remove(sota_path)
                    last_2_dirs = Path(sota_path).parts[-2:]
                    common_ensparc_path = [p for p in self.ensparc_dirs if last_2_dirs[0] in p and last_2_dirs[1] in p]
                    if len(common_ensparc_path) == 0: continue
                    common_ensparc_path = common_ensparc_path[0]
                    self.ensparc_dirs.remove(common_ensparc_path)
                    common_ensparc_path_videos = [str(p) for p in glob.glob(f"{common_ensparc_path}/*/*") if "mp4" in str(p)]
                    if len(common_ensparc_path_videos) == 0: continue
                    ensparc_video = [j for j in common_ensparc_path_videos if "Neutral" in j][0]
                    # ensparc_video = random.choice(common_ensparc_path_videos)
                    unpaired_enparc_path = random.choice(self.ensparc_dirs)
                    self.ensparc_dirs.remove(unpaired_enparc_path)
                    unpaired_enparc_path_videos = [str(p) for p in glob.glob(f"{unpaired_enparc_path}/*/*") if "mp4" in str(p)]
                    if len(unpaired_enparc_path_videos) == 0: continue
                    unpaired_ensparc_video = [j for j in unpaired_enparc_path_videos if "Neutral" in j][0]
                    # unpaired_ensparc_video = random.choice(unpaired_enparc_path_videos)
                    break
                sota_video = [str(p) for p in glob.glob(f"{sota_path}/*") if "mp4" in str(p)][0]
                # 3 vids: sota_video, ensparc_video, unpaired_ensparc_video
                if i % 2 == 0:
                    shutil.copy(sota_video, Path(catch_paths) / self.runid / key / f"pair_{i}_{which_sota}_video_answer_true.mp4")
                    subprocess.call(f"ffmpeg -y -i {unpaired_ensparc_video} -c copy -an {str(Path(catch_paths) / self.runid / key / f'pair_{i}_only_video.mp4')}", shell=True)
                    subprocess.call(f"ffmpeg -y -i {sota_video} -q:a 0 -map a {str(Path(catch_paths) / self.runid / key / f'pair_{i}_only_audio.mp3')}", shell=True)
                    subprocess.call(f"ffmpeg -i {str(Path(catch_paths) / self.runid / key / f'pair_{i}_only_video.mp4')} -i {str(Path(catch_paths) / self.runid / key / f'pair_{i}_only_audio.mp3')} -c copy {str(Path(catch_paths) / self.runid / key / f'pair_{i}_{which_sota}_ensparc_video_answer_false.mp4')}", shell=True)
                else:
                    shutil.copy(unpaired_ensparc_video, Path(catch_paths) / self.runid / key / f"pair_{i}_{which_sota}_ensparc_video_answer_true.mp4")
                    subprocess.call(f"ffmpeg -y -i {sota_video} -c copy -an {str(Path(catch_paths) / self.runid / key / f'pair_{i}_only_video.mp4')}", shell=True)
                    subprocess.call(f"ffmpeg -y -i {unpaired_ensparc_video} -q:a 0 -map a {str(Path(catch_paths) / self.runid / key / f'pair_{i}_only_audio.mp3')}", shell=True)
                    subprocess.call(f"ffmpeg -i {str(Path(catch_paths) / self.runid / key / f'pair_{i}_only_video.mp4')} -i {str(Path(catch_paths) / self.runid / key / f'pair_{i}_only_audio.mp3')} -c copy {str(Path(catch_paths) / self.runid / key / f'pair_{i}_{which_sota}_video_answer_false.mp4')}", shell=True)
        print(f"Study {key} done")
    
    def lipsync_pair_sota(self, key):
        print(f"Preparing study {key}...")
        self.ensparc_dirs = [str(p) for p in Path(src).glob("*/*")]
        for i in range(self.instance):
            for sota in SOTA:
                which_sota = sota.split("/")[-1].split("_")[0]
                sota_paths = [str(p) for p in Path(sota).glob("*/*")]
                while True:
                    sota_path = random.choice(sota_paths)
                    sota_paths.remove(sota_path)
                    last_2_dirs = Path(sota_path).parts[-2:]
                    common_ensparc_path = [p for p in self.ensparc_dirs if last_2_dirs[0] in p and last_2_dirs[1] in p]
                    if len(common_ensparc_path) == 0: continue
                    common_ensparc_path = common_ensparc_path[0]
                    self.ensparc_dirs.remove(common_ensparc_path)
                    common_ensparc_path_videos = [str(p) for p in glob.glob(f"{common_ensparc_path}/*/*") if "mp4" in str(p)]
                    if len(common_ensparc_path_videos) == 0: continue
                    ensparc_video = random.choice(common_ensparc_path_videos)
                    unpaired_enparc_path = random.choice(self.ensparc_dirs)
                    self.ensparc_dirs.remove(unpaired_enparc_path)
                    unpaired_enparc_path_videos = [str(p) for p in glob.glob(f"{unpaired_enparc_path}/*/*") if "mp4" in str(p)]
                    if len(unpaired_enparc_path_videos) == 0: continue
                    unpaired_ensparc_video = random.choice(unpaired_enparc_path_videos)
                    break
                sota_video = [str(p) for p in glob.glob(f"{sota_path}/*") if "mp4" in str(p)][0]
                unpaired_clip = VideoFileClip(unpaired_ensparc_video)
                unpaired_audio = unpaired_clip.audio
                unpaired_audio = CompositeAudioClip([unpaired_audio])
                # swapping audio with one of the video: sota or ensparc
                paired_videos = [sota_video, ensparc_video]
                if i % 2 == 0:
                    shutil.copy(paired_videos[0], Path(catch_paths) / self.runid / key / f"pair_{i}_{which_sota}_answer_true.mp4")
                    subprocess.call(f"ffmpeg -i {paired_videos[1]} -c copy -an {str(Path(catch_paths) / self.runid / key / f'pair_{i}_{which_sota}_only_video.mp4')}", shell=True)
                    ensparc_duration = VideoFileClip(ensparc_video).duration
                    rand_audio_start = random.randint(0, int(self.random_audio_duration - ensparc_duration)-5)
                    subprocess.call(f"ffmpeg -ss {rand_audio_start} -t {ensparc_duration} -i {mismatched_aud} {str(Path(catch_paths) / self.runid / key / f'pair_{i}_{which_sota}_only_audio.mp3')}", shell=True)
                    subprocess.call(f"ffmpeg -i {str(Path(catch_paths) / self.runid / key / f'pair_{i}_{which_sota}_only_video.mp4')} -i {str(Path(catch_paths) / self.runid / key / f'pair_{i}_{which_sota}_only_audio.mp3')} -c copy {str(Path(catch_paths) / self.runid / key / f'pair_{i}_{which_sota}_answer_false.mp4')}", shell=True)
                else:
                    shutil.copy(paired_videos[1], Path(catch_paths) / self.runid / key / f"pair_{i}_{which_sota}_ensparc_answer_true.mp4")
                    subprocess.call(f"ffmpeg -i {paired_videos[0]} -c copy -an {str(Path(catch_paths) / self.runid / key / f'pair_{i}_{which_sota}_only_video.mp4')}", shell=True)
                    sota_duration = VideoFileClip(sota_video).duration
                    rand_audio_start = random.randint(0, int(self.random_audio_duration - sota_duration)-5)
                    subprocess.call(f"ffmpeg -ss {rand_audio_start} -t {sota_duration} -i {mismatched_aud} {str(Path(catch_paths) / self.runid / key / f'pair_{i}_{which_sota}_only_audio.mp3')}", shell=True)
                    subprocess.call(f"ffmpeg -i {str(Path(catch_paths) / self.runid / key / f'pair_{i}_{which_sota}_only_video.mp4')} -i {str(Path(catch_paths) / self.runid / key / f'pair_{i}_{which_sota}_only_audio.mp3')} -c copy {str(Path(catch_paths) / self.runid / key / f'pair_{i}_{which_sota}_answer_false.mp4')}", shell=True)
        print(f"Study {key} done")
                
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Prepare catch trials')
    parser.add_argument('--runid', type=str, default="run0", help='Run id')
    # parser.add_argument('--instance', type=int, default=3, help='Number of instances per study') # 3
    parser.add_argument('--instance', type=int, default=1, help='Number of instances per study') # test 1 2
    ct = catchTrials(parser.parse_args().runid, parser.parse_args().instance)
    # ct.prepare_catch_v0()
    ct.prepare_catch_v1()
    