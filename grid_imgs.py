
import re
import cv2
import time
import shutil
import random
import ffmpeg
import prosodic
import subprocess
import numpy as np
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw
from natsort import natsorted
from moviepy.editor import *

src = "/is/cluster/work/rdanecek/testing/enspark/ablations/2023_05_18_01-26-32_-6224330163499889169_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmmmLmm/mturk_videos_lrs3/test"
FF_evoca = "/is/cluster/work/rdanecek/testing/enspark/ablations/2023_05_10_13-16-00_-3885098104460673227_FaceFormer_MEADP_Awav2vec2T_Elinear_DFaceFormerDecoder_Seml_PPE_predV_LV/mturk_videos_lrs3/test" 

CT = "/is/cluster/fast/scratch/rdanecek/testing/enspark/baselines/CT_test_reorg"
MT = "/is/cluster/fast/scratch/rdanecek/testing/enspark/baselines/MT_test_reorg"
FF = "/is/cluster/fast/scratch/rdanecek/testing/enspark/baselines/FF_test_reorg" 
VOCA = "/is/cluster/fast/scratch/rdanecek/testing/enspark/baselines/VOCA_test_reorg"

lrs3_samples = "/ps/project/EmotionalFacialAnimation/data/lrs3/extracted/test"
lrs3_audios_only = "/is/cluster/fast/scratch/rdanecek/testing/enspark/baselines/lrs3_audios_only"

grid_imgs = "/is/cluster/fast/scratch/rdanecek/testing/enspark/baselines/grid_imgs/v4"

subject = "M003"
emotion = "Neutral"
generations = 5
images_per_row = 6
max_txt_len = 10                   
without_phoneme = True

# samples:
# 7,8: none
# 9: 2 

desired_dirs = [str(p) for p in Path(src).glob("*/*/*") if all([subject in str(p), emotion in str(p)])] # 118
# desired dirs based on FF_evoca
desired_dirs = [str(p) for p in Path(FF_evoca).glob("*/*/*") if all([subject in str(p), emotion in str(p)])] 

loop = [(9, True), (9, False), (10, True), (11, True), (11, False)]

for kk, (max_txt_len, without_phoneme) in enumerate(loop):
    for i in range(generations):
        for f in Path(grid_imgs).glob("_tmp_*"): shutil.rmtree(f)
        
        while True:
            
            try: sampled = random.sample(desired_dirs, 1)[0]
            except: print(f"max txt len: {max_txt_len}, without phoneme: {without_phoneme} has no samples"); break
            
            desired_dirs.remove(sampled)
            sampled_id = Path(sampled).parts[-3:-1]
            sampled_full_id = Path(sampled).parts[-3:]
            
            script = Path(lrs3_samples) / sampled_id[0] / f"{sampled_id[1]}.txt"
            if not script.exists(): continue
            
            script_txt = open(script, "r").readline().strip().rsplit("Text:  ", 1)[-1]
            txt_count = len(re.findall(r'\w+', script_txt))
            if txt_count > max_txt_len: continue
            
            audio = Path(lrs3_audios_only) / sampled_id[0] / f"{sampled_id[1]}.wav"
            if not audio.exists(): continue
            
            FF_evoca_dir = sampled
            CT_dir = Path(CT) / sampled_id[0] / sampled_id[1]
            MT_dir = Path(MT) / sampled_id[0] / sampled_id[1]
            FF_dir = Path(FF) / sampled_id[0] / sampled_id[1]
            VOCA_dir = Path(VOCA) / sampled_id[0] / sampled_id[1]
            emote_dir = Path(src) / sampled_full_id[0] / sampled_full_id[1] / sampled_full_id[2]
            
            emote_vid = list(Path(sampled).rglob("*.mp4"))
            # FF_evoca_vid = list(Path(FF_evoca_dir).rglob("*.mp4"))
            CT_vid = list(Path(CT_dir).rglob("*.mp4"))
            MT_vid = list(Path(MT_dir).rglob("*.mp4"))
            FF_vid = list(Path(FF_dir).rglob("*.mp4"))
            VOCA_vid = list(Path(VOCA_dir).rglob("*.mp4"))
            
            # mesh dirs
            emote_new_mesh = Path(emote_dir) / "one_view_new"
            FF_evoca_new_mesh = Path(sampled) / "one_view_new"
            CT_new_mesh = Path(CT_dir) / "one_view_new"
            MT_new_mesh = Path(MT_dir) / "one_view_new"
            FF_new_mesh = Path(FF_dir) / "one_view_new"
            VOCA_new_mesh = Path(VOCA_dir) / "one_view_new"
            
            if not emote_new_mesh.exists() or not FF_evoca_new_mesh.exists() or not CT_new_mesh.exists() or not MT_new_mesh.exists() or not FF_new_mesh.exists() or not VOCA_new_mesh.exists(): continue
            
            if len(emote_vid) == len(CT_vid) == len(MT_vid) == len(FF_vid) == len(VOCA_vid) == 0: continue
            break
        
        tmp_dir = Path(grid_imgs) / f"_tmp_{sampled_id}_imgs_{images_per_row}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        
        for mesh_dir, fps, name in zip([emote_new_mesh, CT_new_mesh, MT_new_mesh, FF_new_mesh, VOCA_new_mesh, FF_evoca_new_mesh], [25, 30, 30, 30, 120, 30], ["emote", "CT", "MT", "FF", "VOCA", "FF_variant"]):
            subprocess.call([
                "ffmpeg", "-framerate", str(fps), "-i", str(mesh_dir)+"/%05d.png", "-vcodec", "ffvhuff", str(tmp_dir / f"{name}.avi")
            ])
            video = str(tmp_dir / f"{name}.avi")
            out_vid = str(tmp_dir / f"{name}_downsampled.avi")
            subprocess.call([
                "ffmpeg", "-i", video, "-filter_complex", "color=white,format=rgb24[c];[c][0]scale2ref[c][i];[c][i]overlay=format=auto:shortest=1,setsar=1", "-c:a", "copy", out_vid
            ])
            
        # for vid, name in {emote_vid[0]: "emote", CT_vid[0]: "CT", MT_vid[0]: "MT", FF_vid[0]: "FF", VOCA_vid[0]: "VOCA"}.items():
        #     subprocess.call([
        #         "ffmpeg", "-i", str(vid), "-filter:v", "fps=fps=25", str(tmp_dir / f"{name}.mp4")
        #     ])
        
        # emote_vid = tmp_dir / "emote.mp4"
        # CT_vid = tmp_dir / "CT.mp4"
        # MT_vid = tmp_dir / "MT.mp4"
        # FF_vid = tmp_dir / "FF.mp4"
        # VOCA_vid = tmp_dir / "VOCA.mp4"
        
        emote_vid = tmp_dir / "emote_downsampled.avi"
        CT_vid = tmp_dir / "CT_downsampled.avi"
        MT_vid = tmp_dir / "MT_downsampled.avi"
        FF_vid = tmp_dir / "FF_downsampled.avi"
        VOCA_vid = tmp_dir / "VOCA_downsampled.avi"
        FF_evoca_vid = tmp_dir / "FF_variant_downsampled.avi"
        
        # stressed, syllabified, phonetic transcription
        if not without_phoneme:
            t = prosodic.Text(script_txt)
            t.parse()
            for i in t.bestParses(): phonemes = i 
            phonemes = phonemes.posString(viols=True)
        
        framecount = VideoFileClip(str(emote_vid)).reader.nframes
        selected_frames = np.linspace(0, framecount-1, images_per_row+1, dtype=int)
        movies = [emote_vid, FF_evoca_vid, FF_vid, CT_vid, VOCA_vid, MT_vid]
        model_names = ["Ours", "FF_variant", "Faceformer", "Codetalker", "VOCA", "Meshtalk"]
        
        for movie in movies:
            if "VOCA" in str(movie):
                probe = ffmpeg.probe(str(movie))
                video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
                fps = video_info['avg_frame_rate']
                frame_count = int(video_info['nb_frames'])
                VOCA_selected_frames = np.linspace(0, frame_count-1, images_per_row+1, dtype=int)
            if "FF_variant" in str(movie):
                probe = ffmpeg.probe(str(movie))
                video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
                fps = video_info['avg_frame_rate']
                frame_count = int(video_info['nb_frames'])
                FF_evoca_selected_frames = np.linspace(0, frame_count-1, images_per_row+1, dtype=int)
        all_select_frames = [selected_frames, FF_evoca_selected_frames, selected_frames, selected_frames, VOCA_selected_frames, selected_frames]
        
        for movie, modelname, select_frames in zip(movies, model_names, all_select_frames):
            for i, t in enumerate(select_frames):
                imgpath = os.path.join(str(tmp_dir), f"{modelname}_{i}.jpg")
                subprocess.call([
                    "ffmpeg", "-i", str(movie), "-vf", f"select='eq(n\,{t})'", "-vsync", "0", imgpath
                ])
        
        emote_frames = natsorted(list(Path(tmp_dir).glob("Ours_*.jpg")), key=lambda x: int(x.stem.split("_")[-1]))
        CT_frames = natsorted(list(Path(tmp_dir).glob("Codetalker_*.jpg")), key=lambda x: int(x.stem.split("_")[-1]))
        MT_frames = natsorted(list(Path(tmp_dir).glob("Meshtalk_*.jpg")), key=lambda x: int(x.stem.split("_")[-1]))
        FF_frames = natsorted(list(Path(tmp_dir).glob("Faceformer_*.jpg")), key=lambda x: int(x.stem.split("_")[-1]))
        VOCA_frames = natsorted(list(Path(tmp_dir).glob("VOCA_*.jpg")), key=lambda x: int(x.stem.split("_")[-1]))  
        FF_evoca_frames = natsorted(list(Path(tmp_dir).glob("FF_variant_*.jpg")), key=lambda x: int(x.stem.split("_")[-1]))
        CT_frames = CT_frames[:len(emote_frames)]
        MT_frames = MT_frames[:len(emote_frames)]
        FF_frames = FF_frames[:len(emote_frames)]
        VOCA_frames = VOCA_frames[:len(emote_frames)]
        FF_evoca_frames = FF_evoca_frames[:len(emote_frames)]
        frames_order = [emote_frames, FF_evoca_frames, FF_frames, CT_frames, VOCA_frames, MT_frames]
        
        for frames in frames_order:
            imgs = frames
            imgs = [Image.open(img) for img in imgs]
            min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
            imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))
            imgs_comb = Image.fromarray(imgs_comb)
            imgs_comb.save(os.path.join(str(tmp_dir), f"row_{frames_order.index(frames)}.jpg"))
        
        gen_dir = Path(grid_imgs) / f"{sampled_id[0]}_{sampled_id[1]}_imgs_{images_per_row}_maxtxt_{max_txt_len}"
        if not without_phoneme: gen_dir = Path(grid_imgs) / f"{sampled_id[0]}_{sampled_id[1]}_imgs_{images_per_row}_maxtxt_{max_txt_len}_phonemes"
        gen_dir.mkdir(parents=True, exist_ok=True)
        
        combined_imgs = natsorted(list(Path(tmp_dir).glob("row_*.jpg")), key=lambda x: int(x.stem.split("_")[-1]))  
        # waveform = Image.open(Path(grid_imgs) / "assets" / "waveform.png") 
        ximg, yimg = 480, 650   
        yoffset = 20    
        xlen, ylen = 480 * images_per_row, (650 * len(combined_imgs)) + (yoffset * (len(combined_imgs)-1))
        img = Image.new('RGB', (xlen, ylen), (0, 0, 0))
        index = 0
        for i in range(0, xlen, ximg):
            for j in range(0, ylen, yimg + yoffset):
                if index == len(combined_imgs): break
                im = Image.open(combined_imgs[index])
                img.paste(im, (i, j))
                index += 1
        # img.paste(waveform, (500, 1200))
        # img.save(gen_dir / "tight_grid.jpg")
        img.save(tmp_dir / "tmp_tight_grid.jpg")
        
        img = Image.open(tmp_dir / "tmp_tight_grid.jpg")
        img = img.convert("RGBA")
        datas = img.getdata()

        newData = []
        for item in datas:
            if item[0] < 10 and item[1] < 10 and item[2] < 10:
                newData.append((255, 255, 255))
            else:
                newData.append(item)

        img.putdata(newData)
        img.save(gen_dir / "tight_grid.png", "PNG")
        
        # new img with white bg
        yoffset = 0    
        xlen, ylen = 480 * (images_per_row + 1), (650 * (len(combined_imgs) + 1)) + (yoffset * (len(combined_imgs)-1))
        img = Image.new('RGB', (xlen, ylen), (255, 255, 255))
        index = 0
        for i in range(ximg, xlen, ximg):
            for j in range(yimg, ylen, yimg + yoffset):
                if index == len(combined_imgs): break
                im = Image.open(combined_imgs[index])
                img.paste(im, (i, j))
                index += 1
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf", 65)
        font_script = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf", 50)
        for i, j in enumerate(range(int(1.5 * yimg), int(ylen + (yimg/2)), yimg)):
            draw.text((50, j), f"{model_names[i]}", (0, 0, 0), font=font)

        if not without_phoneme:
            draw.text((20, 120), script_txt, (0, 0, 0), font=font_script)
            draw.text((20, 240), "Phonemes:", (0, 0, 0), font=font_script)    
            draw.text((20, 360), phonemes, (0, 0, 0), font=font_script)
            save_path = os.path.join(str(gen_dir), f"with_txt_{txt_count}_phonemes.jpg")
        else:
            font_script = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf", 85)
            draw.text((20, 240), script_txt, (0, 0, 0), font=font_script)
            save_path = os.path.join(str(gen_dir), f"with_txt_{txt_count}.jpg")
        img.save(save_path)
        
        with open(os.path.join(str(gen_dir), f"dialogue.txt"), "w") as f: f.write(script_txt)
        if not without_phoneme:
            with open(os.path.join(str(gen_dir), f"phonemes.txt"), "w") as f: f.write(phonemes)
        
        for f in Path(grid_imgs).glob("_tmp_*"): shutil.rmtree(f)