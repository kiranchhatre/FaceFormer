
import subprocess
from tqdm import tqdm
from pathlib import Path

lrs3_samples = "/ps/project/EmotionalFacialAnimation/data/lrs3/extracted/test"
lrs3_audios_only = "/is/cluster/fast/scratch/rdanecek/testing/enspark/baselines/lrs3_audios_only"

# for dir in tqdm(Path(lrs3_samples).glob('*')):
#     sample_dir = dir.stem
#     out_path = Path(lrs3_audios_only) / sample_dir
#     out_path.mkdir(parents=True, exist_ok=True)
#     for vids in dir.glob('*.mp4'):
#         vid_name = vids.stem
#         subprocess.call(f"ffmpeg -i {str(vids)} -q:a 0 -map a {str(out_path / vid_name)}.wav", shell=True)

mp4s = list(Path(lrs3_samples).rglob('*.mp4'))
wavs = list(Path(lrs3_audios_only).rglob('*.wav'))
print(f"mp4s: {len(mp4s)}, wavs: {len(wavs)}") # mp4s: 1321, wavs: 1321



CT_lrs3 = "/is/cluster/fast/scratch/rdanecek/testing/enspark/baselines/CT_lrs3_test"
FF_lrs3 = "/is/cluster/fast/scratch/rdanecek/testing/enspark/baselines/FF_lrs3_test"
VOCA_lrs3 = "/is/cluster/fast/scratch/rdanecek/testing/enspark/baselines/VOCA_lrs3_test"
MT_lrs3 = "/is/cluster/fast/scratch/rdanecek/testing/enspark/baselines/MT_lrs3_test"

