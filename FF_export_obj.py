
"""
python FF_export_obj.py --wav_path "/ps/project/EmotionalFacialAnimation/data/mead/ensparc_audios"
change: 
    --condition "FaceTalk_170913_03279_TA"
    --subject "FaceTalk_170809_00138_TA"
    --template_path "templates.pkl"
out: "/ps/project/EmotionalFacialAnimation/data/mead/ensparc_audios/FF"
"""

import numpy as np
import scipy.io.wavfile as wav
import librosa
import os,sys,shutil,argparse,copy,pickle
import math,scipy
from faceformer import Faceformer
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor
from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from moviepy.editor import *

import cv2
import tempfile
from subprocess import call
os.environ['PYOPENGL_PLATFORM'] = 'osmesa' # egl
import pyrender
from psbody.mesh import Mesh
import trimesh

@torch.no_grad()
def test_model(args):
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    
    #build model
    model = Faceformer(args)
    model.load_state_dict(torch.load(os.path.join(args.dataset, '{}.pth'.format(args.model_name))))
    model = model.to(torch.device(args.device))
    model.eval()

    template_file = os.path.join(args.dataset, args.template_path)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1')

    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    one_hot_labels = np.eye(len(train_subjects_list))
    iter = train_subjects_list.index(args.condition)
    one_hot = one_hot_labels[iter]
    one_hot = np.reshape(one_hot,(-1,one_hot.shape[0]))
    one_hot = torch.FloatTensor(one_hot).to(device=args.device)

    temp = templates[args.subject]
             
    template = temp.reshape((-1))
    template = np.reshape(template,(-1,template.shape[0]))
    template = torch.FloatTensor(template).to(device=args.device)
    template_file = os.path.join(args.dataset, args.render_template_path, "FLAME_sample.ply")
    template_4obj = Mesh(filename=template_file)
    
    print("Starting mesh extraction...")
    
    for i, p in enumerate(Path(args.wav_path).glob('*')):
        sample_dir = p.stem
        out_path = os.path.join(args.result_path, sample_dir)
        os.makedirs(out_path, exist_ok=True)
        print(f"Processing {sample_dir} ({i+1}/{len(list(Path(args.wav_path).glob('*')))})")
        audio_fname = os.path.join(args.wav_path, sample_dir)
    
        # for wav_path in tqdm(Path(args.wav_path).glob('*.m4a'), total=len(list(Path(args.wav_path).glob('*.m4a')))):
        for wav_path in tqdm(Path(audio_fname).glob('*.wav')):
            
            test_name = wav_path.stem
            # videoclip = VideoFileClip(str(wav_path)) # lrs3 are wav video files
            # audioclip = videoclip.audio
            # audio_dir  = Path(out_path) / test_name/ "audio"
            # audio_dir.mkdir(parents=True, exist_ok=True)
            # wav_path = audio_dir / f"audio.wav"
            # audioclip.write_audiofile(wav_path)
            
            npy_dir = Path(out_path) / test_name/ "npy"
            npy_dir.mkdir(parents=True, exist_ok=True)
            
            obj_dir = Path(out_path) / test_name / "meshes"
            obj_dir.mkdir(parents=True, exist_ok=True)
            
            # test_name = os.path.basename(wav_path).split(".")[0]
            # track = AudioSegment.from_file(wav_path, format="m4a")
            # _ = track.export(f"{Path(args.wav_path) / test_name}.wav", format="wav")
            # wav_path = os.path.join(args.wav_path, test_name+".wav")
            
            speech_array, _ = librosa.load(os.path.join(wav_path), sr=16000)
            processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            audio_feature = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
            audio_feature = np.reshape(audio_feature,(-1,audio_feature.shape[0]))
            audio_feature = torch.FloatTensor(audio_feature).to(device=args.device)
            prediction = model.predict(audio_feature, template, one_hot)
            prediction = prediction.squeeze() # (seq_len, V*3)
            np.save(npy_dir / f"{test_name}.npy", prediction.detach().cpu().numpy()) 
            predicted_vertices_path = npy_dir / f"{test_name}.npy"
            predicted_vertices = np.load(predicted_vertices_path)
            predicted_vertices = np.reshape(predicted_vertices,(-1,args.vertice_dim//3,3))
            num_frames = predicted_vertices.shape[0]
            center = np.mean(predicted_vertices[0], axis=0)
            for i_frame in range(num_frames):
                render_mesh = Mesh(predicted_vertices[i_frame], template_4obj.f)
                obj_path = obj_dir / f"{i_frame:05d}.obj"
                _ = render_mesh_helper(args,render_mesh, center, export_obj=True, obj_path=obj_path)
        
    # for wav_path in Path(args.wav_path).glob('*.wav'): wav_path.unlink()

# The implementation of rendering is borrowed from VOCA: https://github.com/TimoBolkart/voca/blob/master/utils/rendering.py
def render_mesh_helper(args,mesh, t_center, rot=np.zeros(3), tex_img=None, z_offset=0, export_obj=False, obj_path=None):
    if args.dataset == "BIWI":
        camera_params = {'c': np.array([400, 400]),
                         'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                         'f': np.array([4754.97941935 / 8, 4754.97941935 / 8])}
    elif args.dataset == "vocaset":
        camera_params = {'c': np.array([400, 400]),
                         'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                         'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])}

    frustum = {'near': 0.01, 'far': 3.0, 'height': 800, 'width': 800}

    mesh_copy = Mesh(mesh.v, mesh.f)
    mesh_copy.v[:] = cv2.Rodrigues(rot)[0].dot((mesh_copy.v-t_center).T).T+t_center
    intensity = 2.0
    rgb_per_v = None

    primitive_material = pyrender.material.MetallicRoughnessMaterial(
                alphaMode='BLEND',
                baseColorFactor=[0.3, 0.3, 0.3, 1.0],
                metallicFactor=0.8, 
                roughnessFactor=0.8 
            )

    tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f, vertex_colors=rgb_per_v)
    if export_obj: tri_mesh.export(obj_path); return
    
    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=primitive_material,smooth=True)

    if args.background_black:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0])
    else:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[255, 255, 255])
    camera = pyrender.IntrinsicsCamera(fx=camera_params['f'][0],
                                      fy=camera_params['f'][1],
                                      cx=camera_params['c'][0],
                                      cy=camera_params['c'][1],
                                      znear=frustum['near'],
                                      zfar=frustum['far'])

    scene.add(render_mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[:3,3] = np.array([0, 0, 1.0-z_offset])
    scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]])

    angle = np.pi / 6.0
    pos = camera_pose[:3,3]
    light_color = np.array([1., 1., 1.])
    light = pyrender.DirectionalLight(color=light_color, intensity=intensity)

    light_pose = np.eye(4)
    light_pose[:3,3] = pos
    scene.add(light, pose=light_pose.copy())
    
    light_pose[:3,3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] =  cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    try:
        r = pyrender.OffscreenRenderer(viewport_width=frustum['width'], viewport_height=frustum['height'])
        color, _ = r.render(scene, flags=flags)
    except:
        print('pyrender: Failed rendering frame')
        color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

    return color[..., ::-1]

def main():
    parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
    parser.add_argument("--model_name", type=str, default="vocaset")
    parser.add_argument("--dataset", type=str, default="vocaset", help='vocaset or BIWI')
    parser.add_argument("--fps", type=float, default=30, help='frame rate - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=64, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=30, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--vertice_dim", type=int, default=15069, help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--device", type=str, default="cuda")
    # BIWI
    # parser.add_argument("--train_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    # vocaset 
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA") 
    # BIWI
    # parser.add_argument("--test_subjects", type=str, default="F1 F5 F6 F7 F8 M1 M2 M6")
    # vocaset
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA")
    parser.add_argument("--output_path", type=str, default="/is/cluster/scratch/kchhatre/Work/ENSPARC/baselines/FaceFormer/demo/output", help='path of the rendered video sequence')
    parser.add_argument("--wav_path", type=str, default="/is/cluster/fast/scratch/rdanecek/testing/enspark/baselines/lrs3_audios_only", help='path of the input audio signal')
    parser.add_argument("--result_path", type=str, default="/is/cluster/fast/scratch/rdanecek/testing/enspark/baselines/FF_lrs3_test/sub3", help='path of the predictions')
    # # BIWI
    # parser.add_argument("--condition", type=str, default="M3", help='select a conditioning subject from train_subjects')
    # parser.add_argument("--subject", type=str, default="M1", help='select a subject from test_subjects or train_subjects')
    # vocaset
    parser.add_argument("--condition", type=str, default="FaceTalk_170904_00128_TA", help='select a conditioning subject from train_subjects')
    parser.add_argument("--subject", type=str, default="FaceTalk_170809_00138_TA", help='select a subject from test_subjects or train_subjects')
    parser.add_argument("--background_black", type=bool, default=True, help='whether to use black background')
    parser.add_argument("--template_path", type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument("--render_template_path", type=str, default="templates", help='path of the mesh in BIWI/FLAME topology')
    args = parser.parse_args()   

    test_model(args)

if __name__=="__main__":
    main()
    print("Changing permissions of the output files")
    os.system("find /is/cluster/fast/scratch/rdanecek/testing/enspark/baselines -type d -exec chmod 775 {} +")
    print("Done")

    # subject: FaceTalk_170809_00138_TA
    # condition:
    # sub1: FaceTalk_170913_03279_TA
    # sub2: FaceTalk_170728_03272_TA
    # sub3:  FaceTalk_170904_00128_TA 

