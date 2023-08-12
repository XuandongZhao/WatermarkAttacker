from PIL import Image
import cv2
import torch
import os
from imwatermark import WatermarkEncoder, WatermarkDecoder
from torchvision import transforms
import subprocess


class Watermarker:
    def encode(self, img_path, output_path, prompt=''):
        raise NotImplementedError

    def decode(self, img_path):
        raise NotImplementedError


class InvisibleWatermarker(Watermarker):
    def __init__(self, wm_text, method):
        if method == 'rivaGan':
            WatermarkEncoder.loadModel()
        self.method = method
        self.encoder = WatermarkEncoder()
        self.wm_type = 'bytes'
        self.wm_text = wm_text
        self.decoder = WatermarkDecoder(self.wm_type, len(self.wm_text) * 8)

    def encode(self, img_path, output_path):
        img = cv2.imread(img_path)
        self.encoder.set_watermark(self.wm_type, self.wm_text.encode('utf-8'))
        out = self.encoder.encode(img, self.method)
        cv2.imwrite(output_path, out)

    def decode(self, img_path):
        wm_img = cv2.imread(img_path)
        wm_text_decode = self.decoder.decode(wm_img, self.method)
        return wm_text_decode


class StableSignatureWatermarker(Watermarker):
    def __init__(self, stable_diffusion_root_path, msg_extractor, script, key='111010110101000001010111010011010100010000100111'):
        self.stable_diffusion_root_path = stable_diffusion_root_path
        self.key = key
        self.msg_extractor = msg_extractor = torch.jit.load(msg_extractor).to("cuda")
        self.transform_imnet = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.script = script

    def encode(self, img_path, output_dir, prompt=''):
        command = [
            'python', os.path.join(self.stable_diffusion_root_path, f'scripts/{self.script}'),
            '--prompt', prompt,
            '--ckpt', os.path.join(self.stable_diffusion_root_path, 'checkpoints/v2-1_512-ema-pruned.ckpt'),
            '--config', os.path.join(self.stable_diffusion_root_path, 'configs/stable-diffusion/v2-inference.yaml'),
            '--H', '512',
            '--W', '512',
            '--device', 'cuda',
            '--outdir', output_dir,
            '--img_name', img_path,
            '--n_samples', '1',
            '--n_rows', '1',
        ]
        result = subprocess.run(command, capture_output=True, text=True)

        # Print the output or handle error
        if result.returncode != 0:
            print('Error:', result.stderr)
        else:
            print('Output:', result.stdout)

    def decode(self, img_path):
        img = Image.open(img_path)
        img = self.transform_imnet(img).unsqueeze(0).to("cuda")
        msg = self.msg_extractor(img)  # b c h w -> b k
        bool_msg = (msg > 0).squeeze().cpu().numpy().tolist()

        bool_key = StableSignatureWatermarker.str2msg(self.key)
        # compute difference between model key and message extracted from image
        diff = [bool_msg[i] != bool_key[i] for i in range(len(bool_msg))]
        bit_acc = 1 - sum(diff) / len(diff)
        print("Bit accuracy: ", bit_acc)

        # compute p-value
        from scipy.stats import binomtest
        pval = binomtest(sum(diff), len(diff), 0.5)
        print("p-value of statistical test: ", pval)
        return bit_acc, pval

    def msg2str(msg):
        return "".join([('1' if el else '0') for el in msg])

    def str2msg(str):
        return [True if el == '1' else False for el in str]
