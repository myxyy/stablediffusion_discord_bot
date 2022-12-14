import argparse, os, discord, random, string, asyncio
from datetime import datetime
from multiprocessing.connection import Connection
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from config import Config
from multiprocessing import Process, Pipe

import urllib.error
import urllib.request

import PIL
from PIL import Image

parser = argparse.ArgumentParser()

queuecount = 0
ticketcount = 0

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open(Config.PATH_TO_SD + "assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x

#parser.add_argument(
#    "--prompt",
#    type=str,
#    nargs="?",
#    default="a painting of a virus monster playing guitar",
#    help="the prompt to render"
#)
#parser.add_argument(
#    "--outdir",
#    type=str,
#    nargs="?",
#    help="dir to write results to",
#    default=Config.PATH_TO_SD+"outputs/txt2img-samples"
#)
parser.add_argument(
    "--skip_grid",
    action='store_true',
    help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
)
parser.add_argument(
    "--naive-weight",
    action='store_true',
    help="do not normalize weight"
)
parser.add_argument(
    "--skip_save",
    action='store_true',
    help="do not save individual samples. For speed measurements.",
)
parser.add_argument(
    "--ddim_steps",
    type=int,
    default=50,
    help="number of ddim sampling steps",
)
parser.add_argument(
    "--plms",
    action='store_true',
    help="use plms sampling",
)
parser.add_argument(
    "--laion400m",
    action='store_true',
    help="uses the LAION400M model",
)
parser.add_argument(
    "--float16",
    action='store_true',
    help="uses the LAION400M model",
)
parser.add_argument(
    "--fixed_code",
    action='store_true',
    help="if enabled, uses the same starting code across samples ",
)
parser.add_argument(
    "--repeat",
    action='store_true',
    help="repeat ",
)
parser.add_argument(
    "--init-img",
    type=str,
    nargs="?",
    help="path to the input image"
)
parser.add_argument(
    "--ddim_eta",
    type=float,
    default=0.0,
    help="ddim eta (eta=0.0 corresponds to deterministic sampling",
)
parser.add_argument(
    "--n_iter",
    type=int,
    default=2,
    help="sample this often",
)
parser.add_argument(
    "--H",
    type=int,
    default=512,
    help="image height, in pixel space",
)
parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="image width, in pixel space",
)
parser.add_argument(
    "--C",
    type=int,
    default=4,
    help="latent channels",
)
parser.add_argument(
    "--f",
    type=int,
    default=8,
    help="downsampling factor",
)
parser.add_argument(
    "--n_samples",
    type=int,
    default=3,
    help="how many samples to produce for each given prompt. A.k.a. batch size",
)
parser.add_argument(
    "--n_rows",
    type=int,
    default=0,
    help="rows in the grid (default: n_samples)",
)
parser.add_argument(
    "--scale",
    type=float,
    default=7.5,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
#parser.add_argument(
#    "--from-file",
#    type=str,
#    help="if specified, load prompts from this file",
#)
#parser.add_argument(
#    "--config",
#    type=str,
#    default=Config.PATH_TO_SD+"configs/stable-diffusion/v1-inference.yaml",
#    help="path to config which constructs model",
#)
parser.add_argument(
    "--ckpt",
    type=str,
    help="path to checkpoint of model",
    choices=list(Config.MODEL_DICT.keys()),
    default=list(Config.MODEL_DICT.keys())[0]
)
parser.add_argument(
    "--strength",
    type=float,
    default=0.75,
    help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
)
parser.add_argument(
    "--seed",
    type=int,
    help="the seed (for reproducible sampling)",
)
parser.add_argument(
    "--precision",
    type=str,
    help="evaluate at this precision",
    choices=["full", "autocast"],
    default="autocast"
)

# calc
def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def img2img(opt, filename: str, seed: int, connection: Connection):
    try:
        def patch_conv(klass):
            init = klass.__init__
            def __init__(self, *args, **kwargs):
                return init(self, *args, **kwargs, padding_mode='circular' if opt.repeat else 'zeros')
            klass.__init__ = __init__

        for klass in [torch.nn.Conv2d, torch.nn.ConvTranspose2d]:
            patch_conv(klass)

        seed_everything(seed)

        config = OmegaConf.load(Config.PATH_TO_SD+"configs/stable-diffusion/v1-inference.yaml")
        model = load_model_from_config(config, Config.MODEL_DICT[opt.ckpt])

        if opt.float16:
            model.to(torch.float16)
        
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)

        if opt.plms:
            raise NotImplementedError("PLMS sampler not (yet) supported")
            sampler = PLMSSampler(model)
        else:
            sampler = DDIMSampler(model)

        os.makedirs(Config.OUTDIR, exist_ok=True)
        outpath = Config.OUTDIR

        batch_size = opt.n_samples
        n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)

        assert os.path.isfile(opt.init_img)
        init_image = load_img(opt.init_img).to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

        sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

        assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(opt.strength * opt.ddim_steps)
        print(f"target t_enc is {t_enc} steps")

        precision_scope = autocast if opt.precision == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    all_samples = list()
                    index = 0
                    for n in trange(opt.n_iter, desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if opt.scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            c = None
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            for i in range(len(prompts[0])):
                                ps = []
                                ws = []
                                for pwpairs in prompts:
                                    (p,w) = pwpairs[i]
                                    ps += [p]
                                    ws += [w]
                                if c == None:
                                    c = model.get_learned_conditioning(ps) * ws[0]
                                else:
                                    c += model.get_learned_conditioning(ps) * ws[0]
 
                            # encode (scaled latent)
                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                            # decode it
                            samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                                    unconditional_conditioning=uc,)

                            x_samples = model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                            if not opt.skip_save:
                                for x_sample in x_samples:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    Image.fromarray(x_sample.astype(np.uint8)).save(
                                        os.path.join(sample_path, f"{filename}_{index}.png"))
                                    index += 1
                            all_samples.append(x_samples)

                    if not opt.skip_grid:
                        # additionally, save as grid
                        grid = torch.stack(all_samples, 0)
                        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                        grid = make_grid(grid, nrow=n_rows)

                        # to image
                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'{filename}.png'))

        print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
            f" \nEnjoy.")
    except:
        import traceback
        traceback.print_exc()
        connection.send(-1)
    else:
        connection.send(index)


def txt2img(opt, filename: str, seed: int, connection: Connection):
    def patch_conv(klass):
        init = klass.__init__
        def __init__(self, *args, **kwargs):
            return init(self, *args, **kwargs, padding_mode='circular' if opt.repeat else 'zeros')
        klass.__init__ = __init__

    for klass in [torch.nn.Conv2d, torch.nn.ConvTranspose2d]:
        patch_conv(klass)

    # load
    try:
        seed_everything(seed)

        config = OmegaConf.load(Config.PATH_TO_SD+"configs/stable-diffusion/v1-inference.yaml")
        model = load_model_from_config(config, Config.MODEL_DICT[opt.ckpt])

        if opt.float16:
            model.to(torch.float16)
        
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)

        if opt.plms:
            sampler = PLMSSampler(model)
        else:
            sampler = DDIMSampler(model)

        os.makedirs(Config.OUTDIR, exist_ok=True)
        outpath = Config.OUTDIR

        batch_size = opt.n_samples
        n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        grid_count = len(os.listdir(outpath)) - 1

        start_code = None

        precision_scope = autocast if opt.precision=="autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    tic = time.time()
                    all_samples = list()
                    index = 0
                    for n in trange(min(opt.n_iter,4), desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if opt.scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            c = None
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            for i in range(len(prompts[0])):
                                ps = []
                                ws = []
                                for pwpairs in prompts:
                                    (p,w) = pwpairs[i]
                                    ps += [p]
                                    ws += [w]
                                if c == None:
                                    c = model.get_learned_conditioning(ps) * ws[0]
                                else:
                                    c += model.get_learned_conditioning(ps) * ws[0]
                            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                            samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                            conditioning=c,
                                                            batch_size=opt.n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=opt.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=opt.ddim_eta,
                                                            x_T=start_code)

                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                            x_checked_image = x_samples_ddim

                            x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                            if not opt.skip_save:
                                for x_sample in x_checked_image_torch:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    img = Image.fromarray(x_sample.astype(np.uint8))
                                    img.save(os.path.join(sample_path, f"{filename}_{index}.png"))
                                    index += 1

                            if not opt.skip_grid:
                                all_samples.append(x_checked_image_torch)

                    if not opt.skip_grid:
                        # additionally, save as grid
                        grid = torch.stack(all_samples, 0)
                        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                        grid = make_grid(grid, nrow=n_rows)

                        # to image
                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        img = Image.fromarray(grid.astype(np.uint8))
                        img.save(os.path.join(outpath, f'{filename}.png'))
                        grid_count += 1

                    toc = time.time()

        print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
            f" \nEnjoy.")
    except:
        import traceback
        traceback.print_exc()
        connection.send(-1)
    else:
        connection.send(index)

# discord events
intents = discord.Intents.all()
from discord import app_commands 
from discord.ui import View, Button

def prompt_weight_analyze(prompt):
    pwpairs = []
    while True:
        j = prompt.find('::')
        if j == -1:
            pwpairs += [(prompt,1)]
            break
        else:
            pwpairs += [(prompt[:j],float(prompt[j+2:].split()[0]))]
            k = prompt[j:].find(' ')
            if k == -1:
                break
            prompt = prompt[j+k+1:]
    return pwpairs

def prompt_analyze(promptArgs):
    i = promptArgs.find(' --')
    if i == -1:
        return (promptArgs, [])
    else:
        prompt = promptArgs[0:i]
        args = promptArgs[i+1:len(promptArgs)].split()
        return prompt, args

def write_log(interaction: discord.Interaction, filename: str, prompt: str, code: str):
    log_file = open(Config.LOG, mode='a')
    log_file.write(
        f'{datetime.now()} {interaction.user.name}#{interaction.user.discriminator} {filename}\n'\
        f'{prompt}\n'\
        f'{code}\n'
    )
    log_file.close()

async def oekaki_body(interaction: discord.Interaction, prompt: str):
    global queuecount
    global ticketcount
    ticketnumber = ticketcount
    ticketcount += 1
    filename=''.join(random.choices(string.ascii_letters + string.digits, k=16))
    user = interaction.user

    try:
        promptT, promptArgs = prompt_analyze(prompt)
        print([promptT] + promptArgs)

        promptArgsDisp = [promptT] + promptArgs
        for i in range(0,len(promptArgsDisp)):
            if promptArgsDisp[i] == '--init-img':
                if i+1 < len(promptArgsDisp):
                    promptArgsDisp[i+1] = f'<{promptArgsDisp[i+1]}>'
        promptDisp = ' '.join(promptArgsDisp)

        args = promptArgs
        opt = parser.parse_args(args)
        print(f'H:{opt.H},W:{opt.W}')
        pws = prompt_weight_analyze(promptT)
        if opt.naive_weight:
            opt.prompt = pws
        else:
            a = 0
            for pw in pws:
                (p,w) = pw
                a += w
            if abs(a) < 0.00000001:
                pws += [('',1)]
                a += 1
            npws = []
            for pw in pws:
                (p,w) = pw
                npws += [(p, w/a)]
            opt.prompt = npws
        print(opt.prompt)
    except:
        await interaction.edit_original_response(content=f'?????????:prompt??????????????????{user.mention}:**{promptDisp}**')
        queuecount += 1
        write_log(interaction, filename, prompt, f'prompt failed q:{queuecount} t:{ticketcount}')
        return

    previousqueuecount = queuecount - 1
    while queuecount < ticketnumber:
        if previousqueuecount < queuecount:
            await interaction.edit_original_response(content=f'??????{ticketnumber-queuecount}??????????????????{user.mention}:**{promptDisp}**\n??????????????????????????????')
            previousqueuecount += 1
        await asyncio.sleep(1)
    await interaction.edit_original_response(content=f'??????????????????{user.mention}:**{promptDisp}**')

    if not opt.init_img:
        pass
    else:
        try:
            headers = { "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0" }
            request = urllib.request.Request(opt.init_img,headers=headers)
            with urllib.request.urlopen(request) as web_file, open(f'{Config.INDIR}{filename}.png', 'wb') as file:
                file.write(web_file.read())
            opt.init_img = f'{Config.INDIR}{filename}.png'
        except urllib.error.URLError as e:
            print(e)
            await interaction.edit_original_response(content=f'???????????????????????????????????????????????????{user.mention}:**{promptDisp}**')
            queuecount += 1
            write_log(interaction, filename, prompt, f'file read failed q:{queuecount} t:{ticketcount}')
            return

    receiver, sender = Pipe()
    g_cuda = torch.Generator(device='cuda')
    seed = (g_cuda.seed() if opt.seed is None else opt.seed) % 4294967295
    calc_proc = Process(target=txt2img if not opt.init_img else img2img, args=(opt, filename, seed, sender))
    calc_proc.start()
    n_imgs = 0
    is_failed = False
    while True:
        if receiver.poll():
            msg = receiver.recv()
            if msg > 0:
                n_imgs = msg
                break
            else:
                is_failed = True
                break
        await asyncio.sleep(1)
    
    if is_failed:
        await interaction.edit_original_response(content=f'????????????????????????{user.mention}:**{promptDisp}**\n?????????????????????????????????????????????')
        queuecount += 1
        write_log(interaction, filename, prompt, f'generation failed q:{queuecount} t:{ticketcount}')
        return

    class ImageSelectButton(Button):
        def __init__(self, label: str, callback_msg, filename: str, row: int):
            super().__init__(label=label, row=row)
            self.filename=filename
            self.callback_message=callback_msg
        async def callback(self, interaction: discord.Interaction):
            await interaction.response.send_message(content=self.callback_message(interaction), file=discord.File(f"{Config.OUTDIR}samples/{self.filename}.png"))
    
    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    view = View(timeout=None)
    row = 0
    index = 0
    index_in_column = 0
    while index < n_imgs:
        view.add_item(ImageSelectButton(label=f"{index}", callback_msg=(lambda interaction, index=index: f"{index}?????????????????????{interaction.user.mention}:**{promptDisp}**"), filename=f'{filename}_{index}', row=row))
        index_in_column += 1
        if (not index_in_column < n_rows):
            index_in_column = 0
            row += 1
        index += 1

    await interaction.edit_original_response(content=f'???????????????{user.mention}:**{promptDisp}** (seed={seed})', attachments=[discord.File(f"{Config.OUTDIR}{filename}.png")], view=view)
    calc_proc.join()
    queuecount += 1
    write_log(interaction, filename, prompt, f'successed q:{queuecount} t:{ticketcount}')



if __name__ == '__main__':
    #client

    guilds = []
    for guildid in Config.GUILD_IDS:
        guilds.append(discord.Object(id=guildid))
    print(guilds)

    class aclient(discord.Client):
        def __init__(self):
            super().__init__(intents=intents)
            self.synced = False 
        async def on_ready(self):
            await self.wait_until_ready()
            if not self.synced:
                # tree.clear_commands(guild = discord.Object(id=))
                for guild in guilds:
                    await tree.sync(guild=guild)
                self.synced = True
            print('login')
            print(discord.__version__)

    client = aclient()
    tree = app_commands.CommandTree(client)

    @tree.command(guilds=guilds, name='oekaki')
    async def oekaki(interaction: discord.Interaction, prompt: str):
        print(f"prompt:**{prompt}**")
        await interaction.response.defer()
        await oekaki_body(interaction, prompt)

    @tree.command(guilds=guilds, name='oshiete')
    async def oshiete(interaction: discord.Interaction):
        content = \
            '**?????????????????????????????????**\n'\
            '??????????????????prompt???`apple --H 128 --W 256`??????????????????????????????\n'\
            '\n'\
            '`--W` ???????????????????????????????????????\n64?????????????????????????????????????????????????????????????????????:512???\n'\
            '`--H` ??????????????????????????????????????????\n64?????????????????????????????????????????????????????????????????????:512???\n'\
            '`--n_samples` ??????????????????????????????????????????????????????\n??????????????????????????????????????????\n????????????????????????????????????????????????????????????:3???\n'\
            '`--n_iter` ???????????????????????????????????????????????????\n????????????????????????????????????????????????????????????:2?????????:4???\n'\
            '`--repeat` ????????????????????????????????????????????????????????????????????????\n'\
            '`--float16` ????????????float16???????????????????????????VRAM??????????????????\n'\
            '`--init-img` ??????????????????URL??????????????????png?????????\n'\
            '`--strength` ??????????????????????????????????????????????????????????????????????????????0~1??????????????????:0.75???\n'\
            '`--seed` ??????????????????????????????\n'\
            f'`--ckpt` ????????????????????????????????????{list(Config.MODEL_DICT.keys())}??????????????????????????????????????????:{list(Config.MODEL_DICT.keys())[0]}???'
        await interaction.response.send_message(content=content)
    client.run(Config.TOKEN)