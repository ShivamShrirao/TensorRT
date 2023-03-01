import argparse
import inspect
import os
import time
from collections import OrderedDict
from copy import copy
from typing import List, Optional, Union

import numpy as np
import tensorrt as trt
import torch
from diffusers import EulerAncestralDiscreteScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import \
    prepare_mask_and_masked_image
from PIL import Image
from polygraphy import cuda
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import (CreateConfig, Profile, engine_from_bytes,
                                    engine_from_network,
                                    network_from_onnx_path, save_engine)
from polygraphy.backend.trt import util as trt_util
from transformers import CLIPTokenizer

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def parseArgs():
    parser = argparse.ArgumentParser(description="Options for Stable Diffusion Demo")
    # Stable Diffusion configuration
    parser.add_argument('prompt', nargs='*', help="Text prompt(s) to guide image generation")
    parser.add_argument('--negative-prompt', nargs='*',
                        default=[''], help="The negative prompt(s) to guide the image generation.")
    parser.add_argument('--repeat-prompt', type=int, default=1,
                        choices=[1, 2, 4, 8, 16], help="Number of times to repeat the prompt (batch size multiplier)")
    parser.add_argument('--height', type=int, default=768, help="Height of image to generate (must be multiple of 8)")
    parser.add_argument('--width', type=int, default=768, help="Height of image to generate (must be multiple of 8)")
    parser.add_argument('--num_inference_steps', type=int, default=24, help="Number of denoising steps")
    parser.add_argument('--model_name_or_path', type=str, default="stabilityai/stable-diffusion-2-1-base",
                        help="HuggingFace model name or path to pretrained model")

    # TensorRT engine build
    parser.add_argument('--engine-dir', default='engine', help="Output directory for TensorRT engine")
    parser.add_argument('--seed', type=int, default=None, help="Seed for random generator to get consistent results")

    parser.add_argument('--output-dir', default='output', help="Output directory for logs and image artifacts")
    parser.add_argument('--hf-token', type=str, help="HuggingFace API access token for downloading model checkpoints")
    parser.add_argument('-v', '--verbose', action='store_true', help="Show verbose output")
    return parser.parse_args()


class Engine():
    def __init__(
        self,
        engine_path,
        text_maxlen=77,
        embedding_dim=1024,
    ):
        self.text_maxlen = text_maxlen
        self.embedding_dim = embedding_dim
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.engine_bytes = bytes_from_path(self.engine_path)

    def __del__(self):
        self.deactivate()

    def deactivate(self):
        [buf.free() for buf in self.buffers.values() if isinstance(buf, cuda.DeviceArray)]
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()

    def activate(self):
        self.engine = engine_from_bytes(self.engine_bytes)
        self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device='cuda'):
        for idx in range(trt_util.get_bindings_per_profile(self.engine)):
            binding = self.engine[idx]
            if shape_dict and binding in shape_dict:
                shape = shape_dict[binding]
            else:
                shape = self.engine.get_binding_shape(binding)
            dtype = trt_util.np_dtype_from_trt(self.engine.get_binding_dtype(binding))
            if self.engine.binding_is_input(binding):
                self.context.set_binding_shape(idx, shape)
            # Workaround to convert np dtype to torch
            np_type_tensor = np.empty(shape=[], dtype=dtype)
            torch_type_tensor = torch.from_numpy(np_type_tensor)
            tensor = torch.empty(tuple(shape), dtype=torch_type_tensor.dtype, device=device)
            self.tensors[binding] = tensor
            self.buffers[binding] = cuda.DeviceView(ptr=tensor.data_ptr(), shape=shape, dtype=dtype)

    def infer(self, feed_dict, stream):
        start_binding, end_binding = trt_util.get_active_profile_bindings(self.context)
        # shallow copy of ordered dict
        device_buffers = copy(self.buffers)
        for name, buf in feed_dict.items():
            assert isinstance(buf, cuda.DeviceView)
            device_buffers[name] = buf
        bindings = [0] * start_binding + [buf.ptr for buf in device_buffers.values()]
        noerror = self.context.execute_async_v2(bindings=bindings, stream_handle=stream.ptr)
        if not noerror:
            raise ValueError(f"ERROR: inference failed.")

        return self.tensors

    def build(self, onnx_path, fp16, input_profile=None, enable_preview=False):
        print(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
        p = Profile()
        if input_profile:
            for name, dims in input_profile.items():
                assert len(dims) == 3
                p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        preview_features = []
        if enable_preview:
            trt_version = [int(i) for i in trt.__version__.split(".")]
            # FASTER_DYNAMIC_SHAPES_0805 should only be used for TRT 8.5.1 or above.
            if trt_version[0] > 8 or \
                    (trt_version[0] == 8 and (trt_version[1] > 5 or (trt_version[1] == 5 and trt_version[2] >= 1))):
                preview_features = [trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805]

        engine = engine_from_network(network_from_onnx_path(onnx_path), config=CreateConfig(fp16=fp16, profiles=[p],
                                                                                            preview_features=preview_features))
        save_engine(engine, path=self.engine_path)


class UNet(Engine):
    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = image_height // 8, image_width // 8
        return {
            'sample': (2*batch_size, 13, latent_height, latent_width),
            'encoder_hidden_states': (2*batch_size, self.text_maxlen, self.embedding_dim),
            'latent': (2*batch_size, 4, latent_height, latent_width)
        }


class CLIP(Engine):
    def get_shape_dict(self, batch_size, image_height, image_width):
        return {
            'input_ids': (batch_size, self.text_maxlen),
            'text_embeddings': (batch_size, self.text_maxlen, self.embedding_dim)
        }


class VAEEncode(Engine):
    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = image_height // 8, image_width // 8
        return {
            'images': (batch_size, 3, image_height, image_width),
            'latent': (batch_size, 4, latent_height, latent_width)
        }


class VAEDecode(Engine):
    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = image_height // 8, image_width // 8
        return {
            'latent': (batch_size, 4, latent_height, latent_width),
            'images': (batch_size, 3, image_height, image_width)
        }


class TRTStableDiffusionInpaintPosePipeline:
    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        scheduler: EulerAncestralDiscreteScheduler,
        engine_dir: str ='engine',
        device: str ='cuda',
    ):
        self.device = device
        self.vae_scale_factor = 8
        self.latent_channels = 4
        self.engine_dir = engine_dir

        self.tokenizer = tokenizer
        self.scheduler = scheduler

        self.models = {
            'clip': CLIP(f'{engine_dir}/clip.plan'),
            'unet_fp16': UNet(f'{engine_dir}/unet_fp16.plan'),
            'vae_encode': VAEEncode(f'{engine_dir}/vae_encode.plan'),
            'vae_decode': VAEDecode(f'{engine_dir}/vae_decode.plan'),
        }
        self.unet_model_key = 'unet_fp16'
        self.clip_model_key = 'clip'
        self.stream = cuda.Stream()
    
    def loadEngines(self, keys: List[str] = None):
        if keys is None:
            keys = self.models.keys()
        for key in keys:
            start_time = time.time()
            self.models[key].activate()
            print(f"Loaded {key} engine in {time.time() - start_time:.3f} seconds")

    def unloadEngines(self, keys: List[str] = None):
        if keys is None:
            keys = self.models.keys()
        for key in keys:
            start_time = time.time()
            self.models[key].deactivate()
            print(f"Unloaded {key} engine in {time.time() - start_time:.3f} seconds")
    
    def allocateBuffers(self, batch_size: int, height: int, width: int, keys: List[str] = None):
        if keys is None:
            keys = self.models.keys()
        for key in keys:
            start_time = time.time()
            engine = self.models[key]
            engine.allocate_buffers(
                shape_dict=engine.get_shape_dict(batch_size, height, width),
                device=self.device
            )
            print(f"Allocated buffers for {key} engine in {time.time() - start_time:.3f} seconds")

    def teardown(self):
        for engine in self.models.values():
            del engine
        self.stream.free()
        del self.stream

    def runEngine(self, model_name, feed_dict):
        engine = self.models[model_name]
        return engine.infer(feed_dict, self.stream)

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.type(torch.int32).to(device)

            text_input_ids_inp = cuda.DeviceView(ptr=text_input_ids.data_ptr(),
                                                 shape=text_input_ids.shape, dtype=np.int32)
            prompt_embeds = self.runEngine(self.clip_model_key, {"input_ids": text_input_ids_inp})['text_embeddings']

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input_ids = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.type(torch.int32).to(device)

            uncond_input_ids_inp = cuda.DeviceView(ptr=uncond_input_ids.data_ptr(),
                                                   shape=uncond_input_ids.shape, dtype=np.int32)
            negative_prompt_embeds = self.runEngine(self.clip_model_key, {"input_ids": uncond_input_ids_inp})['text_embeddings']

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        prompt_embeds = prompt_embeds.to(dtype=torch.float16)
        return prompt_embeds

    def prepare_mask_latents(
            self, mask, masked_image, pose_inputs, batch_size, height, width, dtype, device, do_classifier_free_guidance):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // 8, width // 8), mode='area'
        )
        pose_inputs = torch.nn.functional.interpolate(
            pose_inputs, size=(height // 8, width // 8)
        )

        mask = mask.to(device=device, dtype=dtype)
        pose_inputs = pose_inputs.to(device=device, dtype=dtype)
        masked_image = masked_image.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        sample_inp = cuda.DeviceView(ptr=masked_image.data_ptr(), shape=masked_image.shape, dtype=np.float32)
        masked_image_latents = self.runEngine('vae_encode', {"images": sample_inp})['latent']
        masked_image_latents = 0.18215 * masked_image_latents

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)
        if pose_inputs.shape[0] < batch_size:
            pose_inputs = pose_inputs.repeat(batch_size // pose_inputs.shape[0], 1, 1, 1)

        if do_classifier_free_guidance:
            mask = torch.cat([mask] * 2)
            pose_inputs = torch.cat([pose_inputs] * 2)
            masked_image_latents = torch.cat([masked_image_latents] * 2)

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return mask, masked_image_latents, pose_inputs

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: Union[torch.FloatTensor, Image.Image],
        mask_image: Union[torch.FloatTensor, Image.Image],
        pose_inputs: Union[torch.FloatTensor, Image.Image],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ):
        # 0. Default height and width to unet
        height = height
        width = width

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt, self.device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # 4. Preprocess mask and image
        mask, masked_image = prepare_mask_and_masked_image(image, mask_image)

        # 5. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # 6. Prepare latent variables
        shape = (batch_size, self.latent_channels, height // self.vae_scale_factor, width // self.vae_scale_factor)
        latents = torch.randn(shape, generator=generator, device=self.device, dtype=torch.float32)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # 7. Prepare mask latent variables
        mask, masked_image_latents, pose_inputs = self.prepare_mask_latents(
            mask,
            masked_image,
            pose_inputs,
            batch_size * num_images_per_prompt,
            height,
            width,
            latents.dtype,
            self.device,
            do_classifier_free_guidance
        )

        # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 10. Denoising loop
        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            # concat latents, mask, masked_image_latents in the channel dimension
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, i)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents, pose_inputs], dim=1)

            dtype = np.float16 if text_embeddings.dtype == torch.float16 else np.float32
            if t.dtype != torch.float32:
                timestep_float = t.float()
            else:
                timestep_float = t
            sample_inp = cuda.DeviceView(ptr=latent_model_input.data_ptr(),
                                         shape=latent_model_input.shape, dtype=np.float32)
            timestep_inp = cuda.DeviceView(ptr=timestep_float.data_ptr(), shape=timestep_float.shape, dtype=np.float32)
            embeddings_inp = cuda.DeviceView(ptr=text_embeddings.data_ptr(), shape=text_embeddings.shape, dtype=dtype)
            noise_pred = self.runEngine(
                self.unet_model_key, {"sample": sample_inp, "timestep": timestep_inp, "encoder_hidden_states": embeddings_inp})['latent']

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            # latents = self.scheduler.step(noise_pred, latents, i, t)

        # 11. Post-processing
        latents = 1. / 0.18215 * latents
        sample_inp = cuda.DeviceView(ptr=latents.data_ptr(), shape=latents.shape, dtype=np.float32)
        image = self.runEngine('vae_decode', {"latent": sample_inp})['images']
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        return image


if __name__ == "__main__":
    args = parseArgs()

    # Process prompt
    prompt = args.prompt * args.repeat_prompt
    if len(args.negative_prompt) == 1:
        negative_prompt = args.negative_prompt * len(prompt)
    else:
        negative_prompt = args.negative_prompt

    tokenizer = CLIPTokenizer.from_pretrained(
        args.model_name_or_path,
        subfolder="tokenizer",
    )
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
        args.model_name_or_path,
        subfolder="scheduler",
    )
    # sched_opts = {'num_train_timesteps': 1000, 'beta_start': 0.00085, 'beta_end': 0.012}
    # scheduler = DPMScheduler(
    #     device='cuda', **sched_opts
    # )
    # scheduler.set_timesteps(args.num_inference_steps)
    # # Pre-compute latent input scales and linear multistep coefficients
    # scheduler.configure()

    pipeline = TRTStableDiffusionInpaintPosePipeline(
        tokenizer=tokenizer,
        scheduler=scheduler,
    )

    pipeline.models['unet_fp16_2'] = UNet(f'{args.engine_dir}/unet_fp16.plan')
    pipeline.models['clip_2'] = CLIP(f'{args.engine_dir}/clip.plan')

    pipeline.unet_model_key = 'unet_fp16'
    pipeline.clip_model_key = 'clip'

    batch_size = len(prompt)
    pipeline.loadEngines(['vae_encode', 'vae_decode'])
    pipeline.allocateBuffers(batch_size, args.height, args.width, ['vae_encode', 'vae_decode'])

    generator = None
    if args.seed is not None:
        generator = torch.Generator().manual_seed(args.seed)

    image = Image.new("RGB", (args.width, args.height), color=(0, 0, 0))
    mask_image = Image.new("L", (args.width, args.height), color=255)
    pose_inputs = torch.zeros((1, 4, args.height, args.width), dtype=torch.float32)


    pipeline.loadEngines(['clip', 'unet_fp16'])
    pipeline.allocateBuffers(batch_size, args.height, args.width, ['clip', 'unet_fp16'])
    for i in range(2):
        start_time = time.time()
        images = pipeline(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            pose_inputs=pose_inputs,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=7.5,
            negative_prompt=negative_prompt,
            generator=generator,
        )
    print("Inference time: ", time.time() - start_time)
    Image.fromarray((255 * images[0]).astype(np.uint8)).save("output/out.png")
    pipeline.unloadEngines(['clip', 'unet_fp16'])


    pipeline.loadEngines(['clip_2', 'unet_fp16_2'])
    pipeline.allocateBuffers(batch_size, args.height, args.width, ['clip_2', 'unet_fp16_2'])
    pipeline.clip_model_key = 'clip_2'
    pipeline.unet_model_key = 'unet_fp16_2'
    for i in range(2):
        start_time = time.time()
        images = pipeline(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            pose_inputs=pose_inputs,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=7.5,
            negative_prompt=negative_prompt,
            generator=generator,
        )
    print("Inference time: ", time.time() - start_time)

    Image.fromarray((255 * images[0]).astype(np.uint8)).save("output/out1.png")
    pipeline.teardown()
