from typing import Callable, List, Optional, Union

import PIL
import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import \
    prepare_mask_and_masked_image
from utilities import DPMScheduler, Engine, LMSDiscreteScheduler
from polygraphy import cuda


class TRTStableDiffusionInpaintPosePipeline:
    def __init__(
        self,
        scheduler="LMSD",
        device='cuda',
    ):
        # A scheduler to be used in combination with unet to denoise the encoded image latents.
        # This demo uses an adaptation of LMSDiscreteScheduler or DPMScheduler:
        sched_opts = {'num_train_timesteps': 1000, 'beta_start': 0.00085, 'beta_end': 0.012}
        if scheduler == "DPM":
            self.scheduler = DPMScheduler(device=self.device, **sched_opts)
        elif scheduler == "LMSD":
            self.scheduler = LMSDiscreteScheduler(device=self.device, **sched_opts)
        else:
            raise ValueError(f"Scheduler should be either DPM or LMSD")
        self.stream = cuda.Stream()

    def teardown(self):
        for engine in self.engine.values():
            del engine
        self.stream.free()
        del self.stream

    def prepare_mask_latents(
        self, mask, masked_image, pose_inputs, batch_size, height, width, dtype, device, generator
    ):
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
        image: Union[torch.FloatTensor, PIL.Image.Image],
        mask_image: Union[torch.FloatTensor, PIL.Image.Image],
        pose_inputs: Union[torch.FloatTensor, PIL.Image.Image],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs
        self.check_inputs(prompt, height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        if t5_emb is not None:
            if do_classifier_free_guidance:
                empty_t5 = torch.zeros_like(text_embeddings)
                empty_t5[text_embeddings.shape[0]//2:] = t5_emb
            else:
                empty_t5 = t5_emb
            text_embeddings = torch.cat([text_embeddings, empty_t5], dim=1)

        # 4. Preprocess mask and image
        mask, masked_image = prepare_mask_and_masked_image(image, mask_image)

        # 5. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare mask latent variables
        mask, masked_image_latents, pose_inputs = self.prepare_mask_latents(
            mask,
            masked_image,
            pose_inputs,
            batch_size * num_images_per_prompt,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            do_classifier_free_guidance,
        )

        # 8. Check that sizes of mask, masked image and latents match
        num_channels_mask = mask.shape[1]
        num_channels_masked_image = masked_image_latents.shape[1]
        num_channels_pose = pose_inputs.shape[1] if pose_inputs is not None else 0
        if num_channels_latents + num_channels_mask + num_channels_masked_image + num_channels_pose != self.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                f" = {num_channels_latents+num_channels_masked_image+num_channels_mask} + `num_channels_pose`: {num_channels_pose}. Please verify the config of"
                " `pipeline.unet` or your `mask_image` or `image` input."
            )

        # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 10. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                # concat latents, mask, masked_image_latents in the channel dimension
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents, pose_inputs], dim=1)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 11. Post-processing
        image = self.decode_latents(latents)

        # 12. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)

        # 13. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
