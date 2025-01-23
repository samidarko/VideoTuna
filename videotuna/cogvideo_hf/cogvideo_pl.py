import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# from videotuna.base.ddpm3d import DDPM
import pytorch_lightning as pl
import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXTransformer3DModel,
)
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import T5EncoderModel, T5Tokenizer

from videotuna.utils.common_utils import instantiate_from_config


def has_nan(tensor):
    return torch.isnan(tensor).any()


# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class CogVideoXWorkFlow(pl.LightningModule):
    def __init__(
        self,
        first_stage_config,
        cond_stage_config,
        denoiser_config,
        scheduler_config,
        learning_rate: float = 6e-6,
        adapter_config=None,
        logdir=None,  # notice: this is not configured in config.yaml but configured in train.py
    ):
        super().__init__()
        self.logdir = logdir
        self.learning_rate = learning_rate
        # condtion stage use T5 class, which is availale at lvdm.module.encoders.condtion.FrozenT5Embedder
        # but we need to be aware of the model name and tokenizer name
        # here is the same with DDPM
        # print(first_stage_config)
        self.instantiate_first_stage(first_stage_config)
        # max_sequence_length=226
        self.instantiate_cond_stage(cond_stage_config)
        self.vae_scale_factor_spatial = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if hasattr(self, "first_stage_model") and self is not None
            else 8
        )
        self.vae_scale_factor_temporal = (
            self.vae.config.temporal_compression_ratio
            if hasattr(self, "first_stage_model") and self.vae is not None
            else 4
        )

        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )

        # CogVideoX-2b weights are stored in float16
        # CogVideoX-5b weights are stored in bfloat16
        # load_dtype = torch.bfloat16 if "5b" in config.pretrained_model_name_or_path.lower() else torch.float16
        self.model = instantiate_from_config(denoiser_config)
        # self.model = DiffusionWrapper(unet_config, conditioning_key)
        # what I notice is : the most code in DDPM that seems different from there,
        # are most schduler
        self.scheduler = instantiate_from_config(scheduler_config)
        # add adapter config (Support Lora and HRA )
        self.lora_args = []
        if adapter_config is not None:
            self.inject_adapter(adapter_config)

    def inject_adapter(self, adapter_config):
        self.model.requires_grad_(False)
        self.model.enable_gradient_checkpointing()
        print("Injecting lora adapter")
        transformer_adapter_config = instantiate_from_config(adapter_config)
        print(transformer_adapter_config)
        self.model = get_peft_model(self.model, transformer_adapter_config)
        self.model.print_trainable_parameters()

    ## VAE is named as first_stage_model
    ## followed functions are all first stage related.
    def instantiate_first_stage(self, config):
        # import pdb;pdb.set_trace()
        model = instantiate_from_config(config)
        self.vae = model.eval()
        # self.vae.train = disabled_train
        self.vae.requires_grad_(False)

    @torch.no_grad()
    def encode_first_stage(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
        latent_dist = self.vae.encode(x).latent_dist
        return latent_dist

    def _decode_core(self, z, **kwargs):
        z = 1.0 / self.scale_factor * z

        if self.encoder_type == "2d" and z.dim() == 5:
            return self.decode_first_stage_2DAE(z)
        results = self.vae.decode(z, **kwargs)
        return results

    @torch.no_grad()
    def decode_first_stage(self, z, **kwargs):
        return self._decode_core(z, **kwargs)

    def differentiable_decode_first_stage(self, z, **kwargs):
        """same as decode_first_stage but without decorator"""
        return self._decode_core(z, **kwargs)

    ## second stage : text condition and other condtions
    def instantiate_cond_stage(self, config):
        model = instantiate_from_config(config)
        # # in finetune cogvideox don't train as default
        if config.get("freeze", True):
            self.cond_stage_model = model.eval()
            self.cond_stage_model.requires_grad_(False)
        else:
            self.cond_stage_model = model

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, "encode") and callable(
                self.cond_stage_model.encode
            ):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.permute(
            0, 2, 1, 3, 4
        )  # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / self.vae.config.scaling_factor * latents

        frames = self.vae.decode(latents).sample
        return frames

    # Copied from diffusers.pipelines.latte.pipeline_latte.LattePipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        # if callback_on_step_end_tensor_inputs is not None and not all(
        #     k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        # ):
        #     raise ValueError(
        #         f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
        #     )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (
            not isinstance(prompt, str) and not isinstance(prompt, list)
        ):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        text_input_ids=None,
    ):
        device = self.device
        # TODO: fix data type
        dtype = torch.float32
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.cond_stage_model.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = self.cond_stage_model.transformer(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_videos_per_prompt, seq_len, -1
        )

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        # print(">>>> self.encode_prompt",prompt);exit()
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = (
                batch_size * [negative_prompt]
                if isinstance(negative_prompt, str)
                else negative_prompt
            )

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

            return prompt_embeds, negative_prompt_embeds
        return prompt_embeds

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        num_frames,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.permute(
            0, 2, 1, 3, 4
        )  # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / self.vae.config.scaling_factor * latents
        # print(">>>>>>",latents.shape)
        frames = self.vae.decode(latents).sample
        return frames

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def _prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        vae_scale_factor_spatial: int = 8,
        patch_size: int = 2,
        attention_head_dim: int = 64,
        device: Optional[torch.device] = None,
        base_height: int = 480,
        base_width: int = 720,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        grid_height = height // (vae_scale_factor_spatial * patch_size)
        grid_width = width // (vae_scale_factor_spatial * patch_size)
        base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
        base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
        )

        freqs_cos = freqs_cos.to(device=device)
        freqs_sin = freqs_sin.to(device=device)
        return freqs_cos, freqs_sin

    @torch.no_grad()
    def sample(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
    ) -> Union[Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_frames (`int`, defaults to `48`):
                Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal. Generated video will
                contain 1 extra frame because CogVideoX is conditioned with (num_seconds * fps + 1) frames where
                num_seconds is 6 and fps is 4. However, since videos can be saved at any fps, the only condition that
                needs to be satisfied is that of divisibility mentioned above.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `226`):
                Maximum sequence length in encoded prompt. Must be consistent with
                `self.model.config.max_text_seq_length` otherwise may lead to poor results.
        """

        if num_frames > 49:
            raise ValueError(
                "The number of frames must be less than 49 for now due to static positional embeddings. This will be updated in the future to remove this limitation."
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.model.config.sample_size * self.vae_scale_factor_spatial
        width = width or self.model.config.sample_size * self.vae_scale_factor_spatial
        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        # TODO: fix device
        device = "cuda:0"

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents.
        latent_channels = self.model.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(
                height=height,
                width=width,
                num_frames=latents.shape[1],
                vae_scale_factor_spatial=self.vae_scale_factor_spatial,
                patch_size=self.model.config.patch_size,
                attention_head_dim=self.model.config.attention_head_dim,
                device=self.device,
            )
            if self.model.config.use_rotary_positional_embeddings
            else None
        )

        # 8. Denoising loop
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self.interrupt = False
        # for DPM-solver++
        self.model.cuda()
        old_pred_original_sample = None
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # print(i,latent_model_input.max(),latent_model_input.min(),has_nan(latent_model_input))
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])
            # print(i,num_inference_steps)
            # predict noise model_output
            noise_pred = self.model(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                image_rotary_emb=image_rotary_emb,
                # attention_kwargs=attention_kwargs, # None
                return_dict=False,
            )[0]
            noise_pred = noise_pred.float()

            # perform guidance
            if use_dynamic_cfg:
                self._guidance_scale = 1 + guidance_scale * (
                    (
                        1
                        - math.cos(
                            math.pi
                            * ((num_inference_steps - t.item()) / num_inference_steps)
                            ** 5.0
                        )
                    )
                    / 2
                )
            else:
                self.guidance_scale = guidance_scale
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]
            else:
                latents, old_pred_original_sample = self.scheduler.step(
                    noise_pred,
                    old_pred_original_sample,
                    t,
                    timesteps[i - 1] if i > 0 else None,
                    latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )
            latents = latents.to(prompt_embeds.dtype)

            # call the callback, if provided
            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop(
                    "negative_prompt_embeds", negative_prompt_embeds
                )
            # if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
            #     progress_bar.update()

        self.model.cpu()
        if not output_type == "latent":
            video = self.decode_latents(latents)
            # video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # # Offload all models
        # self.maybe_free_model_hooks()
        video = video[None, ...]
        video = video.cpu()
        torch.cuda.empty_cache()
        return video

    # trianing specific functions
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate,
        )
        return optimizer

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        new_satate_dict = checkpoint["state_dict"]
        new_satate_dict = {k: v for k, v in new_satate_dict.items() if "lora" in k}
        checkpoint["state_dict"] = new_satate_dict
        return checkpoint

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        pass

    def encode_video(self, video):
        video = video.to(self.device, dtype=self.dtype).unsqueeze(0)
        # video = video.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
        latent_dist = self.vae.encode(video).latent_dist
        return latent_dist

    def get_batch_input(self, batch):
        """
        Prepare model batch inputs
        """
        # equal to collate_fn
        # the resonable video latents range is [-5,5], approximately.
        # videos
        videos = [self.encode_video(video) for video in batch["video"]]
        videos = [video.sample() * self.vae.config.scaling_factor for video in videos]
        videos = torch.cat(videos, dim=0)
        videos = videos.to(memory_format=torch.contiguous_format).float()
        # prompt
        prompts = [item for item in batch["caption"]]
        return {
            "videos": videos,
            "prompts": prompts,
        }

    def training_step(self, batch, batch_idx):
        batch = self.get_batch_input(batch)
        model_input = (
            batch["videos"].permute(0, 2, 1, 3, 4).to(dtype=self.dtype)
        )  # [B, F, C, H, W]
        prompts = batch["prompts"]

        max_sequence_length = 226
        with torch.no_grad():
            prompt_embeds = self.encode_prompt(
                prompts,
                do_classifier_free_guidance=False,  # set to false for train
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
                device=self.device,
                dtype=self.dtype,
            )

        batch_size, num_frames, num_channels, height, width = model_input.shape

        # Sample noise that will be added to the latents
        noise = torch.randn_like(model_input)

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
        )
        timesteps = timesteps.long()

        # Prepare rotary embeds
        image_rotary_emb = (
            # in the first place, we assume this function is the same during inference and train.
            self._prepare_rotary_positional_embeddings(
                height=height * self.vae_scale_factor_spatial,
                width=width * self.vae_scale_factor_spatial,
                num_frames=num_frames,
                vae_scale_factor_spatial=self.vae_scale_factor_spatial,
                patch_size=self.model.config.patch_size,
                attention_head_dim=self.model.config.attention_head_dim,
                device=self.device,
            )
            if self.model.config.use_rotary_positional_embeddings
            else None
        )

        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_model_input = self.scheduler.add_noise(model_input, noise, timesteps)
        model_output = self.model(
            hidden_states=noisy_model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timesteps,
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
        )[0]
        model_pred = self.scheduler.get_velocity(
            model_output, noisy_model_input, timesteps
        )

        alphas_cumprod = self.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(model_pred.shape):
            weights = weights.unsqueeze(-1)

        target = model_input
        # TODO: inherent loss computation from base class.
        loss = torch.mean(
            (weights * (model_pred - target) ** 2).reshape(batch_size, -1), dim=1
        )
        loss = loss.mean()
        return loss


if __name__ == "__main__":
    # test text encoder
    prompt = ["Elon mask is talking"]
    device = "cuda"
    dtype = "float32"
    tokenizer = T5Tokenizer.from_pretrained("THUDM/CogVideoX-2b", subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        "THUDM/CogVideoX-2b", subfolder="text_encoder"
    ).to(device)
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=226,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    with torch.no_grad():
        prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    print(has_nan(prompt_embeds))
