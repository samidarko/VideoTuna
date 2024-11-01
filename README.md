<p align="center" width="50%">
<img src="https://github.com/user-attachments/assets/38efb5bc-723e-4012-aebd-f55723c593fb" alt="VideoTuna" style="width: 75%; min-width: 450px; display: block; margin: auto; background-color: transparent;">
</p>

# VideoTuna

![Version](https://img.shields.io/badge/version-0.1.0-blue) ![visitors](https://visitor-badge.laobi.icu/badge?page_id=VideoVerses.VideoTuna&left_color=green&right_color=red) [![GitHub](https://img.shields.io/github/stars/VideoVerses/VideoTuna?style=social)](https://github.com/VideoVerses/VideoTuna) [![](https://dcbadge.limes.pink/api/server/AammaaR2?style=flat)](https://discord.gg/AammaaR2) <a href='assets/wechat_group.png'><img src='https://badges.aleen42.com/src/wechat.svg'></a>



🤗🤗🤗 Videotuna is a useful codebase for text-to-video applications.   
🌟 VideoTuna is the first repo that integrates multiple AI video generation models for text-to-video, image-to-video, text-to-image generation (to the best of our knowledge).   
🌟 VideoTuna is the first repo that provides comprehensive pipelines in video generation, including pre-training, continuous training, post-training (alignment), and fine-tuning (to the best of our knowledge).   
🌟 The models of VideoTuna include both U-Net and DiT architectures for visual generation tasks.  
🌟 A new 3D video VAE, and a controllable facial video generation model will be released soon.  


## Features
🌟 **All-in-one framework:** Inference and fine-tune up-to-date video generation models.  
🌟 **Pre-training:** Build your own foundational text-to-video model.  
🌟 **Continuous training:** Keep improving your model with new data.  
🌟 **Domain-specific fine-tuning:** Adapt models to your specific scenario.  
🌟 **Concept-specific fine-tuning:** Teach your models with unique concepts.  
🌟 **Enhanced language understanding:** Improve model comprehension through continuous training.  
🌟 **Post-processing:** Enhance the videos with video-to-video enhancement model.  
🌟 **Post-training/Human preference alignment:** Post-training with RLHF for more attractive results.  


## 🔆 Updates
- [2024-11-01] We make the VideoTuna V0.1.0 public!


## Demo
### 3D Video VAE
The 3D video VAE from VideoTuna can accurately compress and reconstruct the input videos with fine details.

<table class="center">
  <tr>
    <td><a href="https://github.com/user-attachments/assets/51471c42-8a38-4f02-b29b-e34a5279753a"><img src="https://github.com/user-attachments/assets/51471c42-8a38-4f02-b29b-e34a5279753a" width="320"></a></td>
    <td><a href="https://github.com/user-attachments/assets/383f2120-5fed-4d9f-82d8-3de130d6bd65"><img src="https://github.com/user-attachments/assets/383f2120-5fed-4d9f-82d8-3de130d6bd65" width="320"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;" width="320">Groud Truth</td>
    <td style="text-align:center;" width="320">Reconstruction</td>
  </tr>
  <tr>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/vae/gtview9.gif"><img src="assets/demos/vae/fkview9.gif" width="320"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/vae/fkview9.gif"><img src="assets/demos/vae/fkview9.gif" width="320"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;" width="320">Ground Truth</td>
    <td style="text-align:center;" width="320">Reconstruction</td>
  </tr>
  <tr>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/vae/gtface.gif"><img src="assets/demos/vae/gtface.gif" width="320"></a></td>
    <td><a href="https://github.com/VideoVerses/VideoTuna/blob/main/assets/demos/vae/fkface.gif"><img src="assets/demos/vae/fkface.gif" width="320"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;" width="320">Ground Truth</td>
    <td style="text-align:center;" width="320">Reconstruction</td>
  </tr>

  <tr>
    <td><a href="https://github.com/user-attachments/assets/a0ffc2ca-c3e2-485f-b0ea-ead0d733cc8b"><img src="https://github.com/user-attachments/assets/a0ffc2ca-c3e2-485f-b0ea-ead0d733cc8b" width="320"></a></td>
    <td><a href="https://github.com/user-attachments/assets/1465ac70-caa9-42c7-874b-b01e13a78efb"><img src="https://github.com/user-attachments/assets/1465ac70-caa9-42c7-874b-b01e13a78efb" width="320"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;" width="320">Ground Truth</td>
    <td style="text-align:center;" width="320">Reconstruction</td>
  </tr>
  <tr>
    <td><a href="https://github.com/user-attachments/assets/48e2eb49-265b-4eaf-b730-48fa4d7e5bfd"><img src="https://github.com/user-attachments/assets/48e2eb49-265b-4eaf-b730-48fa4d7e5bfd" width="320"></a></td>
    <td><a href="https://github.com/user-attachments/assets/24c893c5-865e-4af4-b003-17bda2ba4f59"><img src="https://github.com/user-attachments/assets/24c893c5-865e-4af4-b003-17bda2ba4f59" width="320"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;" width="320">Ground Truth</td>
    <td style="text-align:center;" width="320">Reconstruction</td>
  </tr>
  <tr>
    <td><a href="https://github.com/user-attachments/assets/c18ed80f-3650-43a7-8438-7914de7e39ab"><img src="https://github.com/user-attachments/assets/c18ed80f-3650-43a7-8438-7914de7e39ab" width="320"></a></td>
    <td><a href="https://github.com/user-attachments/assets/89d38004-021b-4a4d-ab83-5627474f8928"><img src="https://github.com/user-attachments/assets/89d38004-021b-4a4d-ab83-5627474f8928" width="320"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;" width="320">Ground Truth</td>
    <td style="text-align:center;" width="320">Reconstruction</td>
  </tr>
</table>

### Face domain

<table class="center">
  <tr>
    <td><img src="https://github.com/user-attachments/assets/a1562c70-d97c-4324-bb11-47db2b83f443" width="240" alt="Image 1"></td>
    <td><img src="https://github.com/user-attachments/assets/3196810b-48d7-4024-b687-df2009774631" width="240" alt="Image 2"></td>
    <td><img src="https://github.com/user-attachments/assets/4e873f4c-ca56-4549-aaa1-ef24032ae96b" width="240" alt="Image 3"></td>
  </tr>
  <tr>
    <td style="text-align: center;">Input 1</td>
    <td style="text-align: center;">Input 2</td>
    <td style="text-align: center;">Input 3</td>
  </tr>
</table>

<table class="center">
  <tr>
    <td><a href="https://github.com/user-attachments/assets/972dde7a-fa88-479a-a47e-71d3650b1826"><img src="https://github.com/user-attachments/assets/972dde7a-fa88-479a-a47e-71d3650b1826" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/3c216090-9ad1-4911-b990-179b45314d3e"><img src="https://github.com/user-attachments/assets/3c216090-9ad1-4911-b990-179b45314d3e" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/2e2fb78e-2f39-47bd-acaf-3cfbce83b162"><img src="https://github.com/user-attachments/assets/2e2fb78e-2f39-47bd-acaf-3cfbce83b162" width="240"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;">Emotion: Anger</td>
    <td style="text-align:center;">Emotion: Disgust</td>
    <td style="text-align:center;">Emotion: Fear</td>
  </tr>
  <tr>
    <td><a href="https://github.com/user-attachments/assets/f2f55021-4e0d-43a7-9f57-3c94b772f573"><img src="https://github.com/user-attachments/assets/f2f55021-4e0d-43a7-9f57-3c94b772f573" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/600a2f6c-7a8f-4304-bdc3-5f0d65d4fb83"><img src="https://github.com/user-attachments/assets/600a2f6c-7a8f-4304-bdc3-5f0d65d4fb83" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/8ad7c7d8-6492-4435-9436-168f90429be3"><img src="https://github.com/user-attachments/assets/8ad7c7d8-6492-4435-9436-168f90429be3" width="240"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;">Emotion: Happy</td>
    <td style="text-align:center;">Emotion: Sad</td>
    <td style="text-align:center;">Emotion: Surprise</td>
  </tr>
</table>

<table class="center">
  <tr>
    <td><a href="https://github.com/user-attachments/assets/8ba84071-1978-4245-84b3-3a6fc3c9fa5a"><img src="https://github.com/user-attachments/assets/8ba84071-1978-4245-84b3-3a6fc3c9fa5a" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/d180c358-bdff-40b4-aa5a-fa4ec73d80b6"><img src="https://github.com/user-attachments/assets/d180c358-bdff-40b4-aa5a-fa4ec73d80b6" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/37004c20-3d0d-4cff-8b4a-f8a4d184de51"><img src="https://github.com/user-attachments/assets/37004c20-3d0d-4cff-8b4a-f8a4d184de51" width="240"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;">Emotion: Anger</td>
    <td style="text-align:center;">Emotion: Disgust</td>
    <td style="text-align:center;">Emotion: Fear</td>
  </tr>
  <tr>
    <td><a href="https://github.com/user-attachments/assets/025fe090-7d53-4a12-9498-1c814a0ee768"><img src="https://github.com/user-attachments/assets/025fe090-7d53-4a12-9498-1c814a0ee768" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/e8ddf3d1-57ea-4545-a004-66554c19f27b"><img src="https://github.com/user-attachments/assets/e8ddf3d1-57ea-4545-a004-66554c19f27b" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/519b3c87-baa6-408b-b3a5-a95eece9e19e"><img src="https://github.com/user-attachments/assets/519b3c87-baa6-408b-b3a5-a95eece9e19e" width="240"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;">Emotion: Happy</td>
    <td style="text-align:center;">Emotion: Sad</td>
    <td style="text-align:center;">Emotion: Surprise</td>
  </tr>
</table>

<table class="center">
  <tr>
    <td><a href="https://github.com/user-attachments/assets/f55a2b6c-ce10-4716-a001-f747c0da17a4"><img src="https://github.com/user-attachments/assets/f55a2b6c-ce10-4716-a001-f747c0da17a4" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/bb3620af-322d-43bc-9060-c7ce9fc32672"><img src="https://github.com/user-attachments/assets/bb3620af-322d-43bc-9060-c7ce9fc32672" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/46a39738-89c2-43fe-9f98-f0ac0d26e39b"><img src="https://github.com/user-attachments/assets/46a39738-89c2-43fe-9f98-f0ac0d26e39b" width="240"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;">Emotion: Anger</td>
    <td style="text-align:center;">Emotion: Disgust</td>
    <td style="text-align:center;">Emotion: Fear</td>
  </tr>
  <tr>
    <td><a href="https://github.com/user-attachments/assets/2d3d6e0d-2034-4341-8ca3-42e0cda2704f"><img src="https://github.com/user-attachments/assets/2d3d6e0d-2034-4341-8ca3-42e0cda2704f" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/331f23e1-f441-46f7-98a6-b25a684780f3"><img src="https://github.com/user-attachments/assets/331f23e1-f441-46f7-98a6-b25a684780f3" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/542f8535-634a-4f82-ae2a-39f988c6bc55"><img src="https://github.com/user-attachments/assets/542f8535-634a-4f82-ae2a-39f988c6bc55" width="240"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;">Emotion: Happy</td>
    <td style="text-align:center;">Emotion: Sad</td>
    <td style="text-align:center;">Emotion: Surprise</td>
  </tr>
</table>


### Storytelling

<table class="center">
  <tr>
    <td><a href="https://github.com/user-attachments/assets/27aee539-f2bf-467a-8da5-22f506713aa0"><img src="https://github.com/user-attachments/assets/27aee539-f2bf-467a-8da5-22f506713aa0" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/fef5b694-6e1f-42f6-a5a1-7f0b856f3678"><img src="https://github.com/user-attachments/assets/fef5b694-6e1f-42f6-a5a1-7f0b856f3678" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/91408eb8-264d-4d3f-9098-0bfe06022467"><img src="https://github.com/user-attachments/assets/91408eb8-264d-4d3f-9098-0bfe06022467" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/0b822858-da6f-4e6f-82dd-5f243e2feccc"><img src="https://github.com/user-attachments/assets/0b822858-da6f-4e6f-82dd-5f243e2feccc" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/75f69de2-9e5c-48d7-ae55-b28084772836"><img src="https://github.com/user-attachments/assets/75f69de2-9e5c-48d7-ae55-b28084772836" width="240"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;">The picture shows a cozy room with a little girl telling her travel story to her teddybear beside the bed.</td>
    <td style="text-align:center;">As night falls, teddybear sits by the window, his eyes sparkling with longing for the distant place</td>
    <td style="text-align:center;">Teddybear was in a corner of the room, making a small backpack out of old cloth strips, with a map, a compass and dry food next to it.</td>
    <td style="text-align:center;">The first rays of sunlight in the morning came through the window, and teddybear quietly opened the door and embarked on his adventure.</td>
    <td style="text-align:center;">In the forest, the sun shines through the treetops, and teddybear moves among various animals and communicates with them.</td>
  </tr>
  <tr>
    <td><a href="https://github.com/user-attachments/assets/3ae06dbf-f41f-4e7f-b384-fca3abf2c0aa"><img src="https://github.com/user-attachments/assets/3ae06dbf-f41f-4e7f-b384-fca3abf2c0aa" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/09a2fdb9-3e84-40a5-a729-075876b10412"><img src="https://github.com/user-attachments/assets/09a2fdb9-3e84-40a5-a729-075876b10412" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/a382aa30-895d-4476-8f86-b668fe153c16"><img src="https://github.com/user-attachments/assets/a382aa30-895d-4476-8f86-b668fe153c16" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/c906b20e-1576-4a96-9c77-b67a940dcce8"><img src="https://github.com/user-attachments/assets/c906b20e-1576-4a96-9c77-b67a940dcce8" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/04f553d2-8ac5-4085-8c1b-59bddf9deb41"><img src="https://github.com/user-attachments/assets/04f553d2-8ac5-4085-8c1b-59bddf9deb41" width="240"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;">Teddybear leaves his mark on the edge of a clear lake, surrounded by exotic flowers, and the picture is full of mystery and exploration.</td>
    <td style="text-align:center;">Teddybear climbs the rugged mountain road, the weather is changeable, but he is determined.</td>
    <td style="text-align:center;">The picture switches to the top of the mountain, where teddybear stands in the glow of the sunrise, with a magnificent mountain view in the background.</td>
    <td style="text-align:center;">On the way home, teddybear helps a wounded bird, the picture is warm and touching.</td>
    <td style="text-align:center;">Teddybear sits by the little girl's bed and tells her his adventure story, and the little girl is fascinated.</td>
  </tr>
</table>


<table class="center">
  <tr>
    <td><a href="https://github.com/user-attachments/assets/14c5f6f1-7830-46fc-9ebd-8611f339d8ab"><img src="https://github.com/user-attachments/assets/14c5f6f1-7830-46fc-9ebd-8611f339d8ab" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/705eb5c4-b084-4752-a47b-ab20d206b9ee"><img src="https://github.com/user-attachments/assets/705eb5c4-b084-4752-a47b-ab20d206b9ee" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/39a814b6-8692-41f5-819d-eecfd6085b03"><img src="https://github.com/user-attachments/assets/39a814b6-8692-41f5-819d-eecfd6085b03" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/e2b7297d-d0bb-482e-9c16-f01f9a56fbb0"><img src="https://github.com/user-attachments/assets/e2b7297d-d0bb-482e-9c16-f01f9a56fbb0" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/3afc9aa8-92f5-4e27-bfed-950300645748"><img src="https://github.com/user-attachments/assets/3afc9aa8-92f5-4e27-bfed-950300645748" width="240"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;">The scene shows a peaceful village, with moonlight shining on the roofs and streets, creating a peaceful atmosphere.</td>
    <td style="text-align:center;">cat sits by the window, her eyes twinkling in the night, reflecting her special connection with the moon and stars.</td>
    <td style="text-align:center;">Villagers gather in the center of the village for the annual Moon Festival celebration, with lanterns and colored lights adorning the night sky.</td>
    <td style="text-align:center;">cat feels the call of the moon, and her beard trembles with the excitement in her heart.</td>
    <td style="text-align:center;">cat quietly leaves her home in the night and embarks on a path illuminated by the silver moonlight.</td>
  </tr>
  <tr>
    <td><a href="https://github.com/user-attachments/assets/b055930e-4f97-4872-bc07-2e9cb5641d7d"><img src="https://github.com/user-attachments/assets/b055930e-4f97-4872-bc07-2e9cb5641d7d" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/ece35515-295d-4655-a5c5-cca95fd11e92"><img src="https://github.com/user-attachments/assets/ece35515-295d-4655-a5c5-cca95fd11e92" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/804e32f1-75ae-4a0b-b3c0-1d13c6e2d987"><img src="https://github.com/user-attachments/assets/804e32f1-75ae-4a0b-b3c0-1d13c6e2d987" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/e09d8134-7991-4585-a09a-b00283ab6a56"><img src="https://github.com/user-attachments/assets/e09d8134-7991-4585-a09a-b00283ab6a56" width="240"></a></td>
    <td><a href="https://github.com/user-attachments/assets/4c722b97-79d9-4281-8451-b31eb3393c3a"><img src="https://github.com/user-attachments/assets/4c722b97-79d9-4281-8451-b31eb3393c3a" width="240"></a></td>
  </tr>
  <tr>
    <td style="text-align:center;">A group of forest elves dance around glowing mushrooms, their costumes and movements full of magic and vitality.</td>
    <td style="text-align:center;">cat joins the celebration and dances with the elves, the picture is full of joy and freedom.</td>
    <td style="text-align:center;">A wise old owl reveals the secret power of the moon to cat and the light of the moon in the picture becomes brighter.</td>
    <td style="text-align:center;">cat closes her eyes in the moonlight, puts her hands together, and makes a wish, surrounded by the light of stars and the moon.</td>
    <td style="text-align:center;">cat feels the surge of power, and her eyes become more determined.</td>
  </tr>
</table>



## ⏰ TODOs
- [ ] More demo and applications
- [ ] More functionalities such as control modules. (Suggestions are welcome!)


## 🔆 Information

### Code Structure
```
VideoTuna/
    ├── assets
    ├── checkpoints  # put model checkpoints here
    ├── configs      # model and experimental configs
    ├── data         # data processing scripts and dataset files
    ├── docs         # documentations
    ├── eval         # evaluation scripts
    ├── inputs       # input examples for testing 
    ├── scripts      # train and inference python scripts
    ├── shsripts     # train and inference shell scripts
    ├── src          # model-related source code
    ├── tests        # testing scripts
    ├── tools        # some tool scripts
```


### Supported Models

|T2V-Models|HxWxL|Checkpoints|
|:---------|:---------|:--------|
|CogVideoX-2B|720x480, 6s|[Hugging Face](https://huggingface.co/THUDM/CogVideoX-2b)
|CogVideoX-5B|720x480, 6s|[Hugging Face](https://huggingface.co/THUDM/CogVideoX-5b)
|Open-Sora 1.0|512×512x16|[Hugging Face](https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-HQ-16x512x512.pth)
|Open-Sora 1.0|256×256x16|[Hugging Face](https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-HQ-16x256x256.pth)
|Open-Sora 1.0|256×256x16|[Hugging Face](https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-16x256x256.pth)
|VideoCrafter2|320x512x16|[Hugging Face](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt)
|VideoCrafter1|576x1024x16|[Hugging Face](https://huggingface.co/VideoCrafter/Text2Video-1024/blob/main/model.ckpt)
|VideoCrafter1|320x512x16|[Hugging Face](https://huggingface.co/VideoCrafter/Text2Video-512/blob/main/model.ckpt)

|I2V-Models|HxWxL|Checkpoints|
|:---------|:---------|:--------|
|CogVideoX-5B-I2V|720x480, 6s|[Hugging Face](https://huggingface.co/THUDM/CogVideoX-5b-I2V)
|DynamiCrafter|576x1024x16|[Hugging Face](https://huggingface.co/Doubiiu/DynamiCrafter_1024/blob/main/model.ckpt)|
|VideoCrafter1|320x512x16|[Hugging Face](https://huggingface.co/VideoCrafter/Image2Video-512/blob/main/model.ckpt)|

* Note: H: height; W: width; L: length
<!-- |Open-Sora 1.2|240p to 720p, 2~16s|[STDIT](https://huggingface.co/hpcai-tech/OpenSora-STDiT-v3), [VAE](https://huggingface.co/hpcai-tech/OpenSora-VAE-v1.2) -->
<!-- |Open-Sora 1.1|144p & 240p & 480p, 0~15s|[Stage 2](https://huggingface.co/hpcai-tech/OpenSora-STDiT-v2-stage2) -->
<!-- |Open-Sora 1.1|144p to 720p, 0~15s|[Stage 3](https://huggingface.co/hpcai-tech/OpenSora-STDiT-v2-stage3) -->

Please check [docs/CHECKPOINTS.md](https://github.com/VideoVerses/VideoTuna/blob/main/docs/CHECKPOINTS.md) to download all the model checkpoints.

## 🔆 Get started

### 1.Prepare environment
```
conda create --name videotuna python=3.10 -y
conda activate videotuna
pip install -U poetry pip
poetry config virtualenvs.create false
poetry install
pip install optimum-quanto==0.2.1
pip install -r requirements.txt
git clone https://github.com/JingyeChen/SwissArmyTransformer
pip install -e SwissArmyTransformer/
rm -rf SwissArmyTransformer
git clone https://github.com/tgxs002/HPSv2.git
cd ./HPSv2
pip install -e .
cd ..
```

### 2.Prepare checkpoints

Please follow [docs/CHECKPOINTS.md](https://github.com/VideoVerses/VideoTuna/blob/main/docs/CHECKPOINTS.md) to download model checkpoints.  
After downloading, the model checkpoints should be placed as [Checkpoint Structure](https://github.com/VideoVerses/VideoTuna/blob/main/docs/CHECKPOINTS.md#checkpoint-orgnization-structure).

### 3.Inference state-of-the-art T2V/I2V/T2I models

- Inference many T2V models **in one command**: `bash tools/video_comparison/compare.sh`


<!-- |Task|Commands|
|:---------|:---------|
|T2V|`bash tools/video_comparison/compare.sh`|
|I2V|`TODO`|
 -->


- Inference one specific model:

Task|Models|Commands|
|:---------|:---------|:---------|
|T2V|CogvideoX|`bash shscripts/inference_cogVideo_diffusers.sh`|
|T2V|Open Sora V1.0|`bash shscripts/inference_opensora_v10_16x256x256.sh`|
|T2V|VideoCrafter-V2-320x512|`bash shscripts/inference_vc2_t2v_320x512.sh`|
|T2V|VideoCrafter-V1-576x1024|`bash shscripts/inference_vc1_t2v_576x1024.sh`|
|I2V|DynamiCrafter|`bash shscripts/inference_dc_i2v_576x1024.sh`|
|I2V|VideoCrafter|`bash shscripts/inference_vc1_i2v_320x512.sh`|
|T2I|Flux|`bash shscripts/inference_flux.sh`|


### 4. Finetune T2V models
#### Lora finetuning

We support lora finetuning to make the model to learn new concepts/characters/styles.   
- Example config file: `configs/001_videocrafter2/vc2_t2v_lora.yaml`  
- Training lora based on VideoCrafter2: `bash shscripts/train_videocrafter_lora.sh`  
- Inference the trained models: `bash shscripts/inference_vc2_t2v_320x512_lora.sh`   

<!-- Please check [configs/train/003_vc2_lora_ft/README.md](configs/train/003_vc2_lora_ft/README.md) for details.    -->
<!-- 

(1) Prepare data


(2) Finetune  
```
bash configs/train/000_videocrafter2ft/run.sh
``` -->

#### Finetuning for enhanced langugage understanding


### 5. Evaluation
We support VBench evaluation to evaluate the T2V generation performance. 
Please check [eval/README.md](docs/evaluation.md) for details.

### 6. Alignment
We support video alignment post-training to align human perference for video diffusion models. Please check [configs/train/004_rlhf_vc2/README.md](configs/train/004_rlhf_vc2/README.md) for details.


## Acknowledgement
We thank the following repos for sharing their awesome models and codes!
* [VideoCrafter2](https://github.com/AILab-CVC/VideoCrafter): Overcoming Data Limitations for High-Quality Video Diffusion Models
* [VideoCrafter1](https://github.com/AILab-CVC/VideoCrafter): Open Diffusion Models for High-Quality Video Generation
* [DynamiCrafter](https://github.com/Doubiiu/DynamiCrafter): Animating Open-domain Images with Video Diffusion Priors
* [Open-Sora](https://github.com/hpcaitech/Open-Sora): Democratizing Efficient Video Production for All
* [CogVideoX](https://github.com/THUDM/CogVideo): Text-to-Video Diffusion Models with An Expert Transformer
* [VADER](https://github.com/mihirp1998/VADER): Video Diffusion Alignment via Reward Gradients
* [VBench](https://github.com/Vchitect/VBench): Comprehensive Benchmark Suite for Video Generative Models
* [Flux](https://github.com/black-forest-labs/flux): Text-to-image models from Black Forest Labs.
* [SimpleTuner](https://github.com/bghira/SimpleTuner): A fine-tuning kit for text-to-image generation.




## Some Resources
* [LLMs-Meet-MM-Generation](https://github.com/YingqingHe/Awesome-LLMs-meet-Multimodal-Generation): A paper collection of utilizing LLMs for multimodal generation (image, video, 3D and audio).
* [MMTrail](https://mattie-e.github.io/MMTrail/): A multimodal trailer video dataset with language and music descriptions.
* [Seeing-and-Hearing](https://github.com/yzxing87/Seeing-and-Hearing): A versatile framework for Joint VA generation, V2A, A2V, and I2A.
* [Self-Cascade](https://github.com/GuoLanqing/Self-Cascade): A Self-Cascade model for higher-resolution image and video generation.
* [ScaleCrafter](https://github.com/YingqingHe/ScaleCrafter) and [HiPrompt](https://liuxinyv.github.io/HiPrompt/): Free method for higher-resolution image and video generation.
* [FreeTraj](https://github.com/arthur-qiu/FreeTraj) and [FreeNoise](https://github.com/AILab-CVC/FreeNoise): Free method for video trajectory control and longer-video generation.
* [Follow-Your-Emoji](https://github.com/mayuelala/FollowYourEmoji), [Follow-Your-Click](https://github.com/mayuelala/FollowYourClick), and [Follow-Your-Pose](https://follow-your-pose.github.io/): Follow family for controllable video generation.
* [Animate-A-Story](https://github.com/AILab-CVC/Animate-A-Story): A framework for storytelling video generation.
* [LVDM](https://github.com/YingqingHe/LVDM): Latent Video Diffusion Model for long video generation and text-to-video generation.



## 🍻 Contributors

<a href="https://github.com/VideoVerses/VideoTuna/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=VideoVerses/VideoTuna" />
</a>

## 📋 License
Please follow [CC-BY-NC-ND](./LICENSE). If you want a license authorization, please contact yhebm@connect.ust.hk and yxingag@connect.ust.hk.

## 😊 Citation
```
To be updated...
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=VideoVerses/VideoTuna&type=Date)](https://star-history.com/#VideoVerses/VideoTuna&Date)
