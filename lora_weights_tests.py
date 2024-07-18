from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import os
from PIL import Image
from utils.image_processing import generate_caption

# diff_path = "checkpoints/diffusers/stabilityai/stable-diffusion-xl-base-1.0"
diff_path = "checkpoints/diffusers/cagliostrolab/animagine-xl-3.1"
# diff_path = "checkpoints/diffusers/runwayml/stable-diffusion-v1-5"
lora_path = "checkpoints/outputs/LoRAs/07-05/test4"
weight_name = "pytorch_lora_weights.safetensors"

output_path = "outputs/weights_tests/07-08/animegine/test2"

ns = 20  # Num inference steps

prompt1 = "denip style, best quality, high quality"
f_name1 = prompt1.replace(" ", "_") + ".png"

prompt4 = "A group of characters laughing together in a cafe"
f_name4 = prompt4.replace(" ", "_") + ".png"

prompt5 = "denip style, high quality,  A group of characters laughing together in a cafe"
f_name5 = prompt5.replace(" ", "_") + ".png"
#
prompts2 = {
    "Japan_AoT_S4E4_shot2": "A group of man standing in the city, and a girl looks at them angerly.",
    "Japan_AoT_S4E16_shot2": "A girl looking scared sitting in a dark room, with wooden berrals behind her.",
    "Japan_AoT_S4E16_shot1": "A girl looking scared sitting in a dark room, with wooden berrals behind her.",
    "Japan_AoT_S4E4_shot1": "A group of people in the dessert, camera looks from above."
}


prompts3 = {
    "Japan_AoT_S4E4_shot2": "high quality, denip style, A group of man standing in the city, and a girl looks at them angerly",
    "Japan_AoT_S4E16_shot2": "high quality, denip style, A girl sitting in a dark room, with wooden berrals behind her",
    "Japan_AoT_S4E16_shot1": "high quality, denip style, A girl sitting in a dark room, with wooden berrals behind her",
    "Japan_AoT_S4E4_shot1": "high quality, denip style, A group of people in the dessert, camera looks from above"
}

s1_prompt2 = "A girl sitting in a dark room, with wooden berrals behind her"
s1_f_name2 = s1_prompt2.replace(" ", "_") + ".png"
s1_prompt3 = "high quality, denip style, A girl sitting in a dark room, with wooden berrals behind her"
s1_f_name3 = s1_prompt3.replace(" ", "_") + ".png"

s2_prompt2 = "A group of man standing in the city, and a girl looks at them angerly"
s2_f_name2 = s2_prompt2.replace(" ", "_") + ".png"
s2_prompt3 = "A group of man standing in the city, and a girl looks at them angerly, denip style, high quality"
s2_f_name3 = s2_prompt3.replace(" ", "_") + ".png"

s3_prompt2 = "two exhausted children stand in the dessert and two more characters look at them"
s3_f_name2 = s3_prompt2.replace(" ", "_") + ".png"
s3_prompt3 = s3_prompt2 + ", denip style, high quality"
s3_f_name3 = s3_prompt3.replace(" ", "_") + ".png"

# animeinterp_output_path = "outputs/avi_full_results/Disney_v4_21_044474_s1/frame2.png"

def get_image_path(scene):
    return f"outputs/test_aot_DiffimeInterp_latents/{scene}/frame2.png"


if __name__ == '__main__':
    pipeline = AutoPipelineForText2Image.from_pretrained(diff_path).to("cuda")
    # pipeline = AutoPipelineForText2Image.from_pretrained(diff_path).to("cuda")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # root_name = "Disney_v4_21_044474_s1"
    # root_path = os.path.join(lora_path, root_name)
    # image = Image.open(animeinterp_output_path)
    # image = Image.open("temp_img.png")
    for root_name in os.listdir(lora_path):
    # for i in range(0, 1):
        print(f"Processing {root_name}")
        root_path = os.path.join(lora_path, root_name)
        image = Image.open(get_image_path(root_name))
        folder_list = os.listdir(root_path)
        folder_list.sort(reverse=True)
        for idx, folder_name in enumerate(folder_list):
            folder_path = os.path.join(root_path, folder_name)
            if os.path.isfile(folder_path):
                continue

            # Check if it's a directory
            print(f"Processing {folder_name}")

            # Construct the path to the LoRA weights file
            lora_weights_path = os.path.join(folder_path, "pytorch_lora_weights.safetensors")

            # Check if the LoRA weights file exists
            if os.path.exists(lora_weights_path):
            # if False:
                try:
                    # Load the LoRA weights
                    pipeline.load_lora_weights(lora_weights_path)
                    print(f"  - Loaded LoRA weights from {lora_weights_path}")
                except Exception as e:
                    print(f"Error processing {folder_name}: {str(e)}")
            else:
                print(f"No LoRA weights found in {folder_name}")



            # Here you can add any processing or generation steps using the loaded weights
            # For example:
            # result = pipeline("prompt", num_inference_steps=30, guidance_scale=7.5)
            output_root_folder = os.path.join(output_path, root_name)
            if not os.path.exists(output_root_folder):
                os.makedirs(output_root_folder)
            output_folder = os.path.join(output_root_folder, folder_name)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            image.save(os.path.join(output_folder, "seed_image.png"))
            
            img1 = pipeline(prompt1, image=image, num_inference_steps=ns, negative_prompt="photorealistic. disfigure. bad anatomy. blurry. poorly drawn face. poor facial details").images[0]
            # img1 = pipeline(prompt1, num_inference_steps=50, negative_prompt="photorealistic. disfigure. bad anatomy. blurry. poorly drawn face. poor facial details").images[0]
            img1.save(os.path.join(output_folder, f_name1))

            img2 = pipeline(prompts2[root_name], image=image, num_inference_steps=ns, negative_prompt="disfigure. bad anatomy. blurry. poorly drawn face. poor facial details. hyperrealistic. 3D").images[0]
            # img2 = pipeline(prompts2[root_name], negative_prompt="disfigure. bad anatomy. blurry. poorly drawn face. poor facial details. hyperrealistic. 3D").images[0]
            img2.save(os.path.join(output_folder, (prompts2[root_name].replace(" ", "_") + ".png")))

            img3 = pipeline(prompts3[root_name], image=image, num_inference_steps=ns, negative_prompt="disfigure. bad anatomy. blurry. poorly drawn face. poor facial details. hyperrealisric. 3D").images[0]
            # img3 = pipeline(prompts3[root_name], negative_prompt="disfigure. bad anatomy. blurry. poorly drawn face. poor facial details. hyperrealisric. 3D").images[0]
            img3.save(os.path.join(output_folder, (prompts3[root_name].replace(" ", "_") + ".png")))

            # img2 = pipeline(s3_prompt2, image=image, num_inference_steps=50, negative_prompt="disfigure. bad anatomy. blurry. poorly drawn face. poor facial details").images[0]
            # img2 = pipeline(s3_prompt2, num_inference_steps=50, negative_prompt="disfigure. bad anatomy. blurry. poorly drawn face. poor facial details").images[0]
            # img2.save(os.path.join(output_folder, s3_f_name2))
            # img3 = pipeline(s3_prompt3, image=image, num_inference_steps=50, negative_prompt="disfigure. bad anatomy. blurry. poorly drawn face. poor facial details").images[0]
            # img3 = pipeline(s3_prompt3, num_inference_steps=50, negative_prompt="disfigure. bad anatomy. blurry. poorly drawn face. poor facial details").images[0]
            # img3.save(os.path.join(output_folder, s3_f_name3))

            # img4 = pipeline(prompt4, image=image, num_inference_steps=50, negative_prompt="disfigure. bad anatomy. blurry. poorly drawn face. poor facial details").images[0]
            # # img4 = pipeline(prompt4, num_inference_steps=50, negative_prompt="disfigure. bad anatomy. blurry. poorly drawn face. poor facial details").images[0]
            # img4.save(os.path.join(output_folder, f_name4))

            img5 = pipeline(prompt5, image=image, num_inference_steps=ns, negative_prompt="disfigure. bad anatomy. blurry. poorly drawn face. poor facial details").images[0]
            # img5 = pipeline(prompt5, num_inference_steps=50, negative_prompt="disfigure. bad anatomy. blurry. poorly drawn face. poor facial details").images[0]
            img5.save(os.path.join(output_folder, f_name5))

            img6 = pipeline("", image=image, num_inference_steps=ns, negative_prompt="disfigure. blurry. poorly drawn face. poor facial details").images[0]
            # img6 = pipeline("", num_inference_steps=50, negative_prompt="disfigure. blurry. poorly drawn face. poor facial details").images[0]
            img6.save(os.path.join(output_folder, "no_prompt.png"))


            # Unload the LoRA weights
            pipeline.unload_lora_weights()
            print(f"  - Unloaded LoRA weights for {folder_name}")
        print()

print("Finished processing all folders")



# pipeline.load_lora_weights(lora_weights, weight_name=weight_name, adapter_name="lora")
# pipeline.set_adapters("lora")
# pipeline_img.load_lora_weights(lora_weights, weight_name=weight_name, adapter_name="lora")
# pipeline_img.set_adapters("lora")


# anint_output = Image.open(animeinterp_output_path)
# anint_output = anint_output.resize((512, 512))
# anint_output.save(os.path.join(output_path, "frame2.png"))
#
# img4 = pipeline_img(prompt1, image=anint_output).images[0]
# img4.save(os.path.join(output_path, ("img_" + f_name1)))
#
# img5 = pipeline_img(prompt2, image=anint_output).images[0]
# img5.save(os.path.join(output_path, ("img_" + f_name2)))
#
# img6 = pipeline_img(prompt3, image=anint_output).images[0]
# img6.save(os.path.join(output_path, ("img_" + f_name3)))
    