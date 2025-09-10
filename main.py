import base64
import datetime
import glob
import io
import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from tkinter import filedialog, Tk

import gradio as gr
import httpx
import numpy as np
from PIL import Image
from gradio import Warning
from openai import OpenAI, DefaultHttpxClient

MAX_IMAGE_WIDTH = 2048
IMAGE_FORMAT = "JPEG"

# Load configuration with fallback defaults
def load_config():
    default_config = {
        "prompts": {
            "default": "What's in this image? Provide a description without filler text like: The image depicts...",
            "prefix": "",
            "postfix": ""
        },
        "api_settings": {
            "detail_level": "auto",
            "max_tokens": 300
        },
        "paths": {
            "default_dataset_folder": "/home/user/dataset"
        },
        "processing": {
            "default_workers": 2,
            "min_workers": 1,
            "max_workers": 4
        },
        "ui_settings": {
            "gallery_columns": 5,
            "gallery_rows": 2,
            "preview_height": "420px"
        }
    }
    
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        return config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Config file not found or invalid, using defaults: {e}")
        return default_config

# Load configuration
config = load_config()


def load_images_and_text(folder_path):
    image_files = glob.glob(os.path.join(folder_path, "*.jpg")) + glob.glob(
        os.path.join(folder_path, "*.png")
    )
    image_files.sort()
    images = []
    texts = []
    for img_path in image_files:
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()
            images.append(img_path)
            texts.append(text)
    return images, texts


def save_edited_text(image_path, new_text):
    txt_path = os.path.splitext(image_path)[0] + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(new_text)
    return f"Saved changes for {os.path.basename(image_path)}"


def generate_description(api_key, image, prompt, detail, max_tokens):
    try:
        img = (
            Image.fromarray(image)
            if isinstance(image, np.ndarray)
            else Image.open(image)
        )
        img = scale_image(img)

        buffered = io.BytesIO()
        img.save(buffered, format=IMAGE_FORMAT)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Configure HTTP client with proper proxy support and timeout
        http_client_kwargs = {}
        
        # Check for proxy environment variables
        proxy_url = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy') or os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
        if proxy_url:
            http_client_kwargs['proxy'] = proxy_url
        
        # Set timeout configuration
        timeout = httpx.Timeout(60.0, read=30.0, write=10.0, connect=5.0)
        http_client_kwargs['timeout'] = timeout
        
        # Create HTTP client if we have custom configuration
        if http_client_kwargs:
            http_client = DefaultHttpxClient(**http_client_kwargs)
            client = OpenAI(api_key=api_key, http_client=http_client)
        else:
            client = OpenAI(api_key=api_key, timeout=60.0)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}",
                                "detail": detail,
                            },
                        },
                    ],
                }
            ],
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content

    except Exception as e:
        error_msg = f"OpenAI API Error: {str(e)}"
        with open("error_log.txt", "a") as log_file:
            log_file.write(f"{datetime.datetime.now()}: {error_msg}\n")
            log_file.write(traceback.format_exc() + "\n")
        return error_msg


history = []
columns = ["Time", "Prompt", "GPT4-Vision Caption"]


def clear_fields():
    global history
    history = []
    return "", []


def update_history(prompt, response):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    history.append(
        {"Time": timestamp, "Prompt": prompt, "GPT4-Vision Caption": response}
    )
    return [[entry[column] for column in columns] for entry in history]


def scale_image(img):
    if img.width > MAX_IMAGE_WIDTH:
        ratio = MAX_IMAGE_WIDTH / img.width
        new_height = int(img.height * ratio)
        return img.resize((MAX_IMAGE_WIDTH, new_height), Image.Resampling.LANCZOS)
    return img


def get_dir(file_path):
    dir_path, file_name = os.path.split(file_path)
    return dir_path, file_name


def get_folder_path(folder_path=""):
    current_folder_path = folder_path

    initial_dir, initial_file = get_dir(folder_path)

    root = Tk()
    root.wm_attributes("-topmost", 1)
    root.withdraw()

    if sys.platform == "darwin":
        root.call("wm", "attributes", ".", "-topmost", True)

    folder_path = filedialog.askdirectory(initialdir=initial_dir)
    root.destroy()

    if folder_path == "":
        folder_path = current_folder_path

    return folder_path


is_processing = True


def process_folder(
    api_key,
    folder_path,
    prompt,
    detail,
    max_tokens,
    pre_prompt="",
    post_prompt="",
    progress=gr.Progress(),
    num_workers=4,
):
    global is_processing
    is_processing = True

    if not os.path.isdir(folder_path):
        return f"No such directory: {folder_path}"

    file_list = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    progress(0)

    def process_file(file):
        global is_processing
        if not is_processing:
            return "Processing canceled by user"

        image_path = os.path.join(folder_path, file)
        txt_path = os.path.join(folder_path, os.path.splitext(file)[0] + ".txt")

        # Check if the *.txt file already exists
        if os.path.exists(txt_path):
            print(f"File {txt_path} already exists. Skipping.")
            return  # Exit the function

        description = generate_description(
            api_key, image_path, prompt, detail, max_tokens
        )

        # If file doesn't exist, write to it
        with open(txt_path, "w", encoding="utf-8") as f:
            # Build the final text with conditional prefix formatting
            final_text = description
            if pre_prompt.strip():
                final_text = pre_prompt + ", " + final_text
            if post_prompt.strip():
                final_text = final_text + " " + post_prompt
            f.write(final_text)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for i, _ in enumerate(executor.map(process_file, file_list), 1):
            progress((i, len(file_list)))
            if not is_processing:
                break

    is_processing = False
    return f"Processed {len(file_list)} images in folder {folder_path}"


with gr.Blocks() as app:
    with gr.Accordion("Configuration", open=False):
        api_key_input = gr.Textbox(
            label="OpenAI API Key",
            placeholder="Enter your API key here",
            type="password",
            info="The OpenAI API is rate limited to 20 requests per second. A big dataset can take a long time to tag.",
        )
    with gr.Tab("Prompt Engineering"):
        image_input = gr.Image(label="Upload Image")
        with gr.Row():
            prompt_input = gr.Textbox(
                scale=6,
                label="Prompt",
                value=config["prompts"]["default"],
                interactive=True,
            )
            detail_level = gr.Radio(
                ["high", "low", "auto"], scale=2, label="Detail", value=config["api_settings"]["detail_level"]
            )
            max_tokens_input = gr.Number(scale=0, value=config["api_settings"]["max_tokens"], label="Max Tokens")
            submit_button = gr.Button("Generate Caption")
        output = gr.Textbox(label="GPT4-Vision Caption")
        history_table = gr.Dataframe(headers=columns)
        clear_button = gr.Button("Clear")
        clear_button.click(clear_fields, inputs=[], outputs=[output, history_table])

    with gr.Tab("GPT4-Vision Tagging"):
        with gr.Row():
            folder_path_dataset = gr.Textbox(
                scale=8,
                label="Dataset Folder Path",
                placeholder=config["paths"]["default_dataset_folder"],
                interactive=True,
                info="The folder path select button is a bit of hack if it doesn't work you can just copy and paste the path to your dataset.",
            )
            folder_button = gr.Button("📂", elem_id="open_folder_small")
            folder_button.click(
                get_folder_path,
                outputs=folder_path_dataset,
                show_progress="hidden",
            )
        with gr.Row():
            prompt_input_dataset = gr.Textbox(
                scale=6,
                label="Prompt",
                value=config["prompts"]["default"],
                interactive=True,
            )
            detail_level_dataset = gr.Radio(
                ["high", "low", "auto"], scale=2, label="Detail", value=config["api_settings"]["detail_level"]
            )
            max_tokens_input_dataset = gr.Number(scale=0, value=config["api_settings"]["max_tokens"], label="Max Tokens")
        with gr.Row():
            pre_prompt_input = gr.Textbox(
                scale=6,
                label="Prefix",
                value=config["prompts"]["prefix"],
                placeholder="(Optional)",
                info="Will be added at the front of the caption.",
                interactive=True,
            )
            post_prompt_input = gr.Textbox(
                scale=6,
                label="Postfix",
                value=config["prompts"]["postfix"],
                placeholder="(Optional)",
                info="Will be added at the end of the caption.",
                interactive=True,
            )
        with gr.Row():
            worker_slider = gr.Slider(
                minimum=config["processing"]["min_workers"],
                maximum=config["processing"]["max_workers"],
                value=config["processing"]["default_workers"],
                step=1,
                scale=2,
                label="Number of Workers",
            )
            submit_button_dataset = gr.Button("Generate Captions", scale=3)
            cancel_button = gr.Button("Cancel", scale=3)
        with gr.Row():
            processing_results_output = gr.Textbox(label="Processing Results")

    with gr.Tab("View and Edit Captions"):
        with gr.Row():
            folder_path_view = gr.Textbox(
                label="Dataset Folder Path", placeholder=config["paths"]["default_dataset_folder"], scale=8
            )
            folder_button = gr.Button("📂", elem_id="open_folder_small", scale=1)

        load_button = gr.Button("Load Images and Captions")

        with gr.Row():
            image_output = gr.Gallery(
                label="Image",
                show_label=False,
                elem_id="preview_gallery",
                columns=1,
                rows=1,
                height=config["ui_settings"]["preview_height"],
                allow_preview=True,
            )
            text_output = gr.Textbox(label="Caption", lines=5, interactive=True)

        save_button = gr.Button("Save Changes")
        with gr.Row():
            prev_button = gr.Button("Previous")
            next_button = gr.Button("Next")
        status_output = gr.Textbox(label="Status")

        gallery = gr.Gallery(
            label="Image Gallery",
            show_label=False,
            elem_id="gallery",
            columns=config["ui_settings"]["gallery_columns"],
            rows=config["ui_settings"]["gallery_rows"],
            height="auto",
            allow_preview=False,
        )

        current_index = gr.State(0)
        images_list = gr.State([])
        texts_list = gr.State([])

        folder_button.click(
            get_folder_path,
            outputs=folder_path_view,
            show_progress="hidden",
        )

        def save_caption(index, images, texts, new_text):
            if 0 <= index < len(images):
                image_path = images[index]
                save_edited_text(image_path, new_text)
                texts[index] = new_text
                return texts, "Changes saved successfully"
            return texts, "Error: Invalid image index"

        save_button.click(
            save_caption,
            inputs=[current_index, images_list, texts_list, text_output],
            outputs=[texts_list, status_output],
        )

        def load_data_and_display_first(folder_path):
            images, texts = load_images_and_text(folder_path)
            if images and texts:
                img_path = images[0]
                txt = texts[0]
                return (
                    0,
                    images,
                    texts,
                    [(img_path, os.path.basename(img_path))],
                    txt,
                    f"Loaded {len(images)} images",
                    [(img, os.path.basename(img)) for img in images],
                )
            return 0, [], [], [], "", "No images found in the specified folder", []

        def update_display(index, images, texts):
            if 0 <= index < len(images):
                img_path = images[index]
                txt = texts[index]
                return [(img_path, os.path.basename(img_path))], txt
            return [], ""

        def nav_previous(current, images, texts):
            new_index = max(0, current - 1)
            preview_gallery, txt = update_display(new_index, images, texts)
            return new_index, preview_gallery, txt

        def nav_next(current, images, texts):
            new_index = min(len(images) - 1, current + 1)
            preview_gallery, txt = update_display(new_index, images, texts)
            return new_index, preview_gallery, txt

        def gallery_select(evt: gr.SelectData, images, texts):
            index = evt.index
            preview_gallery, txt = update_display(index, images, texts)
            return index, preview_gallery, txt

        load_button.click(
            load_data_and_display_first,
            inputs=[folder_path_view],
            outputs=[
                current_index,
                images_list,
                texts_list,
                image_output,
                text_output,
                status_output,
                gallery,
            ],
        )

        prev_button.click(
            nav_previous,
            inputs=[current_index, images_list, texts_list],
            outputs=[current_index, image_output, text_output],
        )

        next_button.click(
            nav_next,
            inputs=[current_index, images_list, texts_list],
            outputs=[current_index, image_output, text_output],
        )

        gallery.select(
            gallery_select,
            inputs=[images_list, texts_list],
            outputs=[current_index, image_output, text_output],
        )

    def cancel_processing():
        global is_processing
        is_processing = False
        return "Processing canceled"

    cancel_button.click(
        cancel_processing, inputs=[], outputs=[processing_results_output]
    )

    def on_click(api_key, image, prompt, detail, max_tokens):
        if not api_key.strip():
            raise Warning("Please enter your OpenAI API key.")

        if image is None:
            raise Warning("Please upload an image.")

        description = generate_description(api_key, image, prompt, detail, max_tokens)
        new_history = update_history(prompt, description)
        return description, new_history

    submit_button.click(
        on_click,
        inputs=[
            api_key_input,
            image_input,
            prompt_input,
            detail_level,
            max_tokens_input,
        ],
        outputs=[output, history_table],
    )

    def on_click_folder(
        api_key,
        folder_path,
        prompt,
        detail,
        max_tokens,
        pre_prompt,
        post_prompt,
        worker_slider_local,
    ):
        if not api_key.strip():
            raise Warning("Please enter your OpenAI API key.")

        if not folder_path.strip():
            raise Warning("Please enter the folder path.")

        result = process_folder(
            api_key,
            folder_path,
            prompt,
            detail,
            max_tokens,
            pre_prompt,
            post_prompt,
            num_workers=worker_slider_local,
        )
        return result

    submit_button_dataset.click(
        on_click_folder,
        inputs=[
            api_key_input,
            folder_path_dataset,
            prompt_input_dataset,
            detail_level_dataset,
            max_tokens_input_dataset,
            pre_prompt_input,
            post_prompt_input,
            worker_slider,
        ],
        outputs=[processing_results_output],
    )

app.launch(share=True)
