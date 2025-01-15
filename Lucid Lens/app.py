from flask import Flask, render_template, jsonify, request, send_from_directory
from diffusers import StableDiffusionPipeline
from flask_ngrok import run_with_ngrok
from PIL import Image
import os


app = Flask(__name__)
OUTPUT_DIR = "static/generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/generate', methods=['POST'])
def generate_image():
    data = request.json
    prompt = data.get('prompt', '')

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        safety_checker = None,
        requires_safety_checker = False
    )

    pipe.to("cpu")

    image = pipe(prompt, height=256, width=256).images[0]

    
    image_path = os.path.join(OUTPUT_DIR, "visual.png")
    image.save(image_path)

    # Return the URL to the saved image
    return jsonify({"image_url": f"/static/generated_images/visual.png"})


@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory("static", path)

if __name__ == '__main__':
    app.run(debug=True)


